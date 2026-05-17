#!/usr/bin/env python3
"""
handmouse_pcV2.py — El Hareketi Mouse Kontrolü (Sistem Tepsisi Servisi)
=======================================================================
handmouse_pc.py'nin servis versiyonu.
Sistem tepsisinde (gizli simgeler menüsünde) çalışır; arka planda sessizce
el hareketlerini dinler. OpenCV penceresi açmaz.

Tepsideki simgeye sağ tıklayarak:
  • Kamera seç  (Laptop kamera / WiFi telefon kamerası)
  • Yayını başlat / durdur
  • IP ve Port ayarla (Bağlantı Ayarları dialog)
  • Uygulamadan çık

KURULUM:
    pip install mediapipe opencv-python pyautogui numpy pystray pillow

ÇALIŞTIRMA:
    python handmouse_pcV2.py

EL HAREKETLERİ:
    ☝  Tek işaret parmağı  → Cursor hareketi (avuç merkezi)
    🤏 Pinch (baş+işaret)  → Sol tıklama
    ✌  İki parmak          → Kaydırma (scroll)
    🖖  Üç parmak          → Sağ tıklama
    ✊  Yumruk              → Sürükleme (drag)
    🤘 İşaret+serçe kaydır → Pencere geçişi
"""

# ── Uyarıları bastır (MediaPipe / TensorFlow) ────────────────────────────────
import os
os.environ.setdefault("GLOG_minloglevel",     "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui
import numpy as np
import time
import math
import threading
import queue
import urllib.request
from collections import deque
from enum import Enum, auto

# ── Zorunlu bağımlılıklar ───────────────────────────────────────────────────
try:
    import pystray
    from pystray import MenuItem as Item
    from PIL import Image, ImageDraw
except ImportError:
    print("[HATA] pystray ve Pillow gerekli:")
    print("       pip install pystray pillow")
    sys.exit(1)

try:
    import tkinter as tk
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ═══════════════════════════════════════════════════════════════════════════════
#  AYARLAR
# ═══════════════════════════════════════════════════════════════════════════════

CAM_WIDTH           = 640
CAM_HEIGHT          = 480

OEF_MINCUTOFF       = 0.5     # One Euro Filter — düşük: sakin = titremesiz
OEF_BETA            = 0.8     # One Euro Filter — yüksek: hareket = gecikmesiz
CURSOR_DEAD_ZONE_PX = 4       # Mikro titremeleri yut (piksel)
SMOOTH_ALPHA        = 0.50    # EMA önceden yumuşatma katsayısı
PINCH_THRESHOLD     = 0.055   # Pinch (sol tık) mesafe eşiği (0–1)
CLICK_COOLDOWN_MS   = 350     # Çift tetiklenme önleme (ms)
SCROLL_SENSITIVITY  = 720     # Scroll hızı
ACTIVE_X_MIN        = 0.10    # Kameradaki aktif bölge sınırları
ACTIVE_X_MAX        = 0.90
ACTIVE_Y_MIN        = 0.08
ACTIVE_Y_MAX        = 0.92
TARGET_LOOP_FPS     = 60      # Ana döngü üst FPS sınırı
SWIPE_THRESHOLD     = 0.10    # Pencere geçişi için min kaydırma mesafesi
SWIPE_COOLDOWN_MS   = 650     # İki swipe arası minimum bekleme (ms)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DOSYASI
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model() -> None:
    """Model dosyası yoksa otomatik indir."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[BİLGİ] '{MODEL_PATH}' bulunamadı, indiriliyor...")
    try:
        def _prog(blk, blk_sz, total):
            pct = min(blk * blk_sz / total * 100, 100) if total > 0 else 0
            print(f"\r  %{pct:.0f}", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_prog)
        print("\n[BİLGİ] Model indirildi.")
    except Exception as e:
        print(f"\n[HATA] İndirme başarısız: {e}")
        raise SystemExit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  ONE EURO FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class _LPF:
    """Dahili düşük geçiren filtre."""
    def __init__(self):            self._y = None
    def __call__(self, x, alpha):
        self._y = x if self._y is None else alpha * x + (1.0 - alpha) * self._y
        return self._y
    def reset(self):               self._y = None
    @property
    def value(self):               return self._y


class OneEuroFilter:
    """
    Uyarlanabilir One Euro Filter.
    El dururken → ağır filtre (titreme yok).
    El hareket ederken → hafif filtre (gecikme yok).
    """
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self._freq = freq; self._mincutoff = mincutoff
        self._beta = beta; self._dcutoff   = dcutoff
        self._x = _LPF(); self._dx = _LPF(); self._last_ts = None

    @staticmethod
    def _alpha(cutoff, freq):
        te  = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, ts=None):
        if ts is not None and self._last_ts is not None:
            dt = ts - self._last_ts
            if dt > 1e-6:
                self._freq = 1.0 / dt
        self._last_ts = ts
        prev = self._x.value if self._x.value is not None else x
        dx   = (x - prev) * self._freq
        edx  = self._dx(dx, self._alpha(self._dcutoff, self._freq))
        cutoff = self._mincutoff + self._beta * abs(edx)
        return self._x(x, self._alpha(cutoff, self._freq))

    def reset(self):
        self._x.reset(); self._dx.reset(); self._last_ts = None

# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE TANIMLARI
# ═══════════════════════════════════════════════════════════════════════════════

class Gesture(Enum):
    NONE        = auto()
    MOVE        = auto()
    LEFT_CLICK  = auto()
    RIGHT_CLICK = auto()
    SCROLL      = auto()
    DRAG        = auto()
    WIN_SWITCH  = auto()

GESTURE_LABELS = {
    Gesture.NONE:        "---",
    Gesture.MOVE:        "HAREKET",
    Gesture.LEFT_CLICK:  "SOL TIKLAMA",
    Gesture.RIGHT_CLICK: "SAG TIKLAMA",
    Gesture.SCROLL:      "SCROLL",
    Gesture.DRAG:        "SURUKLE",
    Gesture.WIN_SWITCH:  "PENCERE GECIS",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  EL TESPİT VE SINIFLANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════

class HandDetector:
    """MediaPipe Tasks API — VIDEO modu ile çalışır."""

    PALM_INDICES = [0, 5, 9, 13, 17]
    CONNECTIONS  = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(self):
        ensure_model()
        base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.70,
        )
        self._lm = mp_vision.HandLandmarker.create_from_options(opts)

    def __del__(self):
        if hasattr(self, "_lm"):
            self._lm.close()

    def detect(self, rgb: np.ndarray, ts_ms: int):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._lm.detect_for_video(img, ts_ms)

    @staticmethod
    def palm_center(lm) -> tuple:
        idxs = HandDetector.PALM_INDICES
        return (sum(lm[i].x for i in idxs) / len(idxs),
                sum(lm[i].y for i in idxs) / len(idxs))

    @staticmethod
    def fingers_up(lm) -> list:
        """[başparmak, işaret, orta, yüzük, serçe] — True = açık"""
        thumb = lm[4].x < lm[3].x   # ayna görüntüsü için
        rest  = [lm[tip].y < lm[pip].y
                 for tip, pip in [(8,6),(12,10),(16,14),(20,18)]]
        return [thumb] + rest

    @staticmethod
    def pinch_dist(lm) -> float:
        dx = lm[4].x - lm[8].x; dy = lm[4].y - lm[8].y
        return math.sqrt(dx*dx + dy*dy)

    def classify(self, lm) -> Gesture:
        ext   = self.fingers_up(lm)
        pinch = self.pinch_dist(lm)
        _, idx, mid, rng, pnk = ext

        if pinch < PINCH_THRESHOLD:                          return Gesture.LEFT_CLICK
        if idx and not mid and not rng and pnk:              return Gesture.WIN_SWITCH
        if idx and mid and rng and not pnk:                  return Gesture.RIGHT_CLICK
        if idx and mid and not rng and not pnk:              return Gesture.SCROLL
        if not idx and not mid and not rng and not pnk:      return Gesture.DRAG
        return Gesture.MOVE

# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD'LER
# ═══════════════════════════════════════════════════════════════════════════════

class CaptureThread(threading.Thread):
    """Kamera okumayı ana döngüden ayırır — en son frame her zaman hazır."""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self._cap = cap; self._lock = threading.Lock()
        self._frame = None; self._fid = 0; self._run = True

    def run(self):
        while self._run:
            ret, fr = self._cap.read()
            if ret and fr is not None:
                with self._lock:
                    self._frame = fr; self._fid += 1

    def get(self):
        with self._lock:
            return self._frame, self._fid

    def stop(self): self._run = False


class DetectorThread(threading.Thread):
    """MediaPipe tespitini arka planda çalıştırır."""
    def __init__(self, detector, t0):
        super().__init__(daemon=True)
        self._det = detector; self._t0 = t0
        self._q   = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._lm  = None; self._seq = 0; self._run = True

    def run(self):
        while self._run:
            try:
                rgb, ts = self._q.get(timeout=0.1)
                res = self._det.detect(rgb, ts)
                lm  = res.hand_landmarks[0] if res.hand_landmarks else None
                with self._lock:
                    self._lm = lm; self._seq += 1
            except queue.Empty:
                continue

    def submit(self, rgb):
        ts = int((time.perf_counter() - self._t0) * 1000)
        try:    self._q.put_nowait((rgb, ts))
        except queue.Full: pass

    def get(self):
        with self._lock:
            return self._lm, self._seq

    def stop(self): self._run = False

# ═══════════════════════════════════════════════════════════════════════════════
#  MOUSE KONTROLÜ
# ═══════════════════════════════════════════════════════════════════════════════

class MouseController:
    def __init__(self):
        self.sw, self.sh = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.0

        self._fx = OneEuroFilter(30.0, OEF_MINCUTOFF, OEF_BETA)
        self._fy = OneEuroFilter(30.0, OEF_MINCUTOFF, OEF_BETA)
        self._last_pos = None
        self._prev_px = self._prev_py = None
        self._last_click = 0.0
        self._dragging   = False
        self._scroll_ref = None
        self._swipe_ref  = None
        self._last_swipe = 0.0

    def _screen_pos(self, nx, ny):
        ts = time.perf_counter()
        cx = (max(ACTIVE_X_MIN, min(ACTIVE_X_MAX, nx)) - ACTIVE_X_MIN) \
             / (ACTIVE_X_MAX - ACTIVE_X_MIN)
        cy = (max(ACTIVE_Y_MIN, min(ACTIVE_Y_MAX, ny)) - ACTIVE_Y_MIN) \
             / (ACTIVE_Y_MAX - ACTIVE_Y_MIN)
        fx = int(self._fx(cx * self.sw, ts))
        fy = int(self._fy(cy * self.sh, ts))
        fx = max(0, min(self.sw - 1, fx))
        fy = max(0, min(self.sh - 1, fy))
        if self._last_pos:
            if (abs(fx - self._last_pos[0]) <= CURSOR_DEAD_ZONE_PX and
                    abs(fy - self._last_pos[1]) <= CURSOR_DEAD_ZONE_PX):
                return None
        self._last_pos = (fx, fy)
        return fx, fy

    def reset(self):
        self._fx.reset(); self._fy.reset()
        self._last_pos = self._prev_px = self._prev_py = None
        self._swipe_ref = None

    def handle(self, gesture: Gesture, lm) -> None:
        now = time.time() * 1000
        px, py = HandDetector.palm_center(lm)

        # EMA önceden yumuşatma
        if self._prev_px is None:
            self._prev_px, self._prev_py = px, py
        px = self._prev_px + SMOOTH_ALPHA * (px - self._prev_px)
        py = self._prev_py + SMOOTH_ALPHA * (py - self._prev_py)
        self._prev_px, self._prev_py = px, py

        pos    = self._screen_pos(px, py)
        cx, cy = pos if pos else (self._last_pos or (self.sw // 2, self.sh // 2))

        if gesture != Gesture.SCROLL:     self._scroll_ref = None
        if gesture != Gesture.WIN_SWITCH: self._swipe_ref  = None
        if gesture != Gesture.DRAG and self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

        if gesture == Gesture.MOVE:
            if pos: pyautogui.moveTo(cx, cy)

        elif gesture == Gesture.LEFT_CLICK:
            if pos: pyautogui.moveTo(cx, cy)
            if now - self._last_click > CLICK_COOLDOWN_MS:
                pyautogui.click()
                self._last_click = now

        elif gesture == Gesture.RIGHT_CLICK:
            if pos: pyautogui.moveTo(cx, cy)
            if now - self._last_click > CLICK_COOLDOWN_MS:
                pyautogui.rightClick()
                self._last_click = now

        elif gesture == Gesture.SCROLL:
            if pos: pyautogui.moveTo(cx, cy)
            py_now = HandDetector.palm_center(lm)[1]
            if self._scroll_ref is not None:
                delta  = py_now - self._scroll_ref
                amount = int(-delta * SCROLL_SENSITIVITY * 10)
                if abs(amount) >= 1:
                    pyautogui.scroll(amount)
            self._scroll_ref = py_now

        elif gesture == Gesture.DRAG:
            if not self._dragging:
                pyautogui.mouseDown(cx, cy)
                self._dragging = True
            elif pos:
                pyautogui.moveTo(cx, cy)

        elif gesture == Gesture.WIN_SWITCH:
            if self._swipe_ref is None:
                self._swipe_ref = (px, py)
            dx = px - self._swipe_ref[0]
            dy = py - self._swipe_ref[1]
            if now - self._last_swipe > SWIPE_COOLDOWN_MS:
                if abs(dx) >= SWIPE_THRESHOLD and abs(dx) >= abs(dy):
                    if dx > 0: pyautogui.hotkey('alt', 'tab')
                    else:      pyautogui.hotkey('alt', 'shift', 'tab')
                    self._last_swipe = now
                    self._swipe_ref  = (px, py)
                elif abs(dy) >= SWIPE_THRESHOLD and abs(dy) > abs(dx):
                    if dy < 0: pyautogui.hotkey('win', 'tab')
                    else:      pyautogui.hotkey('win', 'd')
                    self._last_swipe = now
                    self._swipe_ref  = (px, py)

    def release(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

# ═══════════════════════════════════════════════════════════════════════════════
#  KAMERA TARAMA
# ═══════════════════════════════════════════════════════════════════════════════

def scan_cameras(max_idx: int = 5) -> list:
    """Mevcut laptop kameralarını tara ve index listesi döndür."""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found or [0]

# ═══════════════════════════════════════════════════════════════════════════════
#  EL HAREKETI SERVİSİ
# ═══════════════════════════════════════════════════════════════════════════════

class HandMouseService:
    """
    Arka planda çalışan el hareketi → mouse kontrol servisi.
    Başlat/durdur ve kamera değiştirme thread-safe'dir.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._running     = False
        self._thread      = None
        self.camera_mode  = "laptop"         # "laptop" | "wifi"
        self.camera_index = 0
        self.wifi_ip      = "192.168.1.100"
        self.wifi_port    = "8080"

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Servisi başlat. Zaten çalışıyorsa False döner."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._thread  = threading.Thread(
                target=self._loop, daemon=True, name="HMService"
            )
            self._thread.start()
            return True

    def stop(self):
        """Servisi durdur ve thread'in bitmesini bekle."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
        if thread:
            thread.join(timeout=5.0)
            self._thread = None

    def set_laptop_camera(self, index: int):
        """Laptop kamerasına geç (çalışıyorsa yeniden başlat)."""
        was = self._running
        if was: self.stop()
        self.camera_mode  = "laptop"
        self.camera_index = index
        if was: self.start()

    def set_wifi_camera(self, ip: str, port: str):
        """WiFi kamerasına geç (çalışıyorsa yeniden başlat)."""
        was = self._running
        if was: self.stop()
        self.camera_mode = "wifi"
        self.wifi_ip     = ip.strip()
        self.wifi_port   = port.strip()
        if was: self.start()

    def update_wifi_settings(self, ip: str, port: str):
        """IP/Port güncelle (kamera moduyla birlikte çalışıyorsa yeniden başlat)."""
        self.wifi_ip   = ip.strip()
        self.wifi_port = port.strip()
        if self.camera_mode == "wifi" and self._running:
            self.stop(); self.start()

    # ── İç döngü ─────────────────────────────────────────────────────────────

    def _open_cap(self) -> cv2.VideoCapture:
        if self.camera_mode == "wifi":
            url = f"http://{self.wifi_ip}:{self.wifi_port}/video"
            print(f"[BİLGİ] WiFi kamera bağlanıyor: {url}")
            return cv2.VideoCapture(url)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        return cap

    def _loop(self):
        cap = self._open_cap()
        if not cap.isOpened():
            mode = self.camera_mode
            if mode == "wifi":
                print(f"[HATA] WiFi kamera açılamadı: {self.wifi_ip}:{self.wifi_port}")
            else:
                print(f"[HATA] Laptop kamera #{self.camera_index} açılamadı.")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)

        t0         = time.perf_counter()
        detector   = HandDetector()
        cap_th     = CaptureThread(cap)
        det_th     = DetectorThread(detector, t0)
        controller = MouseController()

        cap_th.start()
        det_th.start()

        # İlk frame'i bekle
        for _ in range(100):
            if cap_th.get()[0] is not None:
                break
            time.sleep(0.05)

        fps_buf  = deque(maxlen=30)
        gesture  = Gesture.NONE
        last_fid = -1
        last_seq = -1
        _dt      = 1.0 / TARGET_LOOP_FPS

        try:
            while self._running:
                t_loop = time.perf_counter()

                frame, fid = cap_th.get()
                if frame is None:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(frame.copy(), 1)   # Ayna görüntü

                if fid != last_fid:
                    det_th.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    last_fid = fid

                lm, seq = det_th.get()
                if seq != last_seq:
                    last_seq = seq
                    if lm is not None:
                        gesture = detector.classify(lm)
                    else:
                        gesture = Gesture.NONE
                        controller.reset()

                if lm is not None:
                    controller.handle(gesture, lm)

                elapsed = time.perf_counter() - t_loop
                fps_buf.append(1.0 / elapsed if elapsed > 1e-6 else TARGET_LOOP_FPS)

                sleep_s = _dt - (time.perf_counter() - t_loop)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        finally:
            controller.release()
            cap_th.stop()
            det_th.stop()
            cap.release()
            print("[BİLGİ] Servis durduruldu.")

# ═══════════════════════════════════════════════════════════════════════════════
#  SİSTEM TEPSİSİ İKONU
# ═══════════════════════════════════════════════════════════════════════════════

def _make_icon(size: int = 64, active: bool = False) -> Image.Image:
    """
    El silüeti şeklinde sistem tepsisi ikonu.
    active=True → mor/parlak (yayın açık)
    active=False → koyu/soluk (yayın kapalı)
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    s   = float(size)

    # Arka plan dairesi
    bg_color = (90, 40, 150, 235) if active else (35, 18, 60, 220)
    d.ellipse([1, 1, size - 2, size - 2], fill=bg_color)

    # El rengi
    hand_color = (220, 150, 255) if active else (155, 90, 200)

    # Avuç tabanı
    d.ellipse(
        [int(s * 0.17), int(s * 0.40), int(s * 0.83), int(s * 0.91)],
        fill=hand_color,
    )

    # Dört parmak: (x1_ratio, y1_ratio, x2_ratio)
    fingers = [
        (0.21, 0.09, 0.37),   # işaret
        (0.40, 0.04, 0.56),   # orta
        (0.59, 0.09, 0.75),   # yüzük
        (0.78, 0.19, 0.91),   # serçe
    ]
    for x1r, y1r, x2r in fingers:
        x1 = int(s * x1r); y1 = int(s * y1r)
        x2 = int(s * x2r); y2 = int(s * 0.50)
        fw = x2 - x1
        d.rectangle([x1, y1 + fw // 2, x2, y2], fill=hand_color)
        d.ellipse([x1, y1, x2, y1 + fw], fill=hand_color)   # yuvarlak uç

    # Baş parmak
    d.ellipse(
        [int(s * 0.03), int(s * 0.44), int(s * 0.23), int(s * 0.71)],
        fill=hand_color,
    )

    # Aktifken küçük yeşil nokta (sağ alt köşe)
    if active:
        r = max(4, int(s * 0.10))
        cx = size - r - 3; cy = size - r - 3
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(80, 230, 120, 255))

    return img

# ═══════════════════════════════════════════════════════════════════════════════
#  BAĞLANTI AYARLARI DIALOG (tkinter)
# ═══════════════════════════════════════════════════════════════════════════════

# Renk şeması — koyu tema
_C = {
    "bg":     "#0d0d1a",
    "card":   "#13131f",
    "accent": "#a064b4",
    "text":   "#dcd8f0",
    "dim":    "#605878",
    "entry":  "#1a1a2e",
    "sep":    "#252535",
    "green":  "#50d080",
    "red":    "#e05060",
}


def _apply_dark_titlebar(hwnd: int):
    """Windows 10/11 — başlık çubuğunu koyu moda al (DWM)."""
    try:
        import ctypes
        val = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(val), ctypes.sizeof(val)
        )
    except Exception:
        pass


def _show_settings_dialog(root: "tk.Tk", service: HandMouseService,
                           on_save=None):
    """
    Bağlantı Ayarları penceresi.
    Ana thread'de (tkinter main loop) çağrılmalıdır.
    """
    dlg = tk.Toplevel(root)
    dlg.title("handmouse v2 — Bağlantı Ayarları")
    dlg.configure(bg=_C["bg"])
    dlg.resizable(False, False)
    dlg.attributes("-topmost", True)

    # Win32 koyu başlık
    dlg.update_idletasks()
    _apply_dark_titlebar(dlg.winfo_id())

    # ── Başlık ──────────────────────────────────────────────────────────────
    hdr = tk.Frame(dlg, bg=_C["card"])
    hdr.pack(fill="x")
    tk.Label(hdr, text="  ⚙  Bağlantı Ayarları", fg=_C["accent"], bg=_C["card"],
             font=("Segoe UI", 12, "bold")).pack(side="left", padx=4, pady=12)
    tk.Frame(dlg, bg=_C["accent"], height=2).pack(fill="x")

    # ── Gövde ───────────────────────────────────────────────────────────────
    body = tk.Frame(dlg, bg=_C["bg"], padx=22, pady=16)
    body.pack(fill="x")

    tk.Label(body, text="WiFi TELEFON KAMERASI", fg=_C["dim"], bg=_C["bg"],
             font=("Segoe UI", 8, "bold")).grid(
                 row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    def _lbl(text, row):
        tk.Label(body, text=text, fg=_C["text"], bg=_C["bg"],
                 font=("Segoe UI", 10)).grid(row=row, column=0, sticky="w",
                                              pady=6, padx=(0, 14))

    def _entry(var, row, width=22):
        e = tk.Entry(body, textvariable=var, bg=_C["entry"], fg=_C["text"],
                     insertbackground=_C["text"], font=("Consolas", 11),
                     bd=0, highlightthickness=1, width=width,
                     highlightbackground=_C["sep"], highlightcolor=_C["accent"])
        e.grid(row=row, column=1, sticky="ew", pady=6)
        return e

    ip_var   = tk.StringVar(value=service.wifi_ip)
    port_var = tk.StringVar(value=service.wifi_port)

    _lbl("IP Adresi:",  1); _entry(ip_var,   1)
    _lbl("Port:",       2); _entry(port_var, 2, width=10)

    # URL önizleme
    url_lbl = tk.Label(body, text="", fg=_C["dim"], bg=_C["bg"],
                        font=("Consolas", 8))
    url_lbl.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _upd(*_):
        url_lbl.config(
            text=f"  URL: http://{ip_var.get()}:{port_var.get()}/video"
        )
    ip_var.trace_add("write", _upd)
    port_var.trace_add("write", _upd)
    _upd()

    # Bilgi notu
    tk.Label(body,
             text="Not: IP Webcam, DroidCam veya benzeri bir uygulama\n"
                  "kullanarak telefonunuzdan yayın yapabilirsiniz.",
             fg=_C["dim"], bg=_C["bg"], font=("Segoe UI", 8),
             justify="left").grid(row=4, column=0, columnspan=2,
                                   sticky="w", pady=(10, 0))

    # ── Ayraç ───────────────────────────────────────────────────────────────
    tk.Frame(dlg, bg=_C["sep"], height=1).pack(fill="x", padx=22, pady=(4, 0))

    # ── Butonlar ─────────────────────────────────────────────────────────────
    btn_row = tk.Frame(dlg, bg=_C["bg"], padx=22, pady=14)
    btn_row.pack(fill="x")

    def _save():
        service.update_wifi_settings(ip_var.get(), port_var.get())
        if on_save:
            on_save()
        dlg.destroy()

    def _cancel():
        dlg.destroy()

    tk.Button(btn_row, text="  Kaydet  ",
              command=_save,
              bg=_C["accent"], fg="#ffffff",
              font=("Segoe UI", 10, "bold"),
              bd=0, pady=8, cursor="hand2",
              activebackground="#b878d0", relief="flat"
              ).pack(side="left")

    tk.Button(btn_row, text="  İptal  ",
              command=_cancel,
              bg=_C["card"], fg=_C["text"],
              font=("Segoe UI", 10),
              bd=0, pady=8, cursor="hand2",
              activebackground=_C["sep"], relief="flat"
              ).pack(side="left", padx=(10, 0))

    dlg.grab_set()
    dlg.focus_set()
    dlg.update_idletasks()

    # Ekranın ortasına yerleştir
    w  = dlg.winfo_reqwidth()
    h  = dlg.winfo_reqheight()
    sw = dlg.winfo_screenwidth()
    sh = dlg.winfo_screenheight()
    dlg.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

# ═══════════════════════════════════════════════════════════════════════════════
#  WINDOWS BAŞLANGIÇ (STARTUP) YÖNETİMİ
# ═══════════════════════════════════════════════════════════════════════════════

_STARTUP_REG_KEY  = r"Software\Microsoft\Windows\CurrentVersion\Run"
_STARTUP_APP_NAME = "handmouse_v2"


def _startup_cmd() -> str:
    """
    Başlangıçta çalıştırılacak komut satırı.
    pythonw.exe kullanır → konsol penceresi açılmaz.
    """
    script = os.path.abspath(__file__)
    # pythonw.exe, python.exe ile aynı Scripts/ klasöründedir
    pythonw = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw):
        pythonw = sys.executable   # fallback: normal python
    # VBS aracısız doğrudan çalışır; tırnak işaretleri şart
    return f'"{pythonw}" "{script}"'


def _is_startup_enabled() -> bool:
    """Registry'de başlangıç girişi var mı?"""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _STARTUP_REG_KEY)
        winreg.QueryValueEx(key, _STARTUP_APP_NAME)
        winreg.CloseKey(key)
        return True
    except (FileNotFoundError, OSError):
        return False


def _set_startup(enable: bool) -> None:
    """Başlangıç girişini ekle veya kaldır."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _STARTUP_REG_KEY,
            0, winreg.KEY_SET_VALUE
        )
        if enable:
            winreg.SetValueEx(key, _STARTUP_APP_NAME, 0,
                              winreg.REG_SZ, _startup_cmd())
            print(f"[BİLGİ] Başlangıca eklendi: {_startup_cmd()}")
        else:
            try:
                winreg.DeleteValue(key, _STARTUP_APP_NAME)
                print("[BİLGİ] Başlangıçtan kaldırıldı.")
            except FileNotFoundError:
                pass
        winreg.CloseKey(key)
    except Exception as e:
        print(f"[HATA] Başlangıç ayarı değiştirilemedi: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEPSI UYGULAMASI
# ═══════════════════════════════════════════════════════════════════════════════

class TrayApp:
    """
    pystray sistem tepsisi uygulaması.
    Mimari:
      • pystray.run_detached() → arka plan thread'i
      • tkinter root (gizli)   → main thread (dialog'lar için gerekli)
      • _actions queue         → pystray → tkinter köprüsü
    """

    def __init__(self):
        self._service  = HandMouseService()
        self._icon: pystray.Icon = None
        self._root: tk.Tk        = None
        self._actions            = queue.Queue()
        self._cameras            = scan_cameras()

    # ── Menü inşası ──────────────────────────────────────────────────────────

    def _build_menu(self) -> pystray.Menu:
        svc = self._service

        # Laptop kamera alt menüsü
        # pystray: action fonksiyonu co_argcount <= 2 olmalı.
        # Default-arg trick (i=idx) co_argcount'u 3'e çıkardığı için
        # fabrika (closure) fonksiyonu kullanıyoruz.
        def _make_laptop_action(i):
            def _action(icon, item):
                svc.set_laptop_camera(i)
                self._refresh_icon()
            return _action

        def _make_laptop_label(i):
            def _label(item):
                mark = "✓" if (svc.camera_mode == "laptop" and svc.camera_index == i) else "   "
                return f"{mark}  Laptop Kamera  {i}"
            return _label

        cam_items = []
        for idx in self._cameras:
            cam_items.append(Item(
                _make_laptop_label(idx),
                _make_laptop_action(idx),
            ))

        cam_items.append(pystray.Menu.SEPARATOR)

        # WiFi kamera seçeneği
        def _wifi_label(item):
            mark = "✓" if svc.camera_mode == "wifi" else "   "
            return f"{mark}  WiFi Telefon Kamerası"

        def _sel_wifi(icon, item):
            svc.set_wifi_camera(svc.wifi_ip, svc.wifi_port)
            self._refresh_icon()

        cam_items.append(Item(_wifi_label, _sel_wifi))

        # Yayın başlat/durdur
        def _toggle(icon, item):
            if svc.is_running:
                svc.stop()
                icon.icon  = _make_icon(active=False)
                icon.title = "handmouse v2  —  Durduruldu"
            else:
                svc.start()
                icon.icon  = _make_icon(active=True)
                icon.title = "handmouse v2  —  Çalışıyor"

        # Ayarlar dialog
        def _open_settings(icon, item):
            self._actions.put(
                lambda: _show_settings_dialog(self._root, svc)
            )

        # Durum satırı (IP / Port bilgisi — dinamik)
        def _conn_info(item):
            if svc.camera_mode == "wifi":
                return f"    IP: {svc.wifi_ip}   Port: {svc.wifi_port}"
            return f"    Laptop Kamera  #{svc.camera_index}"

        # Yayın durumu (dinamik metin)
        def _toggle_label(item):
            return "⏹  Yayını Durdur" if svc.is_running else "▶  Yayını Başlat"

        # Başlangıçta çalıştır (dinamik etiket + toggle)
        def _startup_label(item):
            mark = "✓" if _is_startup_enabled() else "   "
            return f"{mark}  Başlangıçta Çalıştır"

        def _toggle_startup(icon, item):
            _set_startup(not _is_startup_enabled())

        return pystray.Menu(
            # Başlık (devre dışı)
            Item("handmouse  v2", None, enabled=False),
            pystray.Menu.SEPARATOR,

            # Kamera seçimi → alt menü
            Item("Kamera Seç", pystray.Menu(*cam_items)),
            pystray.Menu.SEPARATOR,

            # Yayın başlat / durdur
            Item(_toggle_label, _toggle),
            pystray.Menu.SEPARATOR,

            # Bağlantı bilgisi (devre dışı, yalnızca gösterim)
            Item(_conn_info, None, enabled=False),

            # Ayarlar dialog
            Item("⚙  Bağlantı Ayarları  (IP / Port)…", _open_settings),
            pystray.Menu.SEPARATOR,

            # Başlangıçta otomatik çalıştır
            Item(_startup_label, _toggle_startup),
            pystray.Menu.SEPARATOR,

            # Çıkış
            Item("Çıkış", self._quit),
        )

    def _refresh_icon(self):
        """İkon görselini ve tooltip'i servis durumuna göre güncelle."""
        if self._icon:
            active = self._service.is_running
            self._icon.icon  = _make_icon(active=active)
            self._icon.title = (
                "handmouse v2  —  Çalışıyor"
                if active else
                "handmouse v2  —  Durduruldu"
            )

    # ── Yaşam döngüsü ────────────────────────────────────────────────────────

    def _quit(self, icon=None, item=None):
        """Temiz kapatma."""
        print("[BİLGİ] Kapatılıyor...")
        self._service.stop()
        if self._icon:
            self._icon.stop()
        if self._root:
            self._root.after(0, self._root.quit)

    def _process_actions(self):
        """
        Bekleyen action'ları ana thread'de işle.
        (pystray → tkinter köprüsü — her 150ms çalışır)
        """
        try:
            while True:
                fn = self._actions.get_nowait()
                fn()
        except queue.Empty:
            pass
        if self._root:
            self._root.after(150, self._process_actions)

    # ── Ana giriş noktası ────────────────────────────────────────────────────

    def run(self):
        if not HAS_TK:
            print("[HATA] tkinter bulunamadı; kurulum eksik.")
            sys.exit(1)

        # ── pystray ikonu (arka plan thread'inde çalışır) ────────────────────
        self._icon = pystray.Icon(
            name  = "handmouse",
            icon  = _make_icon(active=False),
            title = "handmouse v2  —  Durduruldu",
            menu  = self._build_menu(),
        )
        self._icon.run_detached()
        print("[BİLGİ] handmouse v2 sistem tepsisinde çalışıyor.")
        print("        Simgeye sağ tıklayarak menüye erişebilirsiniz.")

        # ── tkinter gizli root (main thread — dialog'lar için) ───────────────
        self._root = tk.Tk()
        self._root.withdraw()            # Pencereyi gizle
        self._root.title("handmouse v2") # Görev çubuğunda görünmesin

        # Win32 koyu başlık (gizli pencere için sembolik)
        try:
            self._root.update_idletasks()
            _apply_dark_titlebar(self._root.winfo_id())
        except Exception:
            pass

        # Kapatma protokolü
        self._root.protocol("WM_DELETE_WINDOW", self._quit)

        # Periyodik action pump başlat
        self._root.after(150, self._process_actions)

        # Main loop (bloklar; pystray ve service thread'leri arka planda)
        self._root.mainloop()

        print("[BİLGİ] Tamamlandı.")

# ═══════════════════════════════════════════════════════════════════════════════
#  GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_model()
    app = TrayApp()
    app.run()
