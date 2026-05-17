#!/usr/bin/env python3
"""
handmouse_pc.py — Bilgisayar Kamerasıyla El Hareketi Mouse Kontrolü
=====================================================================
Bilgisayarın dahili (veya harici USB) kamerasını kullanarak el hareketlerini
algılar ve mouse'u gerçek zamanlı olarak kontrol eder.

KURULUM:
    pip install mediapipe opencv-python pyautogui numpy

ÇALIŞTIRMA:
    python handmouse_pc.py              # Dahili kamera (index 0)
    python handmouse_pc.py --cam 1      # İkinci kamera
    python handmouse_pc.py --debug      # Parmak debug bilgisi göster
    python handmouse_pc.py --no-mirror  # Ayna görüntüyü kapat

EL HAREKETLERİ:
    ☝️  Sadece işaret parmağı  → Cursor hareketi (avuç merkezi takip eder)
    🤏  Pinch (baş + işaret)   → Sol tıklama
    ✌️  İki parmak (index+orta) → Kaydırma (scroll)
    🖖  Üç parmak (idx+mid+ring)→ Sağ tıklama
    ✊  Yumruk (4 parmak kapalı)→ Sürükleme (drag)

AYARLAR:
    ACTIVE_X_MIN/MAX, ACTIVE_Y_MIN/MAX → Kamera görüntüsündeki aktif bölge
    OEF_MINCUTOFF, OEF_BETA            → Cursor pürüzsüzlüğü / gecikme dengesi
    PINCH_THRESHOLD                    → Sol tıklama hassasiyeti
"""

import os
# MediaPipe / TensorFlow gereksiz uyarılarını bastır (inference_feedback_manager vb.)
os.environ.setdefault("GLOG_minloglevel",    "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU","1")   # CPU-only → tutarlı performans

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui
import numpy as np
import argparse
import time
import math
import threading
import queue
import urllib.request
from collections import deque
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════════════════════════
#  AYARLAR — ihtiyaca göre buradan değiştir
# ═══════════════════════════════════════════════════════════════════════════════

# Kamera çözünürlüğü
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# One Euro Filter — cursor pürüzsüzlüğü
# OEF_MINCUTOFF : küçük → hareketsizlikte titreme yok / büyük → daha duyarlı
# OEF_BETA      : büyük → hızlı harekette gecikme azalır
OEF_MINCUTOFF   = 0.5
OEF_BETA        = 0.8

# Cursor ölü bölgesi — bu piksel içindeki micro-titremeleri yok say
CURSOR_DEAD_ZONE_PX = 4

# EMA (Üstel Hareketli Ortalama) yumuşatma — dnm.py'den entegre
# Küçük alpha → çok pürüzsüz ama yavaş / Büyük alpha → daha duyarlı ama hafif titrek
# One Euro Filter'dan ÖNCE uygulanır; ham algılama gürültüsünü önceden keser.
SMOOTH_ALPHA = 0.50

# Pinch mesafesi eşiği (normalize, 0–1). Küçült → daha zor tıklama.
PINCH_THRESHOLD   = 0.055

# Tıklama bekleme süresi (ms) — çift tetiklenmeyi önler
CLICK_COOLDOWN_MS = 350

# Scroll hassasiyeti (büyük → hızlı scroll)
SCROLL_SENSITIVITY = 720

# Aktif bölge — elin bu dikdörtgen içinde hareket etmesi yeterli
# Dışarısı yok sayılır; kamera kenarlarındaki kötü algılamayı önler
ACTIVE_X_MIN = 0.10
ACTIVE_X_MAX = 0.90
ACTIVE_Y_MIN = 0.08
ACTIVE_Y_MAX = 0.92

# Ana döngü FPS üst sınırı — gereksiz CPU kullanımını önler
TARGET_LOOP_FPS = 60

# ── Pencere Geçişi (İşaret + Serçe Kaydırma) ────────────────────────────────
# SWIPE_THRESHOLD : kaydırma başlaması için gereken minimum normalize mesafe
# SWIPE_COOLDOWN_MS : iki swipe arası minimum bekleme süresi (ms)
SWIPE_THRESHOLD   = 0.10
SWIPE_COOLDOWN_MS = 650

# ── Kamera Görüntü Modu (M tuşu ile döngüsel değişim) ───────────────────────
OVERLAY_NORMAL      = 0   # Varsayılan: ayrı OpenCV penceresi
OVERLAY_PIP         = 1   # Köşede küçük pencere (Picture-in-Picture)
OVERLAY_TRANSPARENT = 2   # Tam ekran saydam filtre

PIP_W      = 320   # PiP pencere genişliği (piksel)
PIP_H      = 240   # PiP pencere yüksekliği (piksel)
PIP_MARGIN = 12    # Köşeden kenar boşluğu (piksel)
PIP_CORNER = "br"  # "tl" | "tr" | "bl" | "br"

OVERLAY_ALPHA = 0.40   # Saydam mod opaklığı (0=görünmez, 1=tam opak)

WIN_TITLE = "handmouse"

# ── Arayüz Renk Paleti (BGR) ─────────────────────────────────────────────────
UI_BAR_BG    = (20, 14, 12)       # Üst bar arka planı — neredeyse siyah
UI_ACCENT    = (180, 100, 90)      # Vurgu rengi — mavi-mor
UI_TEXT      = (220, 215, 235)     # Ana metin — açık gri-beyaz
UI_TEXT_DIM  = (110, 100, 130)     # Soluk metin
UI_BORDER    = (80, 60, 100)       # İnce çerçeve rengi

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DOSYASI
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model() -> None:
    """Model yoksa otomatik indir."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[BİLGİ] '{MODEL_PATH}' bulunamadı, indiriliyor...")
    try:
        def _progress(blk, blk_sz, total):
            pct = min(blk * blk_sz / total * 100, 100) if total > 0 else 0
            print(f"\r  %{pct:.0f}", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
        print("\n[BİLGİ] Model indirildi.")
    except Exception as e:
        print(f"\n[HATA] İndirme başarısız: {e}")
        print(f"       Manuel indir:\n       {MODEL_URL}")
        raise SystemExit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  ONE EURO FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class _LPF:
    """Dahili düşük geçiren filtre."""
    def __init__(self):
        self._y = None
    def __call__(self, x: float, alpha: float) -> float:
        self._y = x if self._y is None else alpha * x + (1.0 - alpha) * self._y
        return self._y
    def reset(self): self._y = None
    @property
    def value(self): return self._y


class OneEuroFilter:
    """
    Uyarlanabilir One Euro Filter.
    El dururken: ağır filtre → titreme yok.
    El hareket ederken: hafif filtre → gecikme yok.
    """
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self._freq = freq
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._x = _LPF()
        self._dx = _LPF()
        self._last_ts = None

    @staticmethod
    def _alpha(cutoff, freq):
        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, ts: float = None) -> float:
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
    WIN_SWITCH  = auto()   # 🤘 işaret+serçe kaydırma → Alt+Tab / Win+Tab

GESTURE_LABELS = {
    Gesture.NONE:        "---",
    Gesture.MOVE:        "HAREKET",
    Gesture.LEFT_CLICK:  "SOL TIKLAMA",
    Gesture.RIGHT_CLICK: "SAG TIKLAMA",
    Gesture.SCROLL:      "SCROLL",
    Gesture.DRAG:        "SURUKLE",
    Gesture.WIN_SWITCH:  "PENCERE GECIS",
}

GESTURE_COLORS = {
    Gesture.NONE:        (110, 110, 110),
    Gesture.MOVE:        (50,  220,  50),
    Gesture.LEFT_CLICK:  (60,  160, 255),
    Gesture.RIGHT_CLICK: (255, 140,  50),
    Gesture.SCROLL:      (255, 240,  60),
    Gesture.DRAG:        (80,   80, 255),
    Gesture.WIN_SWITCH:  (200,  80, 255),  # mor
}

# ═══════════════════════════════════════════════════════════════════════════════
#  EL TESPİT VE SINIFLANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════

class HandDetector:
    """MediaPipe Tasks API — VIDEO modu ile çalışır."""

    PALM_INDICES = [0, 5, 9, 13, 17]   # bilek + 4 MCP → stabil merkez

    CONNECTIONS = [
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

    # ── Geometri yardımcıları ────────────────────────────────────────────────

    @staticmethod
    def palm_center(lm) -> tuple:
        idxs = HandDetector.PALM_INDICES
        x = sum(lm[i].x for i in idxs) / len(idxs)
        y = sum(lm[i].y for i in idxs) / len(idxs)
        return x, y

    @staticmethod
    def fingers_up(lm) -> list:
        """[başparmak, işaret, orta, yüzük, serçe] — True = açık"""
        thumb = lm[4].x < lm[3].x          # ayna görüntüsü için
        rest  = [lm[tip].y < lm[pip].y
                 for tip, pip in [(8,6),(12,10),(16,14),(20,18)]]
        return [thumb] + rest

    @staticmethod
    def pinch_dist(lm) -> float:
        dx = lm[4].x - lm[8].x
        dy = lm[4].y - lm[8].y
        return math.sqrt(dx*dx + dy*dy)

    def classify(self, lm) -> Gesture:
        ext   = self.fingers_up(lm)
        pinch = self.pinch_dist(lm)
        _, idx, mid, rng, pnk = ext

        if pinch < PINCH_THRESHOLD:                          return Gesture.LEFT_CLICK
        if idx and not mid and not rng and pnk:              return Gesture.WIN_SWITCH   # 🤘 işaret+serçe
        if idx and mid and rng and not pnk:                  return Gesture.RIGHT_CLICK  # 3 parmak
        if idx and mid and not rng and not pnk:              return Gesture.SCROLL
        if not idx and not mid and not rng and not pnk:      return Gesture.DRAG
        return Gesture.MOVE

    # ── Çizim ────────────────────────────────────────────────────────────────

    # Parmak ucu indisleri
    FINGERTIPS = {4, 8, 12, 16, 20}

    def draw(self, frame: np.ndarray, lm) -> None:
        h, w = frame.shape[:2]
        pts  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]

        # Bağlantı çizgileri — ince yarı-saydam mor
        for a, b in self.CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (130, 80, 180), 1, cv2.LINE_AA)

        # Eklem noktaları
        for i, p in enumerate(pts):
            if i in self.FINGERTIPS:
                # Parmak uçları: vurgulu daire
                cv2.circle(frame, p, 6, (220, 160, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 6, (255, 220, 255), 1,  cv2.LINE_AA)
            else:
                # Diğer eklemler: küçük nokta
                cv2.circle(frame, p, 3, (160, 120, 200), -1, cv2.LINE_AA)

        # Avuç merkezi — parlak halka
        cx = int(self.palm_center(lm)[0] * w)
        cy = int(self.palm_center(lm)[1] * h)
        cv2.circle(frame, (cx, cy), 9,  (200, 120, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3,  (220, 180, 255), -1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD'LER
# ═══════════════════════════════════════════════════════════════════════════════

class CaptureThread(threading.Thread):
    """Kamera okumayı ana döngüden ayırır — en son frame her zaman hazır."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self._cap   = cap
        self._lock  = threading.Lock()
        self._frame = None
        self._fid   = 0
        self._run   = True

    def run(self):
        while self._run:
            ret, fr = self._cap.read()
            if ret and fr is not None:
                with self._lock:
                    self._frame = fr
                    self._fid  += 1

    def get(self):
        with self._lock:
            return self._frame, self._fid

    def stop(self): self._run = False


class DetectorThread(threading.Thread):
    """MediaPipe tespitini arka planda çalıştırır."""

    def __init__(self, detector: HandDetector, t0: float):
        super().__init__(daemon=True)
        self._det   = detector
        self._t0    = t0
        self._q     = queue.Queue(maxsize=1)
        self._lock  = threading.Lock()
        self._lm    = None
        self._seq   = 0
        self._run   = True

    def run(self):
        while self._run:
            try:
                rgb, ts = self._q.get(timeout=0.1)
                res = self._det.detect(rgb, ts)
                lm  = res.hand_landmarks[0] if res.hand_landmarks else None
                with self._lock:
                    self._lm  = lm
                    self._seq += 1
            except queue.Empty:
                continue

    def submit(self, rgb: np.ndarray):
        ts = int((time.perf_counter() - self._t0) * 1000)
        try:
            self._q.put_nowait((rgb, ts))
        except queue.Full:
            pass

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
        self._last_pos: tuple = None

        # EMA ön-filtresi — palm koordinatlarını One Euro'ya vermeden önce yumuşatır
        self._prev_px: float = None
        self._prev_py: float = None

        self._last_click = 0.0
        self._dragging   = False
        self._scroll_ref = None

        # WIN_SWITCH — 3 parmak kaydırma
        self._swipe_ref:  tuple = None   # gesture başladığındaki palm konumu
        self._last_swipe: float = 0.0    # son swipe zamanı (ms)

    def _screen_pos(self, nx: float, ny: float):
        """Normalize koordinatı → filtrelenmiş ekran pikseli."""
        ts = time.perf_counter()

        # Aktif bölge → [0, 1]
        cx = (max(ACTIVE_X_MIN, min(ACTIVE_X_MAX, nx)) - ACTIVE_X_MIN) \
             / (ACTIVE_X_MAX - ACTIVE_X_MIN)
        cy = (max(ACTIVE_Y_MIN, min(ACTIVE_Y_MAX, ny)) - ACTIVE_Y_MIN) \
             / (ACTIVE_Y_MAX - ACTIVE_Y_MIN)

        fx = int(self._fx(cx * self.sw, ts))
        fy = int(self._fy(cy * self.sh, ts))
        fx = max(0, min(self.sw - 1, fx))
        fy = max(0, min(self.sh - 1, fy))

        # Dead zone — micro-titremeleri yut
        if self._last_pos:
            if abs(fx - self._last_pos[0]) <= CURSOR_DEAD_ZONE_PX and \
               abs(fy - self._last_pos[1]) <= CURSOR_DEAD_ZONE_PX:
                return None
        self._last_pos = (fx, fy)
        return fx, fy

    def reset(self):
        self._fx.reset(); self._fy.reset()
        self._last_pos = None
        self._prev_px  = None
        self._prev_py  = None
        self._swipe_ref = None

    def handle(self, gesture: Gesture, lm) -> None:
        now = time.time() * 1000
        px, py = HandDetector.palm_center(lm)

        # ── EMA yumuşatma (dnm.py → yumusak) ────────────────────────────────
        # Ham palm koordinatlarındaki ani sıçramaları One Euro'ya girmeden keser.
        if self._prev_px is None:
            self._prev_px, self._prev_py = px, py
        px = self._prev_px + SMOOTH_ALPHA * (px - self._prev_px)
        py = self._prev_py + SMOOTH_ALPHA * (py - self._prev_py)
        self._prev_px, self._prev_py = px, py
        # ─────────────────────────────────────────────────────────────────────

        pos    = self._screen_pos(px, py)
        cx, cy = pos if pos else (self._last_pos or (self.sw//2, self.sh//2))

        # Scroll dışında scroll referansını sıfırla
        if gesture != Gesture.SCROLL:
            self._scroll_ref = None
        # WIN_SWITCH dışında swipe referansını sıfırla
        if gesture != Gesture.WIN_SWITCH:
            self._swipe_ref = None
        # Drag bitti mi?
        if gesture != Gesture.DRAG and self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

        if gesture == Gesture.MOVE:
            if pos:
                pyautogui.moveTo(cx, cy)

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
            # Referans noktasını gesture ilk başladığında kaydet
            if self._swipe_ref is None:
                self._swipe_ref = (px, py)

            dx = px - self._swipe_ref[0]
            dy = py - self._swipe_ref[1]

            if now - self._last_swipe > SWIPE_COOLDOWN_MS:
                if abs(dx) >= SWIPE_THRESHOLD and abs(dx) >= abs(dy):
                    # Yatay kaydırma — pencere ileri / geri
                    if dx > 0:
                        pyautogui.hotkey('alt', 'tab')           # → Sonraki pencere
                    else:
                        pyautogui.hotkey('alt', 'shift', 'tab')  # ← Önceki pencere
                    self._last_swipe = now
                    self._swipe_ref  = (px, py)                  # referansı sıfırla
                elif abs(dy) >= SWIPE_THRESHOLD and abs(dy) > abs(dx):
                    # Dikey kaydırma — görev görünümü / masaüstü
                    if dy < 0:
                        pyautogui.hotkey('win', 'tab')   # ↑ Görev Görünümü
                    else:
                        pyautogui.hotkey('win', 'd')     # ↓ Masaüstüne git
                    self._last_swipe = now
                    self._swipe_ref  = (px, py)

    def release(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

# ═══════════════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════════════

def draw_active_zone(frame: np.ndarray):
    h, w = frame.shape[:2]
    x1 = int(ACTIVE_X_MIN * w); y1 = int(ACTIVE_Y_MIN * h)
    x2 = int(ACTIVE_X_MAX * w); y2 = int(ACTIVE_Y_MAX * h)

    # Dış alanı karart (aktif bölge dışı)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    dark = frame.astype(np.float32)
    dark[mask == 0] *= 0.38
    frame[:] = np.clip(dark, 0, 255).astype(np.uint8)

    # Aktif alan çerçevesi: ince aksan rengi
    cv2.rectangle(frame, (x1, y1), (x2, y2), UI_BORDER, 1)

    # Köşe vurguları — modern "bracket" stili
    k = 18
    col = UI_ACCENT
    for cx_, cy_, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx_, cy_), (cx_ + dx*k, cy_), col, 2, cv2.LINE_AA)
        cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy*k), col, 2, cv2.LINE_AA)


_OVERLAY_TAG = {OVERLAY_NORMAL: "NORMAL", OVERLAY_PIP: "PiP", OVERLAY_TRANSPARENT: "SAYDAM"}

def draw_hud(frame: np.ndarray, gesture: Gesture, fps: float,
             debug: bool, lm=None, overlay_mode: int = OVERLAY_NORMAL):
    h, w  = frame.shape[:2]
    color = GESTURE_COLORS[gesture]
    label = GESTURE_LABELS[gesture]
    tag   = _OVERLAY_TAG.get(overlay_mode, "")

    # ── Üst bar — koyu yarı-saydam ────────────────────────────────────────
    BAR_H = 38
    roi   = frame[0:BAR_H].copy()
    bar   = np.zeros_like(roi); bar[:] = UI_BAR_BG
    cv2.addWeighted(bar, 0.82, roi, 0.18, 0, frame[0:BAR_H])
    # Alt kenarda ince aksan çizgisi
    cv2.line(frame, (0, BAR_H), (w, BAR_H), UI_ACCENT, 1)

    # Sol: uygulama adı
    cv2.putText(frame, "handmouse", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, UI_ACCENT, 1, cv2.LINE_AA)

    # Orta: mod etiketi
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, tag, (w // 2 - tw // 2, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, UI_TEXT_DIM, 1, cv2.LINE_AA)

    # Sağ: FPS
    fps_str = f"{fps:.0f} fps"
    (fw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(frame, fps_str, (w - fw - 12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, UI_TEXT_DIM, 1, cv2.LINE_AA)

    # ── Sol alt: gesture göstergesi ───────────────────────────────────────
    if gesture != Gesture.NONE:
        gy = h - 18
        # Renkli nokta
        cv2.circle(frame, (16, gy - 4), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (16, gy - 4), 6, (255, 255, 255), 1, cv2.LINE_AA)
        # Gesture etiketi
        cv2.putText(frame, label, (28, gy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, UI_TEXT, 2, cv2.LINE_AA)

    # ── Sağ alt: kısa yardım ──────────────────────────────────────────────
    cv2.putText(frame, "M mod  Q cık", (w - 110, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, UI_TEXT_DIM, 1, cv2.LINE_AA)

    # ── Debug ─────────────────────────────────────────────────────────────
    if debug and lm is not None:
        # fingers_up ve pinch_dist statik metot — instance gerekmez
        ext   = HandDetector.fingers_up(lm)
        pinch = HandDetector.pinch_dist(lm)
        names = ["Bas","Isa","Ort","Yuz","Ser"]
        info  = "  ".join(f"{n}:{'▲' if e else '▽'}" for n, e in zip(names, ext))
        cv2.putText(frame, f"{info}  pinch:{pinch:.3f}", (8, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 220, 160), 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLAY — SAYDAM VE PiP PENCERE YÖNETİMİ
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Neden önceki Win32 yaklaşımı çalışmadı:
#  ─────────────────────────────────────────
#  OpenCV, her cv2.imshow() çağrısında pencereyi BitBlt ile çizer. Bu,
#  SetLayeredWindowAttributes ile atanan WS_EX_LAYERED bayrağıyla uyumsuz;
#  OpenCV kendi pencere stilini yönettiğinden dışarıdan eklenen saydamlık
#  çerçeve güncellemelerinde geçersiz hale geliyor.
#
#  Çözüm: Saydam mod için tkinter + Pillow kullan. Tkinter'ın attributes('-alpha')
#  özelliği DWM kompozisyonunu doğrudan kullanır, OpenCV'den bağımsız çalışır.
# ───────────────────────────────────────────────────────────────────────────────

def _win32_set_topmost(title: str) -> None:
    """cv2 penceresini always-on-top yapar (PiP modu için)."""
    try:
        import ctypes
        HWND_TOPMOST = -1
        SWP_FLAGS    = 0x0002 | 0x0001 | 0x0040   # NOMOVE | NOSIZE | SHOWWINDOW
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_FLAGS)
    except Exception:
        pass


def _win32_dark_titlebar(title: str) -> None:
    """Windows 10/11: başlık çubuğunu koyu moda alır (DWM)."""
    try:
        import ctypes
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            val = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(val), ctypes.sizeof(val)
            )
    except Exception:
        pass


class _TransparentOverlay:
    """
    Tam ekran saydam tkinter penceresi — kamera görüntüsünü filtre gibi gösterir.

    • attributes('-alpha', OVERLAY_ALPHA) → görsel saydamlık (DWM kompozisyonu)
    • WS_EX_TRANSPARENT                  → fare tıklamaları alttaki programa geçer
    • Kendi thread'inde çalışır; push() ile frame alır.
    """

    def __init__(self, sw: int, sh: int, alpha: float):
        self._sw, self._sh = sw, sh
        self._alpha = alpha
        self._fq      = queue.Queue(maxsize=2)
        self._running = True
        self._ready   = threading.Event()
        self._root    = None
        threading.Thread(target=self._loop, daemon=True).start()
        self._ready.wait(timeout=4.0)   # tkinter hazır olana kadar bekle

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._root is not None

    def push(self, frame: np.ndarray) -> None:
        """Ana döngüden yeni kare gönder; kuyruğu taşarsa eski kareyi at."""
        try:
            self._fq.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._fq.get_nowait()
                self._fq.put_nowait(frame.copy())
            except Exception:
                pass

    def destroy(self) -> None:
        self._running = False
        try:
            if self._root:
                self._root.after(0, self._root.quit)
        except Exception:
            pass

    # ── private ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            import ctypes
        except ImportError as exc:
            print(f"[HATA] Pillow gerekli → pip install Pillow  ({exc})")
            self._ready.set()
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.geometry(f"{self._sw}x{self._sh}+0+0")
        root.configure(bg='black')

        # root.update() → OS'nin HWND'yi kayıt etmesini bekle.
        # Bu olmadan winfo_id() geçersiz handle döndürebilir.
        root.update()

        # ── Win32 — TEK ELDEN uygula: layered + click-through + alpha + topmost ──
        # Neden tkinter attributes('-alpha') KULLANILMIYOR:
        #   tkinter, attributes('-alpha') ile kendi SetLayeredWindowAttributes'unu
        #   çağırıyor. Ardından biz SetWindowLongW ile WS_EX_TRANSPARENT eklediğimizde
        #   bazı Windows sürümlerinde layered attribute sıfırlanıyor ve pencere
        #   ya görünmez ya da opak kalıyor. Hepsini sırayla kendiniz uygulamak güvenli.
        try:
            GWL_EXSTYLE       = -20
            WS_EX_LAYERED     = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            LWA_ALPHA         = 0x00000002
            HWND_TOPMOST      = -1
            SWP_FLAGS         = 0x0002 | 0x0001 | 0x0040   # NOMOVE | NOSIZE | SHOWWINDOW

            hwnd = root.winfo_id()
            # 1) Layered + click-through stilini ekle
            old = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, old | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
            # 2) Görsel saydamlık (SetWindowLongW'dan SONRA çağır — öncesi yok sayılır)
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd, 0, max(0, min(255, int(self._alpha * 255))), LWA_ALPHA
            )
            # 3) Always-on-top
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_FLAGS
            )
        except Exception as ex:
            print(f"[UYARI] Win32 saydam efekti başarısız, tkinter fallback: {ex}")
            root.attributes('-topmost', True)
            root.attributes('-alpha', self._alpha)
        # ─────────────────────────────────────────────────────────────────────

        lbl = tk.Label(root, bg='black', borderwidth=0, highlightthickness=0)
        lbl.pack(fill='both', expand=True)

        self._root = root

        def _update():
            if not self._running:
                root.quit()
                return
            try:
                frame = self._fq.get_nowait()
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(cv2.resize(rgb, (self._sw, self._sh)))
                photo = ImageTk.PhotoImage(img)
                lbl.config(image=photo)
                lbl.image = photo   # GC referans
            except queue.Empty:
                pass
            except Exception:
                pass
            root.after(16, _update)

        self._ready.set()
        root.after(16, _update)
        root.mainloop()


_CTRL_TITLE = "handmouse [M:mod  Q:cik]"   # saydam modda mini kontrol penceresi


class OverlayManager:
    """
    Üç kamera görüntü modunu yönetir.
    ─────────────────────────────────
    OVERLAY_NORMAL      → Ayrı OpenCV penceresi
    OVERLAY_PIP         → Köşede küçük, always-on-top (Win32)
    OVERLAY_TRANSPARENT → Tam ekran saydam tkinter filtresi
    """

    def __init__(self, mode: int, sw: int, sh: int):
        self._mode = mode
        self._sw, self._sh = sw, sh
        self._tk           = None   # _TransparentOverlay örneği
        self._pip_count    = 0
        self._dark_applied = False
        self._setup(mode)

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def mode(self) -> int:
        return self._mode

    def change_mode(self, new_mode: int) -> None:
        self._mode = new_mode
        self._setup(new_mode)
        names = {OVERLAY_NORMAL: "Normal", OVERLAY_PIP: "PiP Kose",
                 OVERLAY_TRANSPARENT: "Saydam Filtre"}
        print(f"[BİLGİ] Kamera modu → {names.get(new_mode, '?')}")

    def show(self, frame: np.ndarray) -> int:
        """Frame'i göster; cv2.waitKey(1) sonucunu döndür."""
        if self._mode == OVERLAY_TRANSPARENT:
            if self._tk:
                self._tk.push(frame)
            # Mini kontrol penceresi waitKey için zorunlu
            ctrl = np.zeros((28, 240, 3), dtype=np.uint8)
            cv2.putText(ctrl, "M:mod degistir   Q:cik",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
            cv2.imshow(_CTRL_TITLE, ctrl)
        else:
            cv2.imshow(WIN_TITLE, frame)
            # Pencere ilk görüntülenince dark titlebar + topmost uygula
            if self._mode == OVERLAY_PIP and self._pip_count < 8:
                self._pip_count += 1
                if self._pip_count == 5:
                    _win32_set_topmost(WIN_TITLE)
                    _win32_dark_titlebar(WIN_TITLE)
            elif self._mode == OVERLAY_NORMAL and not self._dark_applied:
                self._pip_count += 1
                if self._pip_count >= 5:
                    _win32_dark_titlebar(WIN_TITLE)
                    self._dark_applied = True
        return cv2.waitKey(1) & 0xFF

    def destroy(self) -> None:
        if self._tk:
            self._tk.destroy()
        cv2.destroyAllWindows()

    # ── private ──────────────────────────────────────────────────────────────

    def _setup(self, mode: int) -> None:
        if self._tk:
            self._tk.destroy()
            self._tk = None
        cv2.destroyAllWindows()
        time.sleep(0.05)
        self._pip_count = 0

        if mode == OVERLAY_NORMAL:
            cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_TITLE, CAM_WIDTH, CAM_HEIGHT)
            # Birkaç döngü sonra dark titlebar uygulanacak (_pip_count sıfırlandı)
            self._dark_applied = False

        elif mode == OVERLAY_PIP:
            cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_TITLE, PIP_W, PIP_H)
            corners = {
                "tl": (PIP_MARGIN,                     PIP_MARGIN),
                "tr": (self._sw - PIP_W - PIP_MARGIN,  PIP_MARGIN),
                "bl": (PIP_MARGIN,                     self._sh - PIP_H - PIP_MARGIN - 50),
                "br": (self._sw - PIP_W - PIP_MARGIN,  self._sh - PIP_H - PIP_MARGIN - 50),
            }
            px_, py_ = corners.get(PIP_CORNER, corners["br"])
            cv2.moveWindow(WIN_TITLE, px_, py_)

        elif mode == OVERLAY_TRANSPARENT:
            self._tk = _TransparentOverlay(self._sw, self._sh, OVERLAY_ALPHA)
            if not self._tk.is_ready:
                print("[UYARI] Saydam mod başlatılamadı. Normal moda geçiliyor.")
                self._mode = OVERLAY_NORMAL
                self._setup(OVERLAY_NORMAL)
                return
            # waitKey için mini kontrol penceresi
            cv2.namedWindow(_CTRL_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_CTRL_TITLE, 240, 28)
            cv2.moveWindow(_CTRL_TITLE, 10, self._sh - 55)


# ═══════════════════════════════════════════════════════════════════════════════
#  MEVCUT KAMERALARI TARA
# ═══════════════════════════════════════════════════════════════════════════════

def list_cameras(max_idx: int = 5) -> list:
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found


# ═══════════════════════════════════════════════════════════════════════════════
#  MODERN KONTROL PANELİ (tkinter)
# ═══════════════════════════════════════════════════════════════════════════════

class ControlPanel:
    """
    Yüzen koyu-tema tkinter kontrol paneli.
    Ana döngüden update() ile beslenir; komutlar get_command() ile alınır.
    """

    # Renk şeması
    C_BG      = "#0d0d1a"
    C_CARD    = "#13131f"
    C_ACCENT  = "#a064b4"   # mor-pembe aksan
    C_TEXT    = "#dcd8f0"
    C_DIM     = "#605878"
    C_RED     = "#e05060"
    C_SEP     = "#252535"

    def __init__(self):
        self._state   = {"gesture": "---", "fps": 0.0, "mode": 0, "running": True}
        self._cmd_q   = queue.Queue()
        self._root    = None
        self._ready   = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self._ready.wait(timeout=5.0)

    # ── public ──────────────────────────────────────────────────────────────

    def update(self, gesture_label: str, fps: float, mode: int) -> None:
        self._state.update({"gesture": gesture_label, "fps": fps, "mode": mode})

    def get_command(self):
        """("quit", None) | ("mode", int) | None"""
        try:
            return self._cmd_q.get_nowait()
        except queue.Empty:
            return None

    def destroy(self) -> None:
        self._state["running"] = False
        try:
            if self._root:
                self._root.after(0, self._root.quit)
        except Exception:
            pass

    # ── private ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            self._ready.set()
            return

        C = self   # kısayol

        root = tk.Tk()
        root.title("handmouse")
        root.configure(bg=C.C_BG)
        root.resizable(False, False)
        root.geometry("260+20+60")
        root.attributes("-topmost", True)

        # ── Win32: koyu başlık çubuğu ──────────────────────────────────────
        try:
            import ctypes
            root.update()
            hwnd = root.winfo_id()
            DWMWA_DARK = 20
            val = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_DARK, ctypes.byref(val), ctypes.sizeof(val)
            )
        except Exception:
            pass

        def _label(parent, text, fg=None, font=None, **kw):
            return tk.Label(parent, text=text, fg=fg or C.C_TEXT,
                            bg=kw.pop("bg", C.C_BG),
                            font=font or ("Segoe UI", 10), **kw)

        # ── Başlık kartı ───────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=C.C_CARD, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="●", fg=C.C_ACCENT, bg=C.C_CARD,
                 font=("Segoe UI", 13)).pack(side="left", padx=(14, 5))
        tk.Label(hdr, text="handmouse", fg=C.C_TEXT, bg=C.C_CARD,
                 font=("Segoe UI", 13, "bold")).pack(side="left")

        # Aksan çizgisi
        tk.Frame(root, bg=C.C_ACCENT, height=2).pack(fill="x")

        # ── Gesture bölümü ────────────────────────────────────────────────
        _label(root, "GESTURE", fg=C.C_DIM, font=("Segoe UI", 8)).pack(
            anchor="w", padx=16, pady=(14, 0))
        gesture_var = tk.StringVar(value="---")
        gesture_lbl = tk.Label(root, textvariable=gesture_var,
                               fg=C.C_ACCENT, bg=C.C_BG,
                               font=("Segoe UI", 20, "bold"))
        gesture_lbl.pack(anchor="w", padx=16, pady=(2, 0))

        # ── FPS bölümü ────────────────────────────────────────────────────
        tk.Frame(root, bg=C.C_SEP, height=1).pack(fill="x", padx=16, pady=(14, 0))
        fps_frame = tk.Frame(root, bg=C.C_BG)
        fps_frame.pack(fill="x", padx=16, pady=(8, 0))
        _label(fps_frame, "FPS", fg=C.C_DIM, font=("Segoe UI", 8)).pack(side="left")
        fps_var = tk.StringVar(value="0")
        tk.Label(fps_frame, textvariable=fps_var, fg=C.C_TEXT, bg=C.C_BG,
                 font=("Segoe UI", 12, "bold")).pack(side="right")

        # ── Mod seçici ────────────────────────────────────────────────────
        tk.Frame(root, bg=C.C_SEP, height=1).pack(fill="x", padx=16, pady=(12, 0))
        _label(root, "KAMERA MODU", fg=C.C_DIM, font=("Segoe UI", 8)).pack(
            anchor="w", padx=16, pady=(8, 4))

        mode_btns: list = []
        mode_row = tk.Frame(root, bg=C.C_BG)
        mode_row.pack(fill="x", padx=16, pady=(0, 4))

        for i, lbl in enumerate(["Normal", "PiP", "Saydam"]):
            def _cmd(idx=i):
                self._cmd_q.put(("mode", idx))
            b = tk.Button(mode_row, text=lbl, command=_cmd,
                          bg=C.C_CARD, fg=C.C_TEXT,
                          font=("Segoe UI", 9), bd=0,
                          padx=10, pady=6, cursor="hand2",
                          activebackground=C.C_ACCENT,
                          activeforeground="#ffffff",
                          relief="flat")
            b.pack(side="left", padx=(0, 4))
            mode_btns.append(b)

        # ── Kısayollar ────────────────────────────────────────────────────
        tk.Frame(root, bg=C.C_SEP, height=1).pack(fill="x", padx=16, pady=(10, 6))
        shortcuts = [
            ("☝  Tek parmak", "Hareket"),
            ("🤏  Pinch",      "Sol tık"),
            ("✌  İki parmak",  "Scroll"),
            ("🖖  Üç parmak",  "Sağ tık"),
            ("✊  Yumruk",      "Sürükle"),
        ]
        for icon_txt, action in shortcuts:

            row = tk.Frame(root, bg=C.C_BG)
            row.pack(fill="x", padx=16, pady=1)
            tk.Label(row, text=icon_txt, fg=C.C_DIM, bg=C.C_BG,
                     font=("Segoe UI", 8), width=16, anchor="w").pack(side="left")
            tk.Label(row, text=action, fg=C.C_TEXT, bg=C.C_BG,
                     font=("Segoe UI", 8, "bold"), anchor="e").pack(side="right")

        # -- Kapat butonu --
        tk.Frame(root, bg=C.C_SEP, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Button(root, text="  Kapat  X  ",
                  command=lambda: self._cmd_q.put(("quit", None)),
                  bg="#2a1520", fg=C.C_RED,
                  font=("Segoe UI", 10), bd=0,
                  padx=0, pady=8, cursor="hand2",
                  activebackground="#3a1f2a",
                  activeforeground="#ff8888",
                  relief="flat").pack(fill="x", padx=16, pady=(8, 14))

        self._root = root
        self._ready.set()

        def _tick():
            if not self._state["running"]:
                root.quit()
                return
            gesture_var.set(self._state["gesture"])
            fps_var.set(f"{self._state['fps']:.0f}")
            m = self._state["mode"]
            for idx, btn in enumerate(mode_btns):
                btn.configure(
                    bg=C.C_ACCENT if idx == m else C.C_CARD,
                    fg="#ffffff"   if idx == m else C.C_TEXT,
                )
            root.after(120, _tick)

        root.after(120, _tick)
        root.mainloop()


# =============================================================================
#  MEVCUT KAMERALARI TARA
# =============================================================================

def list_cameras(max_idx: int = 5) -> list:
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found


# =============================================================================
#  ANA DONGU
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bilgisayar kamerasıyla el hareketi mouse kontrolü",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cam", type=int, default=0,
                        metavar="INDEX", help="Kamera indexi (varsayılan: 0)")
    parser.add_argument("--list-cams", action="store_true",
                        help="Mevcut kameraları listele ve cık")
    parser.add_argument("--debug", action="store_true",
                        help="Parmak acık/kapalı ve pinch degerini göster")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Ayna görüntüyü kapat (varsayılan: ayna acık)")
    args = parser.parse_args()

    if args.list_cams:
        cams = list_cameras()
        if cams:
            print(f"[BILGI] Bulunan kameralar: {cams}")
        else:
            print("[UYARI] Hic kamera bulunamadı.")
        return

    print(f"[BILGI] Kamera #{args.cam} acılıyor...")
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[HATA] Kamera #{args.cam} acılamadı!")
        print("       Mevcut kameraları görmek icin: python handmouse_pc.py --list-cams")
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

    sc_w, sc_h = pyautogui.size()
    overlay    = OverlayManager(OVERLAY_NORMAL, sc_w, sc_h)

    # Kontrol panelini basalt
    panel = ControlPanel()

    cap_th.start()
    det_th.start()

    print("[BILGI] Kamera baslatılıyor...", end="", flush=True)
    for _ in range(100):
        if cap_th.get()[0] is not None:
            break
        time.sleep(0.05)
    print(" hazır!\n")
    print("=" * 62)
    print("  EL HAREKETLERI")
    print("  Tek isaret parmagi      --> Cursor hareketi")
    print("  Pinch (bas+isaret)      --> Sol tıklama")
    print("  Iki parmak              --> Scroll")
    print("  Isaret + Sorte kaydır   --> Pencere gecisi")
    print("  Uc parmak               --> Sag tıklama")
    print("  Yumruk                  --> Surukle")
    print()
    print("  Cıkmak icin: Q veya ESC")
    print("=" * 62 + "\n")

    mirror   = not args.no_mirror
    fps_buf  = deque(maxlen=30)
    gesture  = Gesture.NONE
    last_fid = -1
    last_seq = -1
    _dt      = 1.0 / TARGET_LOOP_FPS
    debug    = args.debug

    try:
        while True:
            t_loop = time.perf_counter()

            frame, fid = cap_th.get()
            if frame is None:
                time.sleep(0.005)
                continue

            frame = frame.copy()
            if mirror:
                frame = cv2.flip(frame, 1)

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

            draw_active_zone(frame)
            if lm is not None:
                detector.draw(frame, lm)

            elapsed = time.perf_counter() - t_loop
            fps_buf.append(1.0 / elapsed if elapsed > 1e-6 else TARGET_LOOP_FPS)

            draw_hud(frame, gesture, fps_buf[-1] if fps_buf else 0.0,
                     debug, lm, overlay.mode)

            key = overlay.show(frame)

            # Kontrol paneli guncelle
            panel.update(GESTURE_LABELS[gesture], fps_buf[-1] if fps_buf else 0.0, overlay.mode)

            # Panel komutlarını isle
            cmd = panel.get_command()
            if cmd:
                if cmd[0] == "quit":
                    break
                elif cmd[0] == "mode":
                    overlay.change_mode(cmd[1])

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('m'), ord('M')):
                overlay.change_mode((overlay.mode + 1) % 3)
            elif key in (ord('d'), ord('D')):
                debug = not debug

            sleep_s = _dt - (time.perf_counter() - t_loop)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[BILGI] Kapatılıyor...")
        panel.destroy()
        controller.release()
        cap_th.stop()
        det_th.stop()
        overlay.destroy()
        cap.release()
        cv2.destroyAllWindows()
        print("[BILGI] Tamamlandı.")


if __name__ == "__main__":
    ensure_model()
    main()
