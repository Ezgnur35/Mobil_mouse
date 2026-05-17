# 🖱️ Mobil Mouse

Bilgisayarın dahili kamerası kullanarak el hareketleriyle bilgisayar faresini kontrol eden sistem. Google MediaPipe ile gerçek zamanlı el landmark tespiti yapar; dokunmadan, fare veya klavye olmadan tam mouse kontrolü sağlar.

---

## 👥 Ekip

| İsim | Rol |
|------|-----|
| Ezginur Ünver | Python backend geliştirme, el hareketi algılama |
| Kaan Özdemir | Arayüz & kamera entegrasyonu |
| Yavuz Ünal | Test & Dokümantasyon |

---

## 🛠️ Teknolojiler

- Python 3.x
- MediaPipe — el landmark tespiti
- OpenCV — kamera görüntü işleme
- PyAutoGUI — mouse/klavye kontrolü
- NumPy — sayısal hesaplamalar
- pystray + Pillow — sistem tepsisi arayüzü (V2)

---

## ✋ El Hareketleri

| Hareket | Eylem |
|---------|-------|
| ☝️ Tek işaret parmağı | Cursor hareketi |
| 🤏 Pinch (baş + işaret) | Sol tıklama |
| ✌️ İki parmak | Kaydırma (scroll) |
| 🖖 Üç parmak | Sağ tıklama |
| ✊ Yumruk | Sürükleme (drag) |
| 🤘 İşaret + serçe kaydır | Pencere geçişi (Alt+Tab) |

---

## 📁 Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `handmouse_pc.py` | Bilgisayar kamerasıyla çalışan tam sürüm. OpenCV penceresi + kontrol paneli açar. |
| `handmouse_pcV2.py` | Sistem tepsisi servisi. Arka planda sessizce çalışır, OpenCV penceresi açmaz. |
| `requirements.txt` | Python bağımlılıkları |
| `kurulum.bat` | Windows için otomatik kurulum betiği |

---

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

veya Windows'ta `kurulum.bat` dosyasını çalıştır.

---

## 🚀 Çalıştırma

**Standart sürüm (kamera penceresiyle):**
```bash
python handmouse_pc.py              # Dahili kamera (index 0)
python handmouse_pc.py --cam 1      # İkinci kamera
python handmouse_pc.py --debug      # Debug bilgisi göster
python handmouse_pc.py --list-cams  # Mevcut kameraları listele
```

**Sistem tepsisi servisi (arka planda):**
```bash
python handmouse_pcV2.py
```
Tepsideki simgeye sağ tıklayarak kamera seçebilir, yayını başlatıp durdurabilir, IP/Port ayarlayabilirsin.

---

## 📦 Gereksinimler

```
mediapipe>=0.10.0
opencv-python>=4.8.0
pyautogui>=0.9.54
numpy>=1.24.0
pillow
```

> İlk çalıştırmada `hand_landmarker.task` model dosyası otomatik indirilir (~8 MB).
