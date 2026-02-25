# Face Attendance System - Environment Setup

## Setup Environment Baru

1. **Aktifkan Virtual Environment:**
```bash
env_attendance\Scripts\activate
```

2. **Install Dependencies:**
```bash
pip install deepface customtkinter torch torchvision ultralytics opencv-python numpy pillow
```

3. **Jalankan Aplikasi:**
```bash
python main.py
```

## Struktur Project
```
face_attendance/
├── config/              # Konfigurasi dan settings
├── face_recognition/    # Modul deteksi dan recognisi wajah
├── models/              # Database dan model employee
├── ui/                  # Interface GUI
├── utils/               # Utility functions
├── data/                # Data karyawan dan absensi
├── reports/             # Laporan absensi
└── main.py             # Entry point
```

## Fitur Utama
- ✅ Registrasi wajah dengan multi-pose
- ✅ Deteksi wajah menggunakan YOLO yang sudah ada
- ✅ Face recognition dengan DeepFace
- ✅ Anti-duplikasi karyawan
- ✅ Quality control wajah
- ✅ GUI dengan CustomTkinter

## Catatan Penting
- Gunakan environment `env_attendance` untuk menghindari konflik dependencies
- Model YOLO otomatis menggunakan model yang sudah ada di `data/models/.deepface/weights/`
- Database terpisah untuk attendance system