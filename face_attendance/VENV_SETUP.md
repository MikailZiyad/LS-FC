# Face Attendance System - Virtual Environment Setup

## Cara Menggunakan Virtual Environment di IDE

### 1. Aktivasi Manual (PowerShell)
```powershell
# Masuk ke folder project
cd C:\Users\Asus\Documents\Kodingan\yolo\face_attendance

# Aktifkan virtual environment
env_attendance\Scripts\activate

# Jalankan aplikasi
python main.py
```

### 2. Setup untuk VS Code
1. Buka VS Code di folder `face_attendance`
2. Tekan `Ctrl+Shift+P` → pilih "Python: Select Interpreter"
3. Pilih interpreter: `C:\Users\Asus\Documents\Kodingan\yolo\face_attendance\env_attendance\Scripts\python.exe`
4. Restart VS Code

### 3. Setup untuk PyCharm
1. Buka project di PyCharm
2. File → Settings → Project → Python Interpreter
3. Klik gear icon → Add → Existing environment
4. Pilih: `C:\Users\Asus\Documents\Kodingan\yolo\face_attendance\env_attendance\Scripts\python.exe`
5. Apply dan OK

### 4. Dependencies Terinstall
✅ cv2 (OpenCV) - OK
✅ numpy - OK
✅ torch - OK
✅ deepface - OK
✅ customtkinter - OK
✅ ultralytics - OK
✅ PIL - OK
✅ tf-keras - OK

### 5. Test Import
Jalankan script ini untuk test semua imports:
```bash
python test_imports.py
```

### 6. Jalankan Aplikasi
```bash
# Aktifkan virtual environment dulu
env_attendance\Scripts\activate

# Jalankan aplikasi
python main.py
```

## Troubleshooting

### Error "Import could not be resolved" di IDE
- Pastikan IDE menggunakan interpreter dari virtual environment
- Restart IDE setelah ganti interpreter
- Tunggu beberapa detik untuk indexing

### Error saat running aplikasi
- Pastikan virtual environment sudah diaktifkan
- Cek error message di terminal
- Jalankan `test_imports.py` untuk verifikasi dependencies