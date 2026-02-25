"""
Test sederhana untuk deepface detector tanpa kamera
"""
import cv2
import numpy as np
import sys
import os

# Tambahkan path ke deepface-master
sys.path.insert(0, r"c:\Users\Asus\Documents\Kodingan\yolo\deepface-master\deepface-master")

# Tambahkan path ke project
sys.path.insert(0, r"c:\Users\Asus\Documents\Kodingan\yolo\face_attendance")

try:
    from face_recognition.advanced_detector import AdvancedFaceDetector
    from face_recognition.model_preloader import preloader, initialize_models
    
    print("🧪 Testing DeepFace Detector Sederhana...")
    
    # Initialize models
    print("🔄 Menginisialisasi model preloader...")
    initialize_models()
    
    # Buat gambar test sederhana
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Initialize detector
    print("🔄 Menginisialisasi detector...")
    detector = AdvancedFaceDetector(
        primary_backend="retinaface",
        fallback_backends=["yolov8n", "yolov11n"],
        enable_analysis=True
    )
    
    # Test deteksi
    print("🔄 Testing deteksi wajah...")
    faces = detector.detect_faces(test_image)
    
    if faces:
        print(f"✅ Berhasil mendeteksi {len(faces)} wajah")
        for i, face in enumerate(faces):
            print(f"  Wajah {i+1}: confidence={face.get('confidence', 0):.2f}, backend={face.get('backend', 'unknown')}")
    else:
        print("❌ Tidak ada wajah yang terdeteksi (ini normal untuk gambar test)")
        error_msg = detector.get_last_error()
        if error_msg:
            print(f"📋 Last error: {error_msg}")
    
    # Test dengan backend yang berbeda
    print("\n🔄 Testing dengan backend berbeda...")
    for backend in ["yolov8n", "yolov11n"]:
        try:
            print(f"🔄 Testing {backend}...")
            faces = detector._detect_faces_single_backend(test_image, backend, 0.5)
            if faces:
                print(f"✅ {backend}: Berhasil mendeteksi {len(faces)} wajah")
            else:
                print(f"ℹ️  {backend}: Tidak ada wajah yang terdeteksi")
        except Exception as e:
            print(f"⚠️  {backend}: Error - {e}")
    
    print("\n✅ Test selesai!")
    
except Exception as e:
    print(f"❌ Error utama: {e}")
    import traceback
    traceback.print_exc()