"""
Test dengan gambar wajah dari internet
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Tambahkan path ke deepface-master
sys.path.insert(0, r"c:\Users\Asus\Documents\Kodingan\yolo\deepface-master\deepface-master")

# Tambahkan path ke project
sys.path.insert(0, r"c:\Users\Asus\Documents\Kodingan\yolo\face_attendance")

try:
    from face_recognition.advanced_detector import AdvancedFaceDetector
    from face_recognition.model_preloader import preloader, initialize_models
    
    print("🧪 Testing DeepFace Detector dengan gambar wajah...")
    
    # Initialize models
    print("🔄 Menginisialisasi model preloader...")
    initialize_models()
    
    # Download test image dengan wajah
    print("🔄 Download gambar test...")
    import urllib.request
    
    # Gunakan gambar test dari placeholder dengan wajah
    test_url = "https://via.placeholder.com/400x400/000000/FFFFFF?text=FACE+TEST"
    
    try:
        # Download gambar
        urllib.request.urlretrieve(test_url, "test_face.jpg")
        
        # Baca gambar
        test_image = cv2.imread("test_face.jpg")
        if test_image is None:
            print("❌ Gagal load gambar, buat gambar test...")
            test_image = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(test_image, "FACE", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    except Exception as e:
        print(f"⚠️  Download gagal: {e}, buat gambar test...")
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(test_image, "FACE", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
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
            bbox = face.get('bbox', (0,0,100,100))
            confidence = face.get('confidence', 0)
            backend = face.get('backend', 'unknown')
            print(f"  Wajah {i+1}: bbox={bbox}, confidence={confidence:.2f}, backend={backend}")
            
            # Gambar bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(test_image, f"{backend}: {confidence:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print("❌ Tidak ada wajah yang terdeteksi")
        error_msg = detector.get_last_error()
        if error_msg:
            print(f"📋 Last error: {error_msg}")
    
    # Simpan hasil
    cv2.imwrite("test_result.jpg", test_image)
    print("💾 Hasil disimpan di test_result.jpg")
    
    print("\n✅ Test selesai!")
    
except Exception as e:
    print(f"❌ Error utama: {e}")
    import traceback
    traceback.print_exc()