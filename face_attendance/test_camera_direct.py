"""
Test langsung dengan kamera
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
    from utils.enhanced_visualization import draw_enhanced_face_border
    
    print("🧪 Testing DeepFace Detector dengan Kamera...")
    
    # Initialize models
    print("🔄 Menginisialisasi model preloader...")
    initialize_models()
    
    # Initialize detector
    print("🔄 Menginisialisasi detector...")
    detector = AdvancedFaceDetector(
        primary_backend="retinaface",
        fallback_backends=["yolov8n", "yolov11n"],
        enable_analysis=True
    )
    
    # Initialize camera
    print("🔄 Menginisialisasi kamera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Kamera tidak tersedia!")
        exit()
    
    print("✅ Kamera terhubung!")
    print("📹 Tekan 'q' untuk keluar, 's' untuk ganti backend")
    
    current_backend = 0
    backends = ["retinaface", "yolov8n", "yolov11n"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame")
            break
        
        # Deteksi wajah
        try:
            faces = detector.detect_faces(frame)
            result_frame = frame.copy()
            
            if faces:
                print(f"✅ Terdeteksi {len(faces)} wajah dengan {backends[current_backend]}")
                
                for i, face in enumerate(faces):
                    bbox = face.get('bbox', (0, 0, 100, 100))
                    confidence = face.get('confidence', 0)
                    
                    # Gambar border dengan attribute
                    result_frame = draw_enhanced_face_border(
                        result_frame, 
                        bbox, 
                        face.get('gender', ''),
                        face.get('age', ''),
                        face.get('emotion', ''),
                        face.get('race', ''),
                        confidence,
                        face.get('username', f'Person {i+1}')
                    )
                    
                    print(f"  Wajah {i+1}: gender={face.get('gender', '?')}, age={face.get('age', '?')}, emotion={face.get('emotion', '?')}, confidence={confidence:.2f}")
            else:
                cv2.putText(result_frame, f"No faces detected ({backends[current_backend]})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            result_frame = frame.copy()
            cv2.putText(result_frame, f"Error: {e}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"❌ Error deteksi: {e}")
        
        # Tampilkan info
        cv2.putText(result_frame, f"Backend: {backends[current_backend]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tampilkan frame
        cv2.imshow('DeepFace Detector Test', result_frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            current_backend = (current_backend + 1) % len(backends)
            detector.primary_backend = backends[current_backend]
            print(f"🔄 Switched to backend: {backends[current_backend]}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test selesai!")
    
except Exception as e:
    print(f"❌ Error utama: {e}")
    import traceback
    traceback.print_exc()