#!/usr/bin/env python3
"""
Test script untuk detektor deepface-master yang baru
"""

import cv2
import numpy as np
from face_recognition.advanced_detector import AdvancedFaceDetector

def test_deepface_detectors():
    """Test semua detektor yang tersedia di deepface-master"""
    
    print("🧪 Testing DeepFace-Master Detectors...")
    
    # Inisialisasi detector
    detector = AdvancedFaceDetector(
        primary_backend="retinaface",
        fallback_backends=["yolov8n", "yolov8m", "yolov11n", "yolov12n"],
        enable_analysis=True
    )
    
    # Test dengan gambar webcam atau gambar test
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not available, create test image")
        # Buat gambar test
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test dengan gambar test
        print("\n📸 Testing with test image...")
        
        # Test deteksi wajah dengan analisis
        print(f"\n🔍 Testing dengan analisis wajah...")
        try:
            faces = detector.detect_faces_with_attributes(test_image)
            
            print(f"✅ {len(faces)} faces detected")
            
            for i, face in enumerate(faces):
                print(f"  Face {i+1}: confidence={face['confidence']:.2f}, backend={face.get('backend', 'unknown')}")
                if 'dominant_gender' in face:
                    print(f"    Gender: {face['dominant_gender']} ({face.get('gender_confidence', 0):.2f})")
                if 'age' in face:
                    print(f"    Age: {face['age']}")
                if 'dominant_emotion' in face:
                    print(f"    Emotion: {face['dominant_emotion']} ({face.get('emotion_confidence', 0):.2f})")
                if 'dominant_race' in face:
                    print(f"    Race: {face['dominant_race']} ({face.get('race_confidence', 0):.2f})")
                
        except Exception as e:
            print(f"❌ Detection failed: {e}")
        
        cap.release()
        return
    
    print("✅ Camera connected!")
    
    # Test dengan webcam
    print("\n📹 Testing with live camera...")
    print("Tekan 'q' untuk keluar, 's' untuk ganti backend")
    
    current_backend = "retinaface"
    backends = ["retinaface", "yolov8n", "yolov8m", "yolov11n", "yolov12n"]
    backend_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Deteksi dengan analisis lengkap
            faces = detector.detect_faces_with_attributes(frame)
            
            # Gambar hasil
            result_frame = frame.copy()
            
            for face in faces:
                bbox = face['bbox']
                confidence = face['confidence']
                backend_used = face.get('backend', current_backend)
                
                # Gambar bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tambahkan info dasar
                label1 = f"{backend_used}: {confidence:.2f}"
                cv2.putText(result_frame, label1, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Tambahkan info analisis jika tersedia
                y_offset = y2 + 20
                if 'dominant_gender' in face:
                    gender_info = f"Gender: {face['dominant_gender']} ({face.get('gender_confidence', 0):.2f})"
                    cv2.putText(result_frame, gender_info, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    y_offset += 15
                
                if 'age' in face:
                    age_info = f"Age: {face['age']}"
                    cv2.putText(result_frame, age_info, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    y_offset += 15
                
                if 'dominant_emotion' in face:
                    emotion_info = f"Emotion: {face['dominant_emotion']} ({face.get('emotion_confidence', 0):.2f})"
                    cv2.putText(result_frame, emotion_info, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    y_offset += 15
                
                if 'dominant_race' in face:
                    race_info = f"Race: {face['dominant_race']} ({face.get('race_confidence', 0):.2f})"
                    cv2.putText(result_frame, race_info, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Info di pojok kiri atas
            cv2.putText(result_frame, f"Backend: {current_backend}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(result_frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        except Exception as e:
            result_frame = frame.copy()
            cv2.putText(result_frame, f"Error: {e}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"❌ Error: {e}")
        
        # Tampilkan hasil
        cv2.imshow("DeepFace-Master Detector Test", result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Ganti backend
            backend_index = (backend_index + 1) % len(backends)
            current_backend = backends[backend_index]
            
            # Re-initialize detector dengan backend baru
            detector = AdvancedFaceDetector(
                primary_backend=current_backend,
                fallback_backends=[b for b in backends if b != current_backend],
                enable_analysis=True
            )
            print(f"🔄 Switched to backend: {current_backend}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_deepface_detectors()