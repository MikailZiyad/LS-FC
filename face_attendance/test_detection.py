#!/usr/bin/env python3
"""
Test real-time face detection untuk debugging
"""
import sys
import os
import cv2
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_real_time_detection():
    """Test real-time face detection"""
    try:
        print("🧪 Testing real-time face detection...")
        from face_recognition.detector import FaceDetector
        
        detector = FaceDetector()
        print("✅ Model YOLO berhasil dimuat")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Tidak bisa membuka kamera")
            return False
        
        print("✅ Kamera berhasil dibuka")
        print("📸 Arahkan wajah ke kamera, tekan 'q' untuk keluar")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Tidak bisa membaca frame")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            if len(faces) > 0:
                detection_count += 1
                print(f"🎯 Wajah terdeteksi! Frame {frame_count}: {len(faces)} wajah")
                
                # Draw bounding boxes
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    conf = face['confidence']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face: {conf:.2f}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow('Face Detection Test', frame)
            
            frame_count += 1
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Auto exit after 5 seconds
            if frame_count > 150:  # ~5 seconds at 30fps
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"📊 Total frame: {frame_count}, Deteksi berhasil: {detection_count}")
        
        if detection_count > 0:
            print("✅ Wajah berhasil dideteksi!")
            return True
        else:
            print("⚠️  Tidak ada wajah yang terdeteksi")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_face_quality():
    """Test face quality validation"""
    try:
        print("\n🧪 Testing face quality validation...")
        from face_recognition.trainer import FaceTrainer
        
        trainer = FaceTrainer()
        print("✅ Face trainer berhasil dimuat")
        
        # Test with dummy image
        dummy_face = np.zeros((200, 200, 3), dtype=np.uint8)
        is_valid, message = trainer.validate_face_quality(dummy_face)
        print(f"Face quality check: {is_valid}, Message: {message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Real-Time Face Detection...\n")
    
    # Test 1: Real-time detection
    detection_result = test_real_time_detection()
    
    # Test 2: Face quality
    quality_result = test_face_quality()
    
    print(f"\n📊 Summary:")
    print(f"  Real-time detection: {'✅ PASS' if detection_result else '❌ FAIL'}")
    print(f"  Face quality check: {'✅ PASS' if quality_result else '❌ FAIL'}")
    
    if not detection_result:
        print("\n💡 Tips untuk meningkatkan deteksi:")
        print("  1. Pastikan pencahayaan cukup")
        print("  2. Wajah tidak terlalu jauh dari kamera")
        print("  3. Model YOLO Anda sudah training untuk deteksi wajah")
        print("  4. Cek threshold confidence di config/settings.py")