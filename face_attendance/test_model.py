#!/usr/bin/env python3
"""
Test script untuk cek model YOLO dan face recognition
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yolo_model():
    """Test YOLO model loading and detection"""
    try:
        print("🧪 Testing YOLO model...")
        from face_recognition.detector import FaceDetector
        
        detector = FaceDetector()
        print("✅ Model YOLO berhasil dimuat")
        
        # Test with dummy image
        import numpy as np
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(dummy_img)
        print(f"✅ Deteksi wajah di gambar dummy: {len(faces)} wajah")
        
        return True
        
    except Exception as e:
        print(f"❌ Error YOLO model: {e}")
        return False

def test_face_recognition():
    """Test face recognition model"""
    try:
        print("🧪 Testing face recognition...")
        from face_recognition.recognizer import FaceRecognizer
        
        recognizer = FaceRecognizer()
        print("✅ Face recognizer berhasil dimuat")
        
        return True
        
    except Exception as e:
        print(f"❌ Error face recognition: {e}")
        return False

def test_camera():
    """Test camera access"""
    try:
        print("🧪 Testing camera...")
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"✅ Kamera berhasil diakses, ukuran frame: {frame.shape}")
                return True
            else:
                print("❌ Tidak bisa membaca frame dari kamera")
                return False
        else:
            print("❌ Kamera tidak bisa dibuka")
            return False
            
    except Exception as e:
        print(f"❌ Error kamera: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Face Attendance System...\n")
    
    results = []
    
    # Test 1: YOLO Model
    results.append(("YOLO Model", test_yolo_model()))
    print()
    
    # Test 2: Face Recognition
    results.append(("Face Recognition", test_face_recognition()))
    print()
    
    # Test 3: Camera
    results.append(("Camera Access", test_camera()))
    print()
    
    # Summary
    print("📊 Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    total_pass = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")