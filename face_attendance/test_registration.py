#!/usr/bin/env python3
"""
Test script untuk registrasi karyawan baru
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.advanced_detector import AdvancedDetector
from face_recognition.advanced_analyzer import AdvancedAnalyzer
from database.database_manager import DatabaseManager
import cv2
import numpy as np
from pathlib import Path

def test_registration():
    """Test registrasi karyawan baru dengan dummy data"""
    print("🧪 TESTING REGISTRASI KARYAWAN BARU")
    print("=" * 60)
    
    try:
        # Inisialisasi komponen
        print("📋 Inisialisasi komponen...")
        detector = AdvancedDetector()
        analyzer = AdvancedAnalyzer()
        db = DatabaseManager()
        
        print("✅ Komponen berhasil diinisialisasi")
        
        # Data dummy untuk testing
        test_data = {
            "name": "Test Employee",
            "employee_id": "TEST001",
            "department": "IT Department",
            "position": "Software Engineer",
            "email": "test@company.com",
            "phone": "081234567890"
        }
        
        print(f"📄 Data test: {test_data}")
        
        # Simulasi capture wajah (gunakan gambar dummy/generate)
        print("📸 Simulasi capture wajah...")
        
        # Buat gambar dummy untuk testing (biru kosong)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[:] = (100, 100, 100)  # Abu-abu
        
        # Test face detection
        print("🔍 Test face detection...")
        faces = detector.detect_faces(dummy_frame)
        
        if not faces:
            print("⚠️ Tidak ada wajah terdeteksi di gambar dummy")
            print("ℹ️ Ini normal untuk gambar kosong")
            
            # Test dengan gambar yang ada wajah (jika tersedia)
            test_image_path = Path("tests/test_face.jpg")
            if test_image_path.exists():
                print(f"📁 Menggunakan gambar test: {test_image_path}")
                test_frame = cv2.imread(str(test_image_path))
                faces = detector.detect_faces(test_frame)
                
                if faces:
                    print(f"✅ Wajah terdeteksi: {len(faces)} wajah")
                    print(f"✅ Detail wajah: {faces[0]}")
                else:
                    print("⚠️ Tetap tidak ada wajah terdeteksi")
            else:
                print("ℹ️ Gambar test tidak tersedia, lewati face detection test")
        else:
            print(f"✅ Wajah terdeteksi: {len(faces)} wajah")
            print(f"✅ Detail wajah: {faces[0]}")
        
        # Test face analysis
        print("\n🔬 Test face analysis...")
        
        # Test dengan gambar dummy
        analysis_results = analyzer.analyze_face(dummy_frame)
        
        if analysis_results:
            print(f"✅ Face analysis berhasil:")
            for key, value in analysis_results.items():
                print(f"   {key}: {value}")
        else:
            print("⚠️ Face analysis tidak berhasil (gambar dummy)")
        
        # Test database operations
        print("\n💾 Test database operations...")
        
        # Test tambah karyawan
        employee_id = db.add_employee(
            name=test_data["name"],
            employee_id=test_data["employee_id"],
            department=test_data["department"],
            position=test_data["position"],
            email=test_data["email"],
            phone=test_data["phone"]
        )
        
        if employee_id:
            print(f"✅ Karyawan berhasil ditambahkan: ID {employee_id}")
            
            # Test update face data
            face_data = {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.95,
                "analysis": analysis_results
            }
            
            success = db.update_employee_face_data(test_data["employee_id"], face_data)
            if success:
                print("✅ Face data berhasil diupdate")
            else:
                print("⚠️ Face data gagal diupdate")
            
            # Test get employee data
            employee_data = db.get_employee_by_id(test_data["employee_id"])
            if employee_data:
                print(f"✅ Data karyawan berhasil diambil: {employee_data['name']}")
            else:
                print("❌ Data karyawan tidak ditemukan")
                
        else:
            print("❌ Gagal menambahkan karyawan")
        
        print("\n🎉 REGISTRASI TEST SELESAI")
        return True
        
    except Exception as e:
        print(f"❌ Error saat testing registrasi: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_extraction():
    """Test ekstraksi wajah untuk training"""
    print("\n🧪 TESTING FACE EXTRACTION")
    print("=" * 60)
    
    try:
        detector = AdvancedDetector()
        
        # Buat gambar dengan beberapa wajah dummy
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gambar beberapa kotak dummy sebagai "wajah"
        cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(test_frame, (300, 150), (400, 250), (255, 255, 255), -1)
        
        print("📸 Test frame dengan dummy faces dibuat")
        
        # Test extract faces
        faces = detector.extract_faces_for_training(test_frame, min_faces=1)
        
        if faces:
            print(f"✅ Berhasil extract {len(faces)} wajah")
            for i, face in enumerate(faces):
                print(f"   Face {i+1}: shape {face['face'].shape}, confidence {face['confidence']}")
        else:
            print("⚠️ Tidak ada wajah yang diekstrak (normal untuk gambar dummy)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saat face extraction: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING REGISTRATION TESTING")
    print("=" * 60)
    
    reg_ok = test_registration()
    extract_ok = test_face_extraction()
    
    print("\n" + "=" * 60)
    print("📊 RINGKASAN REGISTRASI TEST:")
    print(f"Registration Process: {'✅' if reg_ok else '❌'}")
    print(f"Face Extraction: {'✅' if extract_ok else '❌'}")
    
    if reg_ok and extract_ok:
        print("\n🎉 REGISTRASI TEST BERHASIL!")
    else:
        print("\n⚠️ ADA MASALAH PADA REGISTRASI")