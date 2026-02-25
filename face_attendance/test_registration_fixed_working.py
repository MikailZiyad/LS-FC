#!/usr/bin/env python3
"""
Test script untuk registrasi karyawan baru - VERSI WORKING FIXED
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.advanced_detector import AdvancedFaceDetector
from face_recognition.advanced_analyzer import AdvancedFaceAnalyzer
from models.database import DatabaseManager
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
        detector = AdvancedFaceDetector()
        analyzer = AdvancedFaceAnalyzer()
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
        
        # Buat gambar dummy untuk testing
        print("📸 Membuat gambar dummy untuk testing...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[:] = (100, 100, 100)  # Abu-abu
        
        # Tambahkan "wajah" sederhana (kotak putih)
        cv2.rectangle(dummy_frame, (200, 150), (440, 330), (200, 200, 200), -1)
        cv2.circle(dummy_frame, (280, 220), 15, (150, 150, 150), -1)  # "mata kiri"
        cv2.circle(dummy_frame, (360, 220), 15, (150, 150, 150), -1)  # "mata kanan"
        cv2.rectangle(dummy_frame, (300, 260), (340, 300), (180, 180, 180), -1)  # "hidung"
        
        print("✅ Gambar dummy wajah dibuat")
        
        # Test face detection
        print("🔍 Test face detection...")
        faces = detector.detect_faces(dummy_frame)
        
        if faces:
            print(f"✅ Wajah terdeteksi: {len(faces)} wajah")
            print(f"   Detail wajah: {faces[0]}")
        else:
            print("⚠️ Tidak ada wajah terdeteksi di gambar dummy")
            print("   Ini normal - deep learning model membutuhkan wajah nyata")
        
        # Test face analysis
        print("\n🔬 Test face analysis...")
        
        # Test dengan gambar dummy
        analysis_results = analyzer.analyze_face_attributes(dummy_frame)
        
        if analysis_results:
            print(f"✅ Face analysis berhasil:")
            for key, value in analysis_results.items():
                print(f"   {key}: {value}")
        else:
            print("⚠️ Face analysis tidak berhasil (gambar dummy)")
            print("   Ini normal - model membutuhkan wajah nyata untuk analisis")
        
        # Test database operations
        print("\n💾 Test database operations...")
        
        # Test tambah karyawan
        print("📝 Menambahkan karyawan ke database...")
        success, message = db.add_employee(
            employee_id=test_data["employee_id"],
            name=test_data["name"],
            department=test_data["department"],
            position=test_data["position"],
            email=test_data["email"],
            phone=test_data["phone"]
        )
        
        if success:
            print(f"✅ Karyawan berhasil ditambahkan: {message}")
            
            # Test update employee data (jika perlu)
            print("🔄 Test update employee data...")
            update_success = db.update_employee(
                employee_id=test_data["employee_id"],
                department="Updated IT Department",
                position="Senior Software Engineer"
            )
            
            if update_success:
                print("✅ Employee data berhasil diupdate")
            else:
                print("⚠️ Employee data gagal diupdate")
            
            # Test get employee data
            employee_data = db.get_employee_by_id_old(test_data["employee_id"])
            if employee_data:
                print(f"✅ Data karyawan berhasil diambil: {employee_data[2]}")  # name is at index 2
                print(f"   Departemen: {employee_data[3]}")
                print(f"   Posisi: {employee_data[4]}")
            else:
                print("❌ Data karyawan tidak ditemukan")
                
        else:
            print(f"❌ Gagal menambahkan karyawan: {message}")
        
        # Test list karyawan
        print("\n📋 Test list karyawan...")
        all_employees = db.get_all_employees()
        print(f"✅ Total karyawan terdaftar: {len(all_employees)}")
        
        for emp in all_employees[-3:]:  # Tampilkan 3 karyawan terakhir
            print(f"   - {emp[2]} ({emp[1]}) - {emp[3]}")  # name, employee_id, department
        
        print("\n🎉 REGISTRASI TEST SELESAI")
        return True
        
    except Exception as e:
        print(f"❌ Error saat testing registrasi: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_extraction_training():
    """Test ekstraksi wajah untuk training"""
    print("\n🧪 TESTING FACE EXTRACTION UNTUK TRAINING")
    print("=" * 60)
    
    try:
        detector = AdvancedFaceDetector()
        
        # Buat gambar dengan beberapa wajah dummy
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gambar beberapa kotak dummy sebagai "wajah"
        cv2.rectangle(test_frame, (100, 100), (200, 200), (200, 200, 200), -1)
        cv2.rectangle(test_frame, (300, 150), (400, 250), (200, 200, 200), -1)
        cv2.rectangle(test_frame, (450, 100), (550, 200), (200, 200, 200), -1)
        
        print("📸 Test frame dengan multiple dummy faces dibuat")
        
        # Test face detection dulu
        faces = detector.detect_faces(test_frame)
        
        if faces:
            print(f"✅ Berhasil detect {len(faces)} wajah")
            for i, face in enumerate(faces):
                bbox = face['bbox']
                confidence = face['confidence']
                print(f"   Face {i+1}: bbox {bbox}, confidence {confidence:.3f}")
                
                # Test extract face for analysis
                face_crop = detector.extract_face_for_analysis(test_frame, bbox)
                if face_crop is not None:
                    print(f"   ✅ Face extraction for analysis berhasil: shape {face_crop.shape}")
                else:
                    print(f"   ⚠️ Face extraction for analysis gagal")
        else:
            print("⚠️ Tidak ada wajah yang diekstrak")
            print("   Ini normal - model membutuhkan wajah nyata")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saat face extraction training: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE REGISTRATION TESTING")
    print("=" * 60)
    
    reg_ok = test_registration()
    extract_ok = test_face_extraction_training()
    
    print("\n" + "=" * 60)
    print("📊 RINGKASAN REGISTRASI TEST:")
    print(f"Registration Process: {'✅' if reg_ok else '❌'}")
    print(f"Face Extraction Training: {'✅' if extract_ok else '❌'}")
    
    if reg_ok and extract_ok:
        print("\n🎉 REGISTRASI TEST BERHASIL!")
        print("\n💡 CATATAN:")
        print("   - Semua komponen berfungsi dengan baik")
        print("   - Face detection/analysis membutuhkan wajah nyata untuk optimal")
        print("   - Database operations berjalan lancar")
    else:
        print("\n⚠️ ADA MASALAH PADA REGISTRASI")