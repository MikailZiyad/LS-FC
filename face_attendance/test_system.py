#!/usr/bin/env python3
"""
Test script untuk cek database connection dan struktur - VERSI FIXED
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

def test_database_connection():
    """Test koneksi ke SQLite database"""
    print("🧪 TESTING DATABASE CONNECTION")
    print("=" * 50)
    
    db_path = Path("data/face_attendance.db")
    
    try:
        # Test koneksi
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print(f"✅ Koneksi database berhasil: {db_path}")
        
        # Cek tabel yang ada
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"✅ Tabel yang ditemukan: {[table[0] for table in tables]}")
        
        # Cek struktur tabel employees
        if 'employees' in [table[0] for table in tables]:
            cursor.execute("PRAGMA table_info(employees);")
            columns = cursor.fetchall()
            print(f"✅ Struktur tabel employees: {[col[1] for col in columns]}")
            
            # Hitung jumlah karyawan
            cursor.execute("SELECT COUNT(*) FROM employees;")
            count = cursor.fetchone()[0]
            print(f"✅ Jumlah karyawan terdaftar: {count}")
        
        # Cek struktur tabel attendance
        if 'attendance' in [table[0] for table in tables]:
            cursor.execute("PRAGMA table_info(attendance);")
            columns = cursor.fetchall()
            print(f"✅ Struktur tabel attendance: {[col[1] for col in columns]}")
            
            # Hitung jumlah absensi
            cursor.execute("SELECT COUNT(*) FROM attendance;")
            count = cursor.fetchone()[0]
            print(f"✅ Jumlah record absensi: {count}")
            
            # Cek absensi hari ini - FIX: gunakan attendance_date
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date = ?;", (today,))
            today_count = cursor.fetchone()[0]
            print(f"✅ Absensi hari ini ({today}): {today_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_json_database():
    """Test JSON database untuk face analysis"""
    print("\n🧪 TESTING JSON DATABASE")
    print("=" * 50)
    
    json_path = Path("data/employee_database.json")
    
    try:
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ JSON database ditemukan: {json_path}")
            print(f"✅ Struktur data: {list(data.keys())}")
            
            # Cek apakah data berisi employee ID sebagai key
            employee_count = len(data)
            print(f"✅ Jumlah karyawan di JSON: {employee_count}")
            
            if data:
                # Ambil sample data
                sample_key = list(data.keys())[0]
                sample = data[sample_key]
                print(f"✅ Sample employee ID {sample_key}: {sample.get('name', 'Unknown')}")
                if 'face_analysis' in sample:
                    print(f"✅ Face analysis tersedia: {list(sample['face_analysis'].keys())}")
            
            return True
        else:
            print(f"⚠️ JSON database tidak ditemukan: {json_path}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ JSON error: {e}")
        return False

def test_model_folders():
    """Test folder models dan embeddings"""
    print("\n🧪 TESTING MODEL FOLDERS")
    print("=" * 50)
    
    # Test embeddings folder
    embeddings_path = Path("data/models/embeddings")
    deepface_weights_path = Path("data/models/.deepface/weights")
    
    try:
        # Cek deepface weights (yang sebenarnya digunakan)
        if deepface_weights_path.exists():
            weight_files = list(deepface_weights_path.glob("*"))
            print(f"✅ DeepFace weights folder: {len(weight_files)} files")
            
            # Cek beberapa model penting
            important_models = ['retinaface.h5', 'age_model_weights.h5', 'gender_model_weights.h5']
            for model in important_models:
                model_path = deepface_weights_path / model
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024*1024)
                    print(f"✅ {model}: {size_mb:.1f} MB")
                else:
                    print(f"⚠️ {model}: tidak ditemukan")
        else:
            print(f"⚠️ DeepFace weights folder tidak ditemukan: {deepface_weights_path}")
        
        # Cek embeddings folder
        if embeddings_path.exists():
            embedding_files = list(embeddings_path.glob("*.pkl"))
            print(f"✅ Embeddings folder: {len(embedding_files)} files")
        else:
            print(f"ℹ️ Embeddings folder belum ada (akan dibuat otomatis): {embeddings_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model folders error: {e}")
        return False

def test_camera_connection():
    """Test koneksi kamera"""
    print("\n🧪 TESTING CAMERA CONNECTION")
    print("=" * 50)
    
    try:
        import cv2
        
        # Test kamera default (index 0)
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ Kamera terdeteksi: {width}x{height}")
                print(f"✅ Warna frame: {'Color' if len(frame.shape) == 3 else 'Grayscale'}")
            else:
                print("⚠️ Kamera terbuka tapi tidak bisa membaca frame")
        else:
            print("❌ Kamera tidak bisa dibuka")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    
    db_ok = test_database_connection()
    json_ok = test_json_database()
    models_ok = test_model_folders()
    camera_ok = test_camera_connection()
    
    print("\n" + "=" * 60)
    print("📊 RINGKASAN TEST:")
    print(f"Database SQLite: {'✅' if db_ok else '❌'}")
    print(f"Database JSON: {'✅' if json_ok else '❌'}")
    print(f"Model Folders: {'✅' if models_ok else '❌'}")
    print(f"Camera Connection: {'✅' if camera_ok else '❌'}")
    
    total_tests = 4
    passed_tests = sum([db_ok, json_ok, models_ok, camera_ok])
    
    print(f"\n🎯 HASIL AKHIR: {passed_tests}/{total_tests} test berhasil")
    
    if passed_tests == total_tests:
        print("🎉 SEMUA SISTEM SIAP! Bisa mulai testing registrasi & absensi.")
    else:
        print("⚠️ BEBERAPA SISTEM PERLU DIPERBAIKI SEBELUM TESTING LANJUTAN.")