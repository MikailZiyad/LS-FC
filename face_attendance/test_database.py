#!/usr/bin/env python3
"""
Test script untuk cek database connection dan struktur
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
            
            # Cek absensi hari ini
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
            
            if 'employees' in data:
                print(f"✅ Jumlah karyawan di JSON: {len(data['employees'])}")
                
                # Cek sample data
                if data['employees']:
                    sample = data['employees'][0]
                    print(f"✅ Sample employee: {sample.get('name', 'Unknown')}")
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

def test_embeddings_folder():
    """Test folder embeddings"""
    print("\n🧪 TESTING EMBEDDINGS FOLDER")
    print("=" * 50)
    
    embeddings_path = Path("data/models/embeddings")
    
    try:
        if embeddings_path.exists():
            # Hitung file embeddings
            embedding_files = list(embeddings_path.glob("*.pkl"))
            print(f"✅ Folder embeddings ditemukan: {embeddings_path}")
            print(f"✅ Jumlah file embeddings: {len(embedding_files)}")
            
            if embedding_files:
                print(f"✅ Sample files: {[f.name for f in embedding_files[:3]]}")
            
            return True
        else:
            print(f"⚠️ Folder embeddings tidak ditemukan: {embeddings_path}")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False

if __name__ == "__main__":
    db_ok = test_database_connection()
    json_ok = test_json_database()
    embeddings_ok = test_embeddings_folder()
    
    print("\n" + "=" * 50)
    if db_ok and json_ok and embeddings_ok:
        print("🎉 SEMUA DATABASE TEST BERHASIL!")
    else:
        print("⚠️ ADA BEBERAPA MASALAH DATABASE")
        
    print("\n📊 RINGKASAN:")
    print(f"Database SQLite: {'✅' if db_ok else '❌'}")
    print(f"Database JSON: {'✅' if json_ok else '❌'}")
    print(f"Folder Embeddings: {'✅' if embeddings_ok else '❌'}")
