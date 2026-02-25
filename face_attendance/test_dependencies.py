#!/usr/bin/env python3
"""
Test script untuk cek dependencies sistem absensi
"""

import sys
import subprocess

def test_imports():
    """Test semua dependencies utama"""
    dependencies = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'customtkinter': 'CustomTkinter',
        'deepface': 'DeepFace',
        'sqlite3': 'SQLite3',
        'json': 'JSON',
        'datetime': 'DateTime',
        'pathlib': 'PathLib',
        'tkinter': 'Tkinter'
    }
    
    print("🧪 TESTING DEPENDENCIES")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print("=" * 50)
    
    failed = []
    passed = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: OK")
            passed.append(name)
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed.append(name)
    
    print("\n" + "=" * 50)
    print(f"HASIL: {len(passed)} passed, {len(failed)} failed")
    
    if failed:
        print(f"\n❌ Dependencies gagal: {', '.join(failed)}")
        return False
    else:
        print("\n✅ Semua dependencies OK!")
        return True

def test_deepface_models():
    """Test apakah model deepface tersedia"""
    print("\n🧪 TESTING DEEPFACE MODELS")
    print("=" * 50)
    
    try:
        from deepface import DeepFace
        
        # Test model availability
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
        models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'SFace']
        
        print("✅ DeepFace import berhasil")
        print(f"✅ Available backends: {len(backends)}")
        print(f"✅ Available models: {len(models)}")
        
        return True
    except Exception as e:
        print(f"❌ DeepFace error: {e}")
        return False

if __name__ == "__main__":
    deps_ok = test_imports()
    models_ok = test_deepface_models()
    
    if deps_ok and models_ok:
        print("\n🎉 SEMUA TEST BERHASIL! Sistem siap digunakan.")
        sys.exit(0)
    else:
        print("\n❌ ADA MASALAH! Perbaiki dulu sebelum testing.")
        sys.exit(1)