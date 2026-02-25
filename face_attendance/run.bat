#!/usr/bin/env python3
"""
Launcher script untuk Face Attendance System
Akan otomatis menggunakan virtual environment yang ada di folder env_attendance
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Path ke virtual environment
    venv_path = current_dir / "env_attendance"
    python_path = venv_path / "Scripts" / "python.exe"
    
    # Cek apakah virtual environment ada
    if not python_path.exists():
        print("❌ Virtual environment tidak ditemukan!")
        print("📝 Silakan jalankan setup terlebih dahulu:")
        print("   python -m venv env_attendance")
        print("   env_attendance\\Scripts\\pip.exe install -r requirements.txt")
        return 1
    
    # Cek apakah main.py ada
    main_script = current_dir / "main.py"
    if not main_script.exists():
        print("❌ main.py tidak ditemukan!")
        return 1
    
    print("🚀 Memulai Face Attendance System...")
    print(f"📁 Folder: {current_dir}")
    print(f"🐍 Python: {python_path}")
    
    # Jalankan main.py dengan virtual environment Python
    try:
        result = subprocess.run(
            [str(python_path), str(main_script)],
            cwd=str(current_dir),
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Error saat menjalankan aplikasi: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Aplikasi dihentikan oleh user")
        return 0

if __name__ == "__main__":
    sys.exit(main())