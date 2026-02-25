import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import customtkinter as ctk

# Import advanced UI modules (we'll create these)
from ui.advanced_registration_ui import AdvancedRegistrationUI
from ui.advanced_attendance_ui import AdvancedAttendanceUI

def main():
    print("🚀 Memulai Face Attendance System")
    print("📁 Project root:", project_root)
    print("🎯 Version: Advanced DeepFace Master Integration")
    
    # Create main window
    root = ctk.CTk()
    root.title("Face Attendance System - Advanced Edition")
    root.geometry("500x400")
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Title
    title_label = ctk.CTkLabel(root, text="🎭 Face Attendance System", 
                              font=("Arial", 24, "bold"))
    title_label.pack(pady=20)
    
    # Subtitle
    subtitle_label = ctk.CTkLabel(root, text="Advanced DeepFace Master Edition", 
                                 font=("Arial", 14, "italic"))
    subtitle_label.pack(pady=5)
    
    # Description
    desc_label = ctk.CTkLabel(root, text="Sistem absensi berbasis pengenalan wajah dengan teknologi canggih", 
                             font=("Arial", 12))
    desc_label.pack(pady=10)
    
    # Feature highlights
    features_frame = ctk.CTkFrame(root)
    features_frame.pack(pady=15, padx=30, fill="x")
    
    features = [
        "✅ Multi-model recognition (ArcFace, Facenet512, GhostFaceNet)",
        "🛡️ Anti-spoofing detection (FasNet)",
        "🎯 Ensemble recognition for higher accuracy",
        "📊 Face quality validation",
        "🔄 Intelligent fallback systems",
        "⚡ Real-time processing optimization"
    ]
    
    for feature in features:
        feature_label = ctk.CTkLabel(features_frame, text=feature, 
                                   font=("Arial", 11))
        feature_label.pack(pady=3, padx=10, anchor="w")
    
    # Buttons frame
    button_frame = ctk.CTkFrame(root)
    button_frame.pack(pady=20)
    
    def open_advanced_registration():
        root.destroy()
        registration_app = AdvancedRegistrationUI()
        registration_app.run()
    
    def open_advanced_attendance():
        root.destroy()
        attendance_app = AdvancedAttendanceUI()
        attendance_app.run()
    
    def open_legacy_registration():
        try:
            from ui.registration_ui import RegistrationUI
            root.destroy()
            registration_app = RegistrationUI()
            registration_app.run()
        except Exception as e:
            print(f"⚠️  Legacy Registration UI tidak tersedia: {e}")
            print("➡️  Membuka Advanced Registration sebagai pengganti.")
            root.destroy()
            registration_app = AdvancedRegistrationUI()
            registration_app.run()
    
    def open_legacy_attendance():
        try:
            from ui.attendance_ui import AttendanceUI
            root.destroy()
            attendance_app = AttendanceUI()
            attendance_app.run()
        except Exception as e:
            print(f"⚠️  Legacy Attendance UI tidak tersedia: {e}")
            print("➡️  Membuka Advanced Attendance sebagai pengganti.")
            root.destroy()
            attendance_app = AdvancedAttendanceUI()
            attendance_app.run()
    
    def exit_app():
        root.destroy()
    
    # Advanced buttons
    advanced_label = ctk.CTkLabel(button_frame, text="🚀 Advanced Mode", 
                                 font=("Arial", 14, "bold"))
    advanced_label.grid(row=0, column=0, columnspan=2, pady=5)
    
    adv_reg_button = ctk.CTkButton(button_frame, text="📝 Registrasi Canggih", 
                                   command=open_advanced_registration, 
                                   width=200, height=40,
                                   fg_color="#2E7D32", hover_color="#1B5E20")
    adv_reg_button.grid(row=1, column=0, padx=5, pady=5)
    
    adv_att_button = ctk.CTkButton(button_frame, text="📊 Absensi Canggih", 
                                   command=open_advanced_attendance, 
                                   width=200, height=40,
                                   fg_color="#1565C0", hover_color="#0D47A1")
    adv_att_button.grid(row=1, column=1, padx=5, pady=5)
    
    # Legacy buttons
    legacy_label = ctk.CTkLabel(button_frame, text="⚙️ Legacy Mode", 
                               font=("Arial", 14, "bold"))
    legacy_label.grid(row=2, column=0, columnspan=2, pady=(15, 5))
    
    leg_reg_button = ctk.CTkButton(button_frame, text="📝 Registrasi Lama", 
                                  command=open_legacy_registration, 
                                  width=200, height=40,
                                  fg_color="#546E7A", hover_color="#37474F")
    leg_reg_button.grid(row=3, column=0, padx=5, pady=5)
    
    leg_att_button = ctk.CTkButton(button_frame, text="📊 Absensi Lama", 
                                  command=open_legacy_attendance, 
                                  width=200, height=40,
                                  fg_color="#546E7A", hover_color="#37474F")
    leg_att_button.grid(row=3, column=1, padx=5, pady=5)
    
    # Exit button
    exit_button = ctk.CTkButton(button_frame, text="❌ Keluar", 
                               command=exit_app, width=410, height=40, 
                               fg_color="#D32F2F", hover_color="#B71C1C")
    exit_button.grid(row=4, column=0, columnspan=2, pady=10)
    
    # Footer
    footer_label = ctk.CTkLabel(root, text="Dibuat dengan ❤️ menggunakan DeepFace Master + YOLOv8", 
                               font=("Arial", 10))
    footer_label.pack(side="bottom", pady=15)
    
    root.mainloop()

def show_system_info():
    """Show system information and capabilities"""
    print("\n" + "="*60)
    print("🎭 FACE ATTENDANCE SYSTEM - ADVANCED EDITION")
    print("="*60)
    print("🚀 Features:")
    print("  • Multi-model face recognition")
    print("  • Anti-spoofing detection")
    print("  • Ensemble recognition")
    print("  • Face quality validation")
    print("  • Intelligent fallback systems")
    print("  • Real-time processing")
    print("\n📋 System Requirements:")
    print("  • Python 3.8+")
    print("  • OpenCV")
    print("  • DeepFace Master")
    print("  • YOLOv8")
    print("  • SQLite")
    print("  • CustomTkinter")
    print("="*60)

if __name__ == "__main__":
    try:
        show_system_info()
        main()
    except KeyboardInterrupt:
        print("\n👋 Program dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
