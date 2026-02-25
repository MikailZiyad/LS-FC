import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.advanced_registration_ui import AdvancedRegistrationUI
from ui.advanced_attendance_ui import AdvancedAttendanceUI
import customtkinter as ctk

def main():
    print("🚀 Memulai Face Attendance System")
    print("📁 Project root:", project_root)
    
    # Create main window
    root = ctk.CTk()
    root.title("Face Attendance System")
    root.geometry("400x300")
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Title
    title_label = ctk.CTkLabel(root, text="🎭 Face Attendance System", 
                              font=("Arial", 20, "bold"))
    title_label.pack(pady=30)
    
    # Description
    desc_label = ctk.CTkLabel(root, text="Sistem absensi berbasis pengenalan wajah", 
                             font=("Arial", 12))
    desc_label.pack(pady=10)
    
    # Buttons
    button_frame = ctk.CTkFrame(root)
    button_frame.pack(pady=30)
    
    def open_registration():
        root.destroy()
        registration_app = AdvancedRegistrationUI()
        registration_app.run()
    
    def open_attendance():
        root.destroy()
        attendance_app = AdvancedAttendanceUI()
        attendance_app.run()
    
    def exit_app():
        root.destroy()
    
    reg_button = ctk.CTkButton(button_frame, text="📝 Registrasi Karyawan", 
                              command=open_registration, width=200, height=40)
    reg_button.pack(pady=10)
    
    att_button = ctk.CTkButton(button_frame, text="📊 Absensi", 
                              command=open_attendance, width=200, height=40)
    att_button.pack(pady=10)
    
    exit_button = ctk.CTkButton(button_frame, text="❌ Keluar", 
                               command=exit_app, width=200, height=40, 
                               fg_color="red", hover_color="darkred")
    exit_button.pack(pady=10)
    
    # Footer
    footer_label = ctk.CTkLabel(root, text="Dibuat dengan ❤️ menggunakan DeepFace + YOLO", 
                               font=("Arial", 10))
    footer_label.pack(side="bottom", pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
