#!/usr/bin/env python3
"""
Advanced Registration UI with DeepFace Master Integration
Enhanced with multi-pose training, quality validation, and anti-spoofing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import time
from typing import List, Dict, Any

from face_recognition.advanced_trainer import AdvancedFaceTrainer
from face_recognition.advanced_detector import AdvancedFaceDetector
from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
from config.advanced_config import ADVANCED_CONFIG
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from models.database import DatabaseManager

class AdvancedRegistrationUI:
    def __init__(self):
        self.root = None
        self.trainer = None
        self.detector = None
        self.recognizer = None
        self.db = DatabaseManager()
        
        # Configuration
        self.config = ADVANCED_CONFIG
        self.current_pose = "front"
        self.pose_samples = {pose: [] for pose in self.config["face_poses"]}
        self.training_metadata = {}
        
        # Camera and capture
        self.cap = None
        self.is_capturing = False
        self.capture_count = 0
        self.quality_scores = []
        
        # Advanced settings
        self.enable_multi_pose = True
        self.enable_quality_validation = True
        self.enable_anti_spoofing = True
        self.selected_model = "ArcFace"
        self.selected_detector = "retinaface"
        
    def run(self):
        """Run the advanced registration UI"""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("📝 Advanced Face Registration - DeepFace Master")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        self.setup_ui()
        self.initialize_systems()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def setup_ui(self):
        """Setup the advanced UI components"""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        self.setup_header(main_frame)
        
        # Configuration panel
        self.setup_config_panel(main_frame)
        
        # Main content area
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Camera and controls
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.setup_camera_panel(left_frame)
        
        # Right panel - Status and samples
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.setup_status_panel(right_frame)
        
        # Progress and controls
        self.setup_progress_panel(main_frame)
    
    def setup_header(self, parent):
        """Setup header with title and system info"""
        header_frame = ctk.CTkFrame(parent)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        title_label = ctk.CTkLabel(header_frame, 
                                   text="📝 Advanced Face Registration System",
                                   font=("Arial", 24, "bold"))
        title_label.pack(pady=10)
        
        subtitle_label = ctk.CTkLabel(header_frame,
                                    text="DeepFace Master Integration • Multi-Pose Training • Quality Validation • Anti-Spoofing",
                                    font=("Arial", 12))
        subtitle_label.pack(pady=5)
        
        # System status
        status_frame = ctk.CTkFrame(header_frame)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.model_status_label = ctk.CTkLabel(status_frame, text="Model: Not loaded")
        self.model_status_label.pack(side="left", padx=10)
        
        self.detector_status_label = ctk.CTkLabel(status_frame, text="Detector: Not loaded")
        self.detector_status_label.pack(side="left", padx=10)
        
        self.camera_status_label = ctk.CTkLabel(status_frame, text="Camera: Not connected")
        self.camera_status_label.pack(side="right", padx=10)
    
    def setup_config_panel(self, parent):
        """Setup configuration panel"""
        config_frame = ctk.CTkFrame(parent)
        config_frame.pack(fill="x", padx=10, pady=10)
        
        config_label = ctk.CTkLabel(config_frame, text="⚙️ Configuration",
                                  font=("Arial", 16, "bold"))
        config_label.pack(pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(config_frame)
        model_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(model_frame, text="Recognition Model:").pack(side="left", padx=10)
        self.model_var = ctk.StringVar(value="ArcFace")
        model_combo = ctk.CTkComboBox(model_frame, variable=self.model_var,
                                    values=["ArcFace", "Facenet512", "GhostFaceNet", "VGG-Face", "SFace"])
        model_combo.pack(side="left", padx=10)
        
        # Detector selection
        detector_frame = ctk.CTkFrame(config_frame)
        detector_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(detector_frame, text="Detection Backend:").pack(side="left", padx=10)
        self.detector_var = ctk.StringVar(value="retinaface")
        detector_combo = ctk.CTkComboBox(detector_frame, variable=self.detector_var,
                                       values=["retinaface", "yolov8n", "yolov8m", "mtcnn", "mediapipe"])
        detector_combo.pack(side="left", padx=10)
        
        # Feature toggles
        features_frame = ctk.CTkFrame(config_frame)
        features_frame.pack(fill="x", padx=20, pady=10)
        
        self.multi_pose_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(features_frame, text="Multi-pose Training", 
                         variable=self.multi_pose_var).pack(side="left", padx=10)
        
        self.quality_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(features_frame, text="Quality Validation", 
                         variable=self.quality_var).pack(side="left", padx=10)
        
        self.spoofing_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(features_frame, text="Anti-Spoofing", 
                         variable=self.spoofing_var).pack(side="left", padx=10)
    
    def setup_camera_panel(self, parent):
        """Setup camera panel with live feed"""
        camera_frame = ctk.CTkFrame(parent)
        camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Camera label
        self.camera_label = ctk.CTkLabel(camera_frame, text="📷 Camera Feed",
                                        font=("Arial", 16, "bold"))
        self.camera_label.pack(pady=10)
        
        # Video display
        self.video_label = ctk.CTkLabel(camera_frame, text="Camera not connected")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Pose indicator
        self.pose_frame = ctk.CTkFrame(camera_frame)
        self.pose_frame.pack(fill="x", padx=10, pady=10)
        
        self.pose_label = ctk.CTkLabel(self.pose_frame, 
                                      text="Current Pose: Front",
                                      font=("Arial", 14))
        self.pose_label.pack(pady=5)
        
        # Pose progress
        self.pose_progress = ctk.CTkProgressBar(self.pose_frame)
        self.pose_progress.pack(fill="x", padx=20, pady=5)
        self.pose_progress.set(0)
        
        # Camera controls
        controls_frame = ctk.CTkFrame(camera_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.connect_button = ctk.CTkButton(controls_frame, text="Connect Camera",
                                           command=self.connect_camera)
        self.connect_button.pack(side="left", padx=5)
        
        self.capture_button = ctk.CTkButton(controls_frame, text="Capture Sample",
                                           command=self.capture_sample,
                                           state="disabled")
        self.capture_button.pack(side="left", padx=5)
        
        self.auto_capture_button = ctk.CTkButton(controls_frame, text="Auto Capture",
                                              command=self.start_auto_capture)
        self.auto_capture_button.pack(side="left", padx=5)
    
    def setup_status_panel(self, parent):
        """Setup status and samples panel"""
        status_frame = ctk.CTkFrame(parent)
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Employee info section
        info_frame = ctk.CTkFrame(status_frame)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(info_frame, text="👤 Employee Information",
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        # Employee ID
        id_frame = ctk.CTkFrame(info_frame)
        id_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(id_frame, text="Employee ID:").pack(side="left", padx=10)
        self.employee_id_entry = ctk.CTkEntry(id_frame, placeholder_text="EMP001")
        self.employee_id_entry.pack(side="left", fill="x", expand=True, padx=10)
        
        # Employee name
        name_frame = ctk.CTkFrame(info_frame)
        name_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(name_frame, text="Employee Name:").pack(side="left", padx=10)
        self.employee_name_entry = ctk.CTkEntry(name_frame, placeholder_text="John Doe")
        self.employee_name_entry.pack(side="left", fill="x", expand=True, padx=10)
        
        # Department
        dept_frame = ctk.CTkFrame(info_frame)
        dept_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(dept_frame, text="Department:").pack(side="left", padx=10)
        self.department_entry = ctk.CTkEntry(dept_frame, placeholder_text="IT Department")
        self.department_entry.pack(side="left", fill="x", expand=True, padx=10)
        
        # Samples section
        samples_frame = ctk.CTkFrame(status_frame)
        samples_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(samples_frame, text="📸 Sample Collection Status",
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        # Pose samples display
        self.pose_frames = {}
        for i, pose in enumerate(self.config["face_poses"]):
            pose_frame = ctk.CTkFrame(samples_frame)
            pose_frame.pack(fill="x", padx=20, pady=5)
            
            pose_label = ctk.CTkLabel(pose_frame, text=f"{pose.title()}: 0/5")
            pose_label.pack(side="left", padx=10)
            
            progress = ctk.CTkProgressBar(pose_frame)
            progress.pack(side="left", fill="x", expand=True, padx=10)
            progress.set(0)
            
            self.pose_frames[pose] = {
                "label": pose_label,
                "progress": progress,
                "samples": []
            }
    
    def setup_progress_panel(self, parent):
        """Setup progress and control panel"""
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        # Quality metrics
        quality_frame = ctk.CTkFrame(progress_frame)
        quality_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(quality_frame, text="📊 Quality Metrics",
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.quality_label = ctk.CTkLabel(quality_frame, text="Average Quality: N/A")
        self.quality_label.pack(side="left", padx=10)
        
        self.spoofing_label = ctk.CTkLabel(quality_frame, text="Anti-Spoofing: N/A")
        self.spoofing_label.pack(side="left", padx=10)
        
        # Overall progress
        self.overall_progress = ctk.CTkProgressBar(progress_frame)
        self.overall_progress.pack(fill="x", padx=40, pady=10)
        self.overall_progress.set(0)
        
        # Action buttons
        action_frame = ctk.CTkFrame(progress_frame)
        action_frame.pack(fill="x", padx=20, pady=10)
        
        self.complete_button = ctk.CTkButton(action_frame, text="✅ Complete Registration",
                                           command=self.complete_registration,
                                           state="disabled")
        self.complete_button.pack(side="left", padx=5)
        
        self.reset_button = ctk.CTkButton(action_frame, text="🔄 Reset",
                                         command=self.reset_registration)
        self.reset_button.pack(side="left", padx=5)
        
        self.back_button = ctk.CTkButton(action_frame, text="⬅️ Back to Main",
                                        command=self.back_to_main)
        self.back_button.pack(side="right", padx=5)
    
    def initialize_systems(self):
        """Initialize advanced systems"""
        try:
            # Initialize trainer with current configuration
            self.trainer = AdvancedFaceTrainer(
                recognition_model=self.selected_model,
                detection_backend=self.selected_detector
            )
            
            # Initialize detector for real-time quality checking
            self.detector = AdvancedFaceDetector(
                primary_backend=self.selected_detector
            )
            
            # Initialize recognizer for anti-spoofing
            self.recognizer = AdvancedFaceRecognizer(
                model_name=self.selected_model,
                detector_backend=self.selected_detector,
                enable_anti_spoofing=self.enable_anti_spoofing
            )
            
            self.update_status_labels()
            print("✅ Advanced systems initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize systems: {e}")
            print(f"❌ Initialization error: {e}")
    
    def update_status_labels(self):
        """Update system status labels"""
        if self.trainer:
            self.model_status_label.configure(
                text=f"Model: {self.trainer.recognizer.model_name}")
            self.detector_status_label.configure(
                text=f"Detector: {self.trainer.recognizer.detector_backend}")
    
    def connect_camera(self):
        """Connect to camera"""
        try:
            if self.cap is None:
                opened = False
                tried = []
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                indices = [CAMERA_INDEX, 0, 1, 2]
                for backend in backends:
                    for idx in indices:
                        try:
                            cap = cv2.VideoCapture(idx, backend)
                            tried.append(f"{idx}:{backend}")
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                                ret, _ = cap.read()
                                if ret:
                                    self.cap = cap
                                    opened = True
                                    break
                                else:
                                    cap.release()
                        except Exception:
                            continue
                    if opened:
                        break

                if opened and self.cap and self.cap.isOpened():
                    self.camera_status_label.configure(text="Camera: Connected")
                    self.connect_button.configure(text="Disconnect Camera")
                    self.capture_button.configure(state="normal")
                    self.start_video_loop()
                else:
                    messagebox.showerror("Camera Error", f"Failed to connect to camera.\nTried: {', '.join(tried)}")
            else:
                self.disconnect_camera()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Camera connection failed: {e}")
    
    def disconnect_camera(self):
        """Disconnect camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_status_label.configure(text="Camera: Not connected")
        self.connect_button.configure(text="Connect Camera")
        self.capture_button.configure(state="disabled")
        self.video_label.configure(text="Camera not connected")
    
    def start_video_loop(self):
        """Start video capture loop"""
        if self.cap and self.cap.isOpened():
            self.is_capturing = True
            self.update_video_frame()
    
    def update_video_frame(self):
        """Update video frame with deepface-master detector and enhanced visualization"""
        if not self.is_capturing or not self.cap:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret and self.detector:
                # Gunakan detektor deepface-master yang lebih canggih
                # Pilih backend terbaik untuk registrasi: retinaface atau yolov8n
                # Set backend yang akan digunakan
                self.detector.primary_backend = "retinaface"  # Paling akurat untuk registrasi
                
                # Deteksi wajah dengan deepface-master
                faces = self.detector.detect_faces_with_attributes(frame)
                
                # Frame hasil dengan visualisasi
                result_frame = frame.copy()
                
                # Tentukan status capture
                capture_status = "waiting"
                if self.is_auto_capturing:
                    capture_status = "capturing"
                elif len(self.captured_samples) >= self.samples_per_person:
                    capture_status = "captured"
                
                # Gambar border untuk setiap wajah yang terdeteksi
                for face_data in faces:
                    # Gunakan border enhanced untuk registrasi
                    result_frame = self.detector.draw_registration_border(
                        result_frame, 
                        face_data, 
                        capture_status=capture_status
                    )
                
                # Validasi kualitas untuk setiap wajah
                for face in faces:
                    face_roi = face["face"]
                    quality_score, issues = self.recognizer.get_face_quality_score(face_roi) if self.recognizer else (50, ["Recognizer not available"])
                    
                    # Anti-spoofing check
                    is_real, spoof_confidence = self.recognizer.detect_spoofing(face_roi) if self.recognizer else (True, 0.0)
                    
                    # Update status berdasarkan hasil validasi
                    if not is_real:
                        self.status_label.configure(text="⚠️ Spoofing detected!", text_color="red")
                    elif quality_score < 70:
                        self.status_label.configure(text=f"⚠️ Low quality ({quality_score:.1f})", text_color="orange")
                    else:
                        self.status_label.configure(text=f"✅ Good quality ({quality_score:.1f})", text_color="green")
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                display_width = 400
                display_height = 300
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
                
                # Convert to PhotoImage
                photo = self.array_to_photoimage(frame_resized)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Schedule next frame
                if self.root:
                    self.root.after(30, self.update_video_frame)
                
        except Exception as e:
            print(f"Video update error: {e}")
            if self.root:
                self.root.after(1000, self.update_video_frame)
    
    def array_to_photoimage(self, array):
        """Convert numpy array to PhotoImage"""
        from PIL import Image, ImageTk
        image = Image.fromarray(array)
        return ImageTk.PhotoImage(image=image)
    
    def capture_sample(self):
        """Capture a face sample with quality validation"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Camera Error", "Please connect camera first")
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Capture Error", "Failed to capture frame")
                return
            
            # Detect face
            if not self.detector:
                messagebox.showwarning("Detector Error", "Detector not initialized")
                return
                
            faces = self.detector.detect_faces(frame)
            if not faces:
                messagebox.showwarning("No Face", "No face detected in frame")
                return
            
            # Use the best face (highest confidence)
            best_face = max(faces, key=lambda x: x["confidence"])
            x, y, w, h = best_face["bbox"]
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Validate face quality
            if self.enable_quality_validation and self.trainer and self.trainer.recognizer:
                is_valid, message = self.trainer.recognizer.validate_face_quality(face_roi)
                if not is_valid:
                    messagebox.showwarning("Quality Check", f"Face quality insufficient: {message}")
                    return
            
            # Anti-spoofing check
            if self.enable_anti_spoofing and self.recognizer:
                is_real, confidence = self.recognizer.detect_spoofing(face_roi)
                if not is_real:
                    messagebox.showwarning("Spoofing Detected", f"Potential spoofing attack detected (confidence: {confidence:.2f})")
                    return
            
            # Add sample to current pose
            self.add_sample_to_pose(face_roi, best_face)
            
            # Update UI
            self.update_pose_display()
            self.update_quality_metrics(face_roi)
            
            messagebox.showinfo("Success", "Face sample captured successfully!")
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Failed to capture sample: {e}")
    
    def add_sample_to_pose(self, face_image, face_data):
        """Add sample to current pose collection"""
        pose_samples = self.pose_frames[self.current_pose]["samples"]
        
        if len(pose_samples) >= 5:  # Max 5 samples per pose
            messagebox.showinfo("Limit Reached", f"Maximum samples collected for {self.current_pose} pose")
            return
        
        # Calculate quality score
        quality_score, issues = self.recognizer.get_face_quality_score(face_image) if self.recognizer else (50, ["Recognizer not available"])
        
        sample_data = {
            "image": face_image,
            "quality": quality_score,
            "issues": issues,
            "timestamp": datetime.now(),
            "face_data": face_data
        }
        
        pose_samples.append(sample_data)
        self.quality_scores.append(quality_score)
    
    def update_pose_display(self):
        """Update pose display with current samples"""
        for pose, data in self.pose_frames.items():
            samples = data["samples"]
            count = len(samples)
            
            # Update label
            data["label"].configure(text=f"{pose.title()}: {count}/5")
            
            # Update progress bar
            progress = count / 5.0
            data["progress"].set(progress)
    
    def update_quality_metrics(self, face_image=None):
        """Update quality metrics display"""
        if self.quality_scores:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores)
            self.quality_label.configure(text=f"Average Quality: {avg_quality:.1f}/100")
        
        if self.enable_anti_spoofing and face_image is not None and self.recognizer:
            is_real, confidence = self.recognizer.detect_spoofing(face_image)
            status = "Real" if is_real else "Spoof"
            self.spoofing_label.configure(text=f"Anti-Spoofing: {status} ({confidence:.2f})")
    
    def start_auto_capture(self):
        """Start automatic sample capture"""
        # This would implement automatic capture based on quality metrics
        # For now, just show a message
        messagebox.showinfo("Auto Capture", "Auto capture feature coming soon!")
    
    def complete_registration(self):
        """Complete the advanced registration process"""
        # Validate inputs
        employee_id = self.employee_id_entry.get().strip()
        employee_name = self.employee_name_entry.get().strip()
        department = self.department_entry.get().strip()
        
        if not employee_id or not employee_name:
            messagebox.showwarning("Validation Error", "Employee ID and Name are required")
            return
        
        # Check if we have enough samples
        total_samples = sum(len(data["samples"]) for data in self.pose_frames.values())
        if total_samples < 5:
            messagebox.showwarning("Insufficient Samples", 
                                 f"Need at least 5 samples, have {total_samples}")
            return
        
        if not self.trainer:
            messagebox.showerror("Trainer Error", "Trainer not initialized")
            return
            
        try:
            # Start registration with trainer
            self.trainer.start_registration(employee_id, employee_name, {
                "department": department,
                "registration_date": datetime.now().isoformat(),
                "total_samples": total_samples,
                "config_used": self.config
            })
            
            # Collect all face samples
            all_samples = []
            for pose, data in self.pose_frames.items():
                for sample in data["samples"]:
                    all_samples.append(sample["image"])
            
            # Process samples with quality validation
            if not self.trainer:
                messagebox.showerror("Trainer Error", "Trainer not initialized")
                return
                
            for i, sample_image in enumerate(all_samples):
                self.trainer.add_face_sample(sample_image, {
                    "sample_index": i,
                    "total_samples": len(all_samples)
                })
            
            # Complete registration
            if not self.trainer:
                messagebox.showerror("Trainer Error", "Trainer not initialized")
                return
                
            result = self.trainer.complete_registration()
            
            if result["success"]:
                avg_quality = result["avg_quality"]
                messagebox.showinfo("Registration Success", 
                                  f"Employee registered successfully!\n"
                                  f"Average quality: {avg_quality:.1f}%\n"
                                  f"Samples used: {result['samples_used']}")
                
                # Reset for next registration
                self.reset_registration()
                
            else:
                messagebox.showerror("Registration Failed", 
                                   f"Registration failed: {result['message']}")
                
        except Exception as e:
            messagebox.showerror("Registration Error", f"Registration error: {e}")
            print(f"Registration error: {e}")
    
    def reset_registration(self):
        """Reset registration state"""
        # Clear employee info
        self.employee_id_entry.delete(0, tk.END)
        self.employee_name_entry.delete(0, tk.END)
        self.department_entry.delete(0, tk.END)
        
        # Clear samples
        for pose_data in self.pose_frames.values():
            pose_data["samples"].clear()
        
        self.quality_scores.clear()
        self.capture_count = 0
        
        # Update displays
        self.update_pose_display()
        self.quality_label.configure(text="Average Quality: N/A")
        self.spoofing_label.configure(text="Anti-Spoofing: N/A")
        self.overall_progress.set(0)
        
        # Disable complete button
        self.complete_button.configure(state="disabled")
    
    def back_to_main(self):
        """Return to main menu"""
        self.on_closing()
        # Import and run main menu
        from main_advanced import main
        main()
    
    def on_closing(self):
        """Handle window closing"""
        self.disconnect_camera()
        if self.root:
            self.root.destroy()

# Example usage function
def run_advanced_registration():
    """Run the advanced registration system"""
    app = AdvancedRegistrationUI()
    app.run()

if __name__ == "__main__":
    run_advanced_registration()
