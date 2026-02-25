#!/usr/bin/env python3
"""
Advanced Attendance UI with DeepFace Master Integration
Enhanced with ensemble recognition, anti-spoofing, and real-time analytics
"""

import sys
import os
# Tambahkan path ke project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import time
from PIL import Image, ImageTk
from typing import List, Dict, Any

from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
from face_recognition.advanced_detector import AdvancedFaceDetector
from config.advanced_config import ADVANCED_CONFIG
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from models.database import DatabaseManager
from utils.enhanced_visualization import draw_enhanced_face_border

class AdvancedAttendanceUI:
    def __init__(self):
        self.root = None
        self.recognizer = None
        self.detector = None
        self.db = DatabaseManager()
        
        # Configuration
        self.config = ADVANCED_CONFIG
        
        # Camera and capture
        self.cap = None
        self.is_capturing = False
        self.current_frame = None
        
        # Recognition settings
        self.enable_ensemble = True
        self.enable_anti_spoofing = True
        self.selected_model = "ArcFace"
        self.selected_detector = "retinaface"
        self.confidence_threshold = 0.7
        
        # Analytics
        self.attendance_stats = {
            "total_recognitions": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "spoofing_attempts": 0,
            "average_confidence": 0.0
        }
        
        # Recognition history
        self.recognition_history = []
        self.max_history = 10
        
    def run(self):
        """Run the advanced attendance UI"""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("📊 Advanced Attendance System - DeepFace Master")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        self.setup_ui()
        self.initialize_systems()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def setup_ui(self):
        """Setup the advanced attendance UI"""
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
        
        # Left panel - Camera and recognition
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.setup_camera_panel(left_frame)
        
        # Middle panel - Recognition results
        middle_frame = ctk.CTkFrame(content_frame)
        middle_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.setup_results_panel(middle_frame)
        
        # Right panel - Analytics and history
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.setup_analytics_panel(right_frame)
        
        # Status bar
        self.setup_status_panel(main_frame)
    
    def setup_header(self, parent):
        """Setup header with title and system info"""
        header_frame = ctk.CTkFrame(parent)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        title_label = ctk.CTkLabel(header_frame, 
                                   text="📊 Advanced Attendance System",
                                   font=("Arial", 28, "bold"))
        title_label.pack(pady=10)
        
        subtitle_label = ctk.CTkLabel(header_frame,
                                    text="DeepFace Master Integration • Ensemble Recognition • Anti-Spoofing • Real-time Analytics",
                                    font=("Arial", 14))
        subtitle_label.pack(pady=5)
        
        # Mode selection buttons
        mode_frame = ctk.CTkFrame(header_frame)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        self.current_mode = ctk.StringVar(value="attendance")
        
        # Attendance button
        self.attendance_btn = ctk.CTkButton(mode_frame, text="📊 Attendance Mode", 
                                           command=lambda: self.switch_mode("attendance"),
                                           fg_color="#2E7D32", hover_color="#388E3C")
        self.attendance_btn.pack(side="left", padx=10)
        
        # Registration button  
        self.registration_btn = ctk.CTkButton(mode_frame, text="📝 Registration Mode",
                                             command=lambda: self.switch_mode("registration"),
                                             fg_color="#1565C0", hover_color="#1976D2")
        self.registration_btn.pack(side="left", padx=10)
        
        # Mode indicator
        self.mode_label = ctk.CTkLabel(mode_frame, text="Current Mode: Attendance",
                                      font=("Arial", 14, "bold"), text_color="#4CAF50")
        self.mode_label.pack(side="right", padx=10)
        
        # System status
        status_frame = ctk.CTkFrame(header_frame)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.model_status_label = ctk.CTkLabel(status_frame, text="Model: Not loaded")
        self.model_status_label.pack(side="left", padx=10)
        
        self.detector_status_label = ctk.CTkLabel(status_frame, text="Detector: Not loaded")
        self.detector_status_label.pack(side="left", padx=10)
        
        self.camera_status_label = ctk.CTkLabel(status_frame, text="Camera: Not connected")
        self.camera_status_label.pack(side="right", padx=10)
        
        self.attendance_status_label = ctk.CTkLabel(status_frame, text="Attendance: Ready")
        self.attendance_status_label.pack(side="right", padx=10)
    
    def setup_config_panel(self, parent):
        """Setup configuration panel"""
        config_frame = ctk.CTkFrame(parent)
        config_frame.pack(fill="x", padx=10, pady=10)
        
        config_label = ctk.CTkLabel(config_frame, text="⚙️ Recognition Configuration",
                                  font=("Arial", 16, "bold"))
        config_label.pack(pady=10)
        
        # Model and detector selection
        selection_frame = ctk.CTkFrame(config_frame)
        selection_frame.pack(fill="x", padx=20, pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(selection_frame)
        model_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(model_frame, text="Recognition Model:").pack(side="left", padx=10)
        self.model_var = ctk.StringVar(value=self.selected_model)
        model_combo = ctk.CTkComboBox(model_frame, variable=self.model_var,
                                      values=["ArcFace", "Facenet512", "GhostFaceNet", "VGG-Face", "SFace"],
                                      command=self.on_model_change)
        model_combo.pack(side="left", padx=10)
        
        # Detector selection
        detector_frame = ctk.CTkFrame(selection_frame)
        detector_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(detector_frame, text="Detection Backend:").pack(side="left", padx=10)
        self.detector_var = ctk.StringVar(value=self.selected_detector)
        detector_combo = ctk.CTkComboBox(detector_frame, variable=self.detector_var,
                                       values=["retinaface", "yolov8n", "yolov8m", "mtcnn", "mediapipe"],
                                       command=self.on_detector_change)
        detector_combo.pack(side="left", padx=10)
        
        # Feature toggles
        features_frame = ctk.CTkFrame(config_frame)
        features_frame.pack(fill="x", padx=20, pady=10)
        
        self.ensemble_var = ctk.BooleanVar(value=self.enable_ensemble)
        ctk.CTkCheckBox(features_frame, text="🎯 Ensemble Recognition", 
                         variable=self.ensemble_var,
                         command=self.on_ensemble_toggle).pack(side="left", padx=10)
        
        self.spoofing_var = ctk.BooleanVar(value=self.enable_anti_spoofing)
        ctk.CTkCheckBox(features_frame, text="🛡️ Anti-Spoofing", 
                         variable=self.spoofing_var,
                         command=self.on_spoofing_toggle).pack(side="left", padx=10)
        
        # Threshold control
        threshold_frame = ctk.CTkFrame(config_frame)
        threshold_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(threshold_frame, text="Confidence Threshold:").pack(side="left", padx=10)
        self.threshold_var = ctk.DoubleVar(value=self.confidence_threshold * 100)
        threshold_slider = ctk.CTkSlider(threshold_frame, variable=self.threshold_var,
                                       from_=30, to=90, number_of_steps=12,
                                       command=self.on_threshold_change)
        threshold_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.threshold_label = ctk.CTkLabel(threshold_frame, text=f"{self.confidence_threshold:.2f}")
        self.threshold_label.pack(side="left", padx=10)
    
    def setup_camera_panel(self, parent):
        """Setup camera panel with live feed"""
        camera_frame = ctk.CTkFrame(parent)
        camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Camera controls
        controls_frame = ctk.CTkFrame(camera_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.connect_button = ctk.CTkButton(controls_frame, text="📷 Connect Camera",
                                           command=self.connect_camera)
        self.connect_button.pack(side="left", padx=5)
        
        self.recognize_button = ctk.CTkButton(controls_frame, text="🔍 Recognize Face",
                                            command=self.recognize_face,
                                            state="disabled")
        self.recognize_button.pack(side="left", padx=5)
        
        self.auto_recognize_button = ctk.CTkButton(controls_frame, text="🤖 Auto Recognize",
                                                 command=self.toggle_auto_recognize)
        self.auto_recognize_button.pack(side="left", padx=5)
        
        # Video display
        self.video_label = ctk.CTkLabel(camera_frame, text="Camera not connected",
                                       font=("Arial", 16))
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Face detection info
        detection_frame = ctk.CTkFrame(camera_frame)
        detection_frame.pack(fill="x", padx=10, pady=10)
        
        self.detection_info_label = ctk.CTkLabel(detection_frame, 
                                                text="Detection: Not active",
                                                font=("Arial", 12))
        self.detection_info_label.pack(pady=5)
    
    def setup_results_panel(self, parent):
        """Setup recognition results panel"""
        results_frame = ctk.CTkFrame(parent)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Results header
        self.results_header_label = ctk.CTkLabel(results_frame, text="🎯 Recognition Results",
                                  font=("Arial", 16, "bold"))
        self.results_header_label.pack(pady=10)
        
        # Current recognition result
        current_frame = ctk.CTkFrame(results_frame)
        current_frame.pack(fill="x", padx=10, pady=10)
        
        self.current_result_label = ctk.CTkLabel(current_frame, 
                                               text="No recognition performed",
                                               font=("Arial", 14))
        self.current_result_label.pack(pady=10)
        
        # Employee info display
        self.employee_info_frame = ctk.CTkFrame(results_frame)
        self.employee_info_frame.pack(fill="x", padx=10, pady=10)
        
        self.employee_photo_label = ctk.CTkLabel(self.employee_info_frame, text="📷")
        self.employee_photo_label.pack(pady=10)
        
        self.employee_details_label = ctk.CTkLabel(self.employee_info_frame, 
                                                 text="Employee details will appear here",
                                                 font=("Arial", 12))
        self.employee_details_label.pack(pady=10)
        
        # Recognition details
        details_frame = ctk.CTkFrame(results_frame)
        details_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.recognition_details_label = ctk.CTkLabel(details_frame,
                                                     text="Recognition details will appear here",
                                                     font=("Arial", 10),
                                                     justify="left")
        self.recognition_details_label.pack(pady=10, padx=10)
        
        # Registration controls (visible only in registration mode)
        self.registration_frame = ctk.CTkFrame(results_frame)
        self.registration_frame.pack(fill="x", padx=10, pady=10)
        
        # Registration input fields
        input_frame = ctk.CTkFrame(self.registration_frame)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(input_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.reg_name_var = ctk.StringVar()
        self.reg_name_entry = ctk.CTkEntry(input_frame, textvariable=self.reg_name_var, width=200)
        self.reg_name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="ID:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.reg_id_var = ctk.StringVar()
        self.reg_id_entry = ctk.CTkEntry(input_frame, textvariable=self.reg_id_var, width=200)
        self.reg_id_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Registration buttons
        button_frame = ctk.CTkFrame(self.registration_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.reg_capture_btn = ctk.CTkButton(button_frame, text="📸 Capture Face",
                                            command=self.capture_face_for_registration,
                                            fg_color="#FF9800", hover_color="#F57C00")
        self.reg_capture_btn.pack(side="left", padx=5)
        
        self.reg_save_btn = ctk.CTkButton(button_frame, text="💾 Save Registration",
                                          command=self.save_registration,
                                          fg_color="#4CAF50", hover_color="#388E3C")
        self.reg_save_btn.pack(side="left", padx=5)
        
        # Initially hide registration frame
        self.registration_frame.pack_forget()
    
    def setup_analytics_panel(self, parent):
        """Setup analytics and history panel"""
        analytics_frame = ctk.CTkFrame(parent)
        analytics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Analytics header
        header_label = ctk.CTkLabel(analytics_frame, text="📊 Real-time Analytics",
                                  font=("Arial", 16, "bold"))
        header_label.pack(pady=10)
        
        # Statistics frame
        stats_frame = ctk.CTkFrame(analytics_frame)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        # Create statistics labels
        self.stats_labels = {}
        stat_items = [
            ("Total Recognitions", "total_recognitions"),
            ("Successful", "successful_recognitions"),
            ("Failed", "failed_recognitions"),
            ("Spoofing Attempts", "spoofing_attempts"),
            ("Avg Confidence", "average_confidence")
        ]
        
        for i, (label, key) in enumerate(stat_items):
            frame = ctk.CTkFrame(stats_frame)
            frame.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(frame, text=f"{label}:").pack(side="left", padx=10)
            self.stats_labels[key] = ctk.CTkLabel(frame, text="0")
            self.stats_labels[key].pack(side="right", padx=10)
        
        # Recognition history
        history_frame = ctk.CTkFrame(analytics_frame)
        history_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(history_frame, text="🕒 Recognition History",
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        # History listbox with scrollbar
        self.history_text = ctk.CTkTextbox(history_frame, height=200)
        self.history_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_status_panel(self, parent):
        """Setup status panel"""
        status_frame = ctk.CTkFrame(parent)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        # Status labels
        self.status_labels = {}
        status_items = [
            ("System Status", "system_status"),
            ("Last Recognition", "last_recognition"),
            ("Processing Time", "processing_time"),
            ("Models Loaded", "models_loaded")
        ]
        
        for i, (label, key) in enumerate(status_items):
            frame = ctk.CTkFrame(status_frame)
            frame.pack(side="left", fill="x", expand=True, padx=5)
            
            ctk.CTkLabel(frame, text=f"{label}:").pack(pady=5)
            self.status_labels[key] = ctk.CTkLabel(frame, text="Ready", font=("Arial", 10))
            self.status_labels[key].pack(pady=5)
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(status_frame)
        nav_frame.pack(side="right", padx=10)
        
        ctk.CTkButton(nav_frame, text="⬅️ Back to Main",
                     command=self.back_to_main).pack(side="left", padx=5)
        
        ctk.CTkButton(nav_frame, text="❌ Exit",
                     command=self.on_closing).pack(side="left", padx=5)
    
    def initialize_systems(self):
        """Initialize advanced recognition systems"""
        try:
            # Initialize recognizer
            self.recognizer = AdvancedFaceRecognizer(
                model_name=self.selected_model,
                detector_backend=self.selected_detector,
                enable_ensemble=self.enable_ensemble,
                enable_anti_spoofing=self.enable_anti_spoofing
            )
            self.recognizer.threshold = self.confidence_threshold
            
            # Initialize detector
            self.detector = AdvancedFaceDetector(
                primary_backend=self.selected_detector,
                enable_analysis=True  # Enable face analysis for gender/age/emotion/race
            )
            
            self.update_status_labels()
            self.update_stats_display()
            
            print("✅ Advanced attendance systems initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize systems: {e}")
            print(f"❌ Initialization error: {e}")
    
    def update_status_labels(self):
        """Update system status labels"""
        if self.recognizer:
            self.model_status_label.configure(text=f"Model: {self.recognizer.model_name}")
            self.detector_status_label.configure(text=f"Detector: {self.recognizer.detector_backend}")
            
            # Update system status
            ensemble_status = "ON" if self.enable_ensemble else "OFF"
            spoofing_status = "ON" if self.enable_anti_spoofing else "OFF"
            self.status_labels["system_status"].configure(
                text=f"Ready (E:{ensemble_status}, S:{spoofing_status})"
            )
            
            self.status_labels["models_loaded"].configure(
                text=f"{len(self.recognizer.ensemble_models) if self.enable_ensemble else 1} models"
            )
    
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
                    self.connect_button.configure(text="📷 Disconnect Camera")
                    self.recognize_button.configure(state="normal")
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
        self.connect_button.configure(text="📷 Connect Camera")
        self.recognize_button.configure(state="disabled")
        self.video_label.configure(text="Camera not connected")
    
    def start_video_loop(self):
        """Start video capture loop"""
        if self.cap and self.cap.isOpened():
            self.is_capturing = True
            self.update_video_frame()
    
    def update_video_frame(self):
        """Update video frame with deepface-master detector and enhanced visualization"""
        if not self.is_capturing or not self.cap or not self.detector:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Gunakan detektor deepface-master yang lebih canggih
                # Pilih backend terbaik untuk absensi: retinaface atau yolov8n
                # Set backend yang akan digunakan
                self.detector.primary_backend = "retinaface"  # Paling akurat
                faces = self.detector.detect_faces_with_attributes(frame)
                
                # Debug: Print faces data structure
                if faces:
                    print(f"🎯 UI received {len(faces)} faces with data:")
                    for i, face in enumerate(faces):
                        print(f"  Face {i+1}: confidence={face.get('confidence', 0):.3f}, has_analysis={'analysis' in face}, analysis_keys={list(face.get('analysis', {}).keys()) if 'analysis' in face else 'None'}")
                
                # Frame hasil dengan visualisasi
                result_frame = frame.copy()
                
                # Gambar border untuk setiap wajah yang terdeteksi
                for face_data in faces:
                    # Gunakan border enhanced untuk absensi
                    result_frame = draw_enhanced_face_border(
                        result_frame, 
                        face_data, 
                        user_info=getattr(self, 'current_user_info', None)
                    )
                
                # Update detection info
                current_backend = self.detector.primary_backend
                self.detection_info_label.configure(
                    text=f"DeepFace-{current_backend.upper()}: {len(faces)} face(s) detected - "
                         f"Highest confidence: {max([f['confidence'] for f in faces], default=0):.2f}"
                )
                
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
    
    def recognize_face(self):
        """Perform advanced face recognition"""
        if self.current_frame is None or self.current_frame.size == 0:
            messagebox.showwarning("No Frame", "No camera frame available")
            return
        
        try:
            start_time = time.time()
            
            # Detect faces
            if not self.detector:
                self.show_recognition_result({
                    "success": False,
                    "message": "Detector not initialized"
                })
                return
                
            faces = self.detector.detect_faces(self.current_frame)
            if not faces:
                self.show_recognition_result({
                    "success": False,
                    "message": "No face detected"
                })
                return
            
            # Use the best face (highest confidence)
            best_face = max(faces, key=lambda x: x["confidence"])
            x, y, w, h = best_face["bbox"]
            
            # Extract face ROI
            face_roi = self.current_frame[y:y+h, x:x+w]
            
            # Anti-spoofing check
            if self.enable_anti_spoofing and self.recognizer:
                is_real, spoof_confidence = self.recognizer.detect_spoofing(face_roi)
                if not is_real:
                    self.show_recognition_result({
                        "success": False,
                        "message": "Spoofing attack detected",
                        "spoofing_confidence": spoof_confidence
                    })
                    self.attendance_stats["spoofing_attempts"] += 1
                    return
            
            # Perform recognition
            if not self.recognizer:
                self.show_recognition_result({
                    "success": False,
                    "message": "Recognizer not initialized"
                })
                return
                
            result = self.recognizer.recognize_face(face_roi)
            
            # Add processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            # Jika recognition berhasil, simpan info user untuk ditampilkan di border
            if result["success"]:
                employee_id = result["employee_id"]
                
                # Simpan face image untuk attendance
                if self.current_frame is not None:
                    try:
                        # Ekstrak face ROI
                        best_face = max(faces, key=lambda x: x["confidence"])
                        x, y, w, h = best_face["bbox"]
                        face_roi = self.current_frame[y:y+h, x:x+w]
                        
                        # Simpan gambar
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        face_image_path = f"data/attendance_images/{employee_id}_{timestamp}.jpg"
                        cv2.imwrite(face_image_path, face_roi)
                        result["face_image_path"] = face_image_path
                        
                    except Exception as e:
                        print(f"⚠️  Gagal menyimpan face image: {e}")
                
                employee = self.db.get_employee(employee_id)
                if employee:
                    self.current_user_info = {
                        "user_id": employee_id,
                        "name": employee["name"],
                        "department": employee.get("department", "N/A")
                    }
                else:
                    self.current_user_info = None
            else:
                self.current_user_info = None
            
            # Update analytics
            self.update_analytics(result)
            
            # Show result
            self.show_recognition_result(result)
            
            # Add to history
            self.add_to_history(result)
            
        except Exception as e:
            self.show_recognition_result({
                "success": False,
                "message": f"Recognition error: {e}"
            })
    
    def toggle_auto_recognize(self):
        """Toggle automatic recognition"""
        # This would implement automatic recognition based on face detection
        # For now, just show a message
        messagebox.showinfo("Auto Recognize", "Auto recognition feature coming soon!")
    
    def show_recognition_result(self, result):
        """Display recognition result"""
        if result["success"]:
            employee_id = result["employee_id"]
            confidence = result["confidence"]
            processing_time = result.get("processing_time", 0)
            
            # Get employee details from database
            employee = self.db.get_employee(employee_id)
            
            if employee:
                # Record attendance dengan face image path
                face_image_path = result.get("face_image_path")
                attendance_result = self.db.record_attendance(employee_id, confidence_score=confidence, face_image_path=face_image_path)
                
                # Update UI
                result_text = f"✅ RECOGNIZED: {employee['name']}"
                details_text = f"""
Employee ID: {employee_id}
Name: {employee['name']}
Department: {employee.get('department', 'N/A')}
Confidence: {confidence:.3f}
Processing Time: {processing_time:.3f}s
Attendance: {'Recorded' if attendance_result else 'Failed'}
                """.strip()
                
                self.current_result_label.configure(text=result_text)
                self.employee_details_label.configure(text=details_text)
                
                # Update status
                self.status_labels["last_recognition"].configure(
                    text=f"{employee['name']} - {confidence:.3f}"
                )
                self.status_labels["processing_time"].configure(
                    text=f"{processing_time:.3f}s"
                )
            else:
                self.current_result_label.configure(text="❌ Employee not found in database")
                self.employee_details_label.configure(text="Employee details unavailable")
        else:
            self.current_result_label.configure(text=f"❌ {result['message']}")
            self.employee_details_label.configure(text="Recognition failed")
            
            # Update status
            self.status_labels["last_recognition"].configure(text="Failed")
    
    def update_analytics(self, result):
        """Update attendance analytics"""
        self.attendance_stats["total_recognitions"] += 1
        
        if result["success"]:
            self.attendance_stats["successful_recognitions"] += 1
            confidence = result["confidence"]
            
            # Update average confidence
            total_success = self.attendance_stats["successful_recognitions"]
            current_avg = self.attendance_stats["average_confidence"]
            self.attendance_stats["average_confidence"] = (
                (current_avg * (total_success - 1) + confidence) / total_success
            )
        else:
            self.attendance_stats["failed_recognitions"] += 1
            
            # Check if it's a spoofing attempt
            if "spoofing" in result.get("message", "").lower():
                self.attendance_stats["spoofing_attempts"] += 1
        
        self.update_stats_display()
    
    def update_stats_display(self):
        """Update statistics display"""
        stats = self.attendance_stats
        
        self.stats_labels["total_recognitions"].configure(text=str(stats["total_recognitions"]))
        self.stats_labels["successful_recognitions"].configure(text=str(stats["successful_recognitions"]))
        self.stats_labels["failed_recognitions"].configure(text=str(stats["failed_recognitions"]))
        self.stats_labels["spoofing_attempts"].configure(text=str(stats["spoofing_attempts"]))
        self.stats_labels["average_confidence"].configure(text=f"{stats['average_confidence']:.3f}")
    
    def add_to_history(self, result):
        """Add recognition result to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if result["success"]:
            history_entry = f"[{timestamp}] ✅ {result['employee_id']} - {result['confidence']:.3f}"
        else:
            history_entry = f"[{timestamp}] ❌ {result['message']}"
        
        self.recognition_history.append(history_entry)
        
        # Keep only recent history
        if len(self.recognition_history) > self.max_history:
            self.recognition_history.pop(0)
        
        # Update history display
        self.update_history_display()
    
    def update_history_display(self):
        """Update history text display"""
        self.history_text.delete("1.0", tk.END)
        for entry in reversed(self.recognition_history):
            self.history_text.insert(tk.END, entry + "\n")
    
    def on_model_change(self, model_name):
        """Handle model change"""
        self.selected_model = model_name
        self.initialize_systems()
    
    def on_detector_change(self, detector_name):
        """Handle detector change"""
        self.selected_detector = detector_name
        self.initialize_systems()
    
    def on_ensemble_toggle(self):
        """Handle ensemble toggle"""
        self.enable_ensemble = self.ensemble_var.get()
        self.initialize_systems()
    
    def on_spoofing_toggle(self):
        """Handle spoofing toggle"""
        self.enable_anti_spoofing = self.spoofing_var.get()
        self.initialize_systems()
    
    def on_threshold_change(self, value):
        """Handle threshold change"""
        self.confidence_threshold = float(value) / 100.0
        self.threshold_label.configure(text=f"{self.confidence_threshold:.2f}")
        
        if self.recognizer:
            self.recognizer.threshold = self.confidence_threshold
    
    def switch_mode(self, mode):
        """Switch between attendance and registration mode"""
        if not self.root:
            return
            
        self.current_mode.set(mode)
        
        if mode == "attendance":
            # Attendance mode
            self.mode_label.configure(text="Current Mode: Attendance", text_color="#4CAF50")
            self.attendance_btn.configure(fg_color="#2E7D32", hover_color="#388E3C")
            self.registration_btn.configure(fg_color="#1565C0", hover_color="#1976D2")
            
            # Update title and subtitle
            self.root.title("📊 Advanced Attendance System - DeepFace Master")
            self.results_header_label.configure(text="🎯 Recognition Results")
            self.current_result_label.configure(text="No recognition performed")
            
            # Hide registration frame, show attendance elements
            self.registration_frame.pack_forget()
            
        else:  # registration mode
            # Registration mode
            self.mode_label.configure(text="Current Mode: Registration", text_color="#FF9800")
            self.registration_btn.configure(fg_color="#FF9800", hover_color="#F57C00")
            self.attendance_btn.configure(fg_color="#2E7D32", hover_color="#388E3C")
            
            # Update title for registration
            self.root.title("📝 Advanced Registration System - DeepFace Master")
            self.results_header_label.configure(text="📝 Registration Form")
            self.current_result_label.configure(text="Ready to register new employee")
            
            # Show registration frame
            self.registration_frame.pack(fill="x", padx=10, pady=10, before=self.employee_info_frame)
        
        print(f"🔄 Switched to {mode} mode")
    
    def capture_face_for_registration(self):
        """Capture current face for registration"""
        if self.current_frame is None or self.current_frame.size == 0:
            messagebox.showwarning("No Frame", "No camera frame available")
            return
        
        if not self.detector:
            messagebox.showerror("Detector Error", "Face detector not initialized")
            return
        
        try:
            # Get face detection
            faces = self.detector.detect_faces_with_attributes(self.current_frame)
            
            if not faces:
                messagebox.showwarning("No Face", "No face detected in current frame")
                return
            
            # Use the first face (most confident)
            face_data = max(faces, key=lambda x: x.get('confidence', 0))
            
            # Debug: Print face data structure
            print(f"🎯 Face data structure: {face_data}")
            
            if face_data.get('confidence', 0) < self.confidence_threshold:
                messagebox.showwarning("Low Confidence", f"Face confidence too low: {face_data['confidence']:.2f}")
                return
            
            # Extract face region
            bbox = face_data['bbox']
            x1, y1, x2, y2 = bbox
            
            # Add some padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(self.current_frame.shape[1], x2 + padding)
            y2 = min(self.current_frame.shape[0], y2 + padding)
            
            # Crop face region
            face_crop = self.current_frame[y1:y2, x1:x2]
            
            # Store for registration
            self.registration_face = face_crop
            self.registration_face_data = face_data
            
            # Update UI with detailed analysis
            analysis = face_data.get('analysis', {})
            if not analysis:
                # Fallback to individual keys if analysis dict is not available
                analysis = {
                    "age": face_data.get("age", "N/A"),
                    "gender": face_data.get("gender", "N/A"),
                    "emotion": face_data.get("emotion", "N/A"),
                    "race": face_data.get("race", "N/A")
                }
            
            age = analysis.get('age', 'N/A')
            gender = analysis.get('gender', 'N/A')
            emotion = analysis.get('emotion', 'N/A')
            race = analysis.get('race', 'N/A')
            
            self.current_result_label.configure(text="✅ Face captured for registration")
            self.recognition_details_label.configure(
                text=f"Face captured successfully!\n\n"
                     f"📊 Face Analysis:\n"
                     f"• Confidence: {face_data['confidence']:.3f}\n"
                     f"• Age: {age}\n"
                     f"• Gender: {gender}\n"
                     f"• Emotion: {emotion}\n"
                     f"• Race: {race}"
            )
            
            # Show captured face in photo label
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_pil = face_pil.resize((150, 150), Image.Resampling.LANCZOS)
            face_photo = ImageTk.PhotoImage(face_pil)
            
            self.employee_photo_label.configure(image=face_photo)
            self.employee_photo_label.image = face_photo  # Keep reference
            
            messagebox.showinfo("Success", "Face captured successfully!\nEnter name and ID, then click Save Registration.")
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Failed to capture face: {e}")
            print(f"❌ Capture error: {e}")
    
    def save_registration(self):
        """Save registration data"""
        if not hasattr(self, 'registration_face') or self.registration_face is None:
            messagebox.showwarning("No Face", "Please capture a face first")
            return
        
        name = self.reg_name_var.get().strip()
        employee_id = self.reg_id_var.get().strip()
        
        if not name or not employee_id:
            messagebox.showwarning("Missing Info", "Please enter both name and employee ID")
            return
        
        if not self.recognizer:
            messagebox.showerror("Recognizer Error", "Face recognizer not initialized")
            return
        
        try:
            # Generate embedding for registration
            embedding = self.recognizer.extract_embedding(self.registration_face)
            
            if embedding is None:
                messagebox.showerror("Embedding Error", "Failed to generate face embedding")
                return
            
            # Create registration data
            registration_data = {
                "name": name,
                "employee_id": employee_id,
                "embedding": embedding,
                "face_data": self.registration_face_data,
                "registration_date": datetime.now().isoformat(),
                "model_used": self.selected_model,
                "detector_used": self.selected_detector
            }
            
            # Save to database/file
            self.save_registration_to_db(registration_data)
            
            # Update UI
            self.current_result_label.configure(text=f"✅ Registration saved: {name}")
            self.employee_details_label.configure(
                text=f"Employee Registered:\n"
                     f"- Name: {name}\n"
                     f"- ID: {employee_id}\n"
                     f"- Model: {self.selected_model}\n"
                     f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Clear form
            self.reg_name_var.set("")
            self.reg_id_var.set("")
            
            messagebox.showinfo("Success", f"Employee '{name}' registered successfully!")
            
        except Exception as e:
            messagebox.showerror("Registration Error", f"Failed to save registration: {e}")
            print(f"❌ Registration error: {e}")
    
    def save_registration_to_db(self, registration_data):
        """Save registration data to database"""
        # For now, save to JSON file
        import json
        import os
        import numpy as np
        
        db_path = "data/employee_database.json"
        os.makedirs("data", exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            "name": registration_data["name"],
            "employee_id": registration_data["employee_id"],
            "embedding": registration_data["embedding"].tolist() if isinstance(registration_data["embedding"], np.ndarray) else registration_data["embedding"],
            "registration_date": registration_data["registration_date"],
            "model_used": registration_data["model_used"],
            "detector_used": registration_data["detector_used"],
            "face_data": {
                "confidence": float(registration_data["face_data"]["confidence"]),
                "bbox": [int(x) for x in registration_data["face_data"]["bbox"]],
                "analysis": {
                    "age": int(registration_data["face_data"].get("analysis", {}).get("age", 0)),
                    "gender": registration_data["face_data"].get("analysis", {}).get("gender", "Unknown"),
                    "emotion": registration_data["face_data"].get("analysis", {}).get("emotion", "Unknown"),
                    "race": registration_data["face_data"].get("analysis", {}).get("race", "Unknown")
                }
            }
        }
        
        # Load existing database or create new
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                database = json.load(f)
        else:
            database = {}
        
        # Add new employee
        employee_id = registration_data["employee_id"]
        database[employee_id] = serializable_data
        
        # Save database
        with open(db_path, 'w') as f:
            json.dump(database, f, indent=2)
        
        print(f"✅ Employee '{registration_data['name']}' saved to database")
    
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
def run_advanced_attendance():
    """Run the advanced attendance system"""
    app = AdvancedAttendanceUI()
    app.run()

if __name__ == "__main__":
    run_advanced_attendance()
