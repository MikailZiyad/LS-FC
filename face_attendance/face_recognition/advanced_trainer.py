import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from face_recognition.advanced_detector import AdvancedFaceDetector
from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
from models.database import DatabaseManager
from config.constants import FACE_POSES, SAMPLES_PER_POSE
from config.settings import FACE_IMAGES_PATH, FACE_CONFIDENCE_THRESHOLD

class AdvancedFaceTrainer:
    """
    Advanced Face Trainer using DeepFace master with multi-pose support
    """
    def __init__(self, 
                 recognition_model: str = "ArcFace",
                 detection_backend: str = "retinaface",
                 enable_anti_spoofing: bool = True):
        
        self.detector = AdvancedFaceDetector(
            primary_backend=detection_backend,
            confidence_threshold=FACE_CONFIDENCE_THRESHOLD
        )
        self.recognizer = AdvancedFaceRecognizer(
            model_name=recognition_model,
            detector_backend=detection_backend,
            enable_anti_spoofing=enable_anti_spoofing
        )
        self.db = DatabaseManager()
        self.face_samples = []
        self.current_pose_index = 0
        self.samples_collected = 0
        self.employee_id = None
        self.employee_name = None
        self.training_metadata = {}
        
        # Quality tracking
        self.quality_scores = []
        self.rejected_samples = []
        
        print(f"✅ AdvancedFaceTrainer initialized with {recognition_model} + {detection_backend}")
    
    def start_registration(self, employee_id: str, employee_name: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start advanced face registration process
        """
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.face_samples = []
        self.current_pose_index = 0
        self.samples_collected = 0
        self.quality_scores = []
        self.rejected_samples = []
        self.training_metadata = metadata or {}
        
        # Create employee folder
        employee_folder = Path(FACE_IMAGES_PATH) / employee_id
        employee_folder.mkdir(exist_ok=True)
        
        # Initialize training metadata
        self.training_metadata.update({
            "employee_id": employee_id,
            "employee_name": employee_name,
            "start_time": datetime.now().isoformat(),
            "model_used": self.recognizer.model_name,
            "detector_used": self.detector.primary_backend,
            "anti_spoofing": self.recognizer.enable_anti_spoofing
        })
        
        result = {
            "success": True,
            "message": f"📝 Starting advanced registration for {employee_name} ({employee_id})",
            "target_samples": len(FACE_POSES) * SAMPLES_PER_POSE,
            "poses": len(FACE_POSES),
            "samples_per_pose": SAMPLES_PER_POSE,
            "model": self.recognizer.model_name,
            "detector": self.detector.primary_backend
        }
        
        print(result["message"])
        return result
    
    def capture_frame(self, frame: np.ndarray, 
                     auto_quality_check: bool = True) -> Dict[str, Any]:
        """
        Capture frame with advanced quality checks and multi-face handling
        """
        if self.current_pose_index >= len(FACE_POSES):
            return {
                "success": False,
                "message": "Registration complete",
                "progress": 100.0,
                "samples_collected": self.samples_collected
            }
        
        # Detect faces with advanced detection
        faces = self.detector.detect_faces_with_attributes(frame)
        
        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected",
                "current_pose": FACE_POSES[self.current_pose_index],
                "progress": self.get_progress(),
                "samples_collected": self.samples_collected
            }
        
        if len(faces) > 1:
            return {
                "success": False,
                "message": f"Multiple faces detected ({len(faces)})",
                "faces_detected": len(faces),
                "current_pose": FACE_POSES[self.current_pose_index],
                "progress": self.get_progress(),
                "samples_collected": self.samples_collected
            }
        
        # Get the detected face
        face_data = faces[0]
        face_image = face_data["face"]
        
        # Anti-spoofing check if enabled
        if self.recognizer.enable_anti_spoofing:
            is_real = face_data.get("is_real", True)
            if not is_real:
                return {
                    "success": False,
                    "message": "Spoofing detected - face is not real",
                    "spoofing_confidence": face_data.get("spoofing_confidence", 0.0),
                    "current_pose": FACE_POSES[self.current_pose_index],
                    "progress": self.get_progress(),
                    "samples_collected": self.samples_collected
                }
        
        # Quality validation
        if auto_quality_check:
            is_valid, quality_message = self.recognizer.validate_face_quality(face_image)
            quality_score, _ = self.recognizer.get_face_quality_score(face_image)
            
            if not is_valid:
                self.rejected_samples.append({
                    "reason": quality_message,
                    "quality_score": quality_score,
                    "pose": FACE_POSES[self.current_pose_index]
                })
                
                return {
                    "success": False,
                    "message": f"Poor face quality: {quality_message}",
                    "quality_score": quality_score,
                    "quality_issues": face_data.get("quality", {}).get("issues", []),
                    "current_pose": FACE_POSES[self.current_pose_index],
                    "progress": self.get_progress(),
                    "samples_collected": self.samples_collected
                }
            
            # Store quality score
            self.quality_scores.append(quality_score)
        
        # Add to samples
        self.face_samples.append(face_image.copy())
        self.samples_collected += 1
        
        # Save sample with metadata
        pose_name = FACE_POSES[self.current_pose_index].replace(" ", "_").lower()
        quality_score, _ = self.recognizer.get_face_quality_score(face_image)
        
        if FACE_IMAGES_PATH and self.employee_id:
            sample_filename = f"{pose_name}_sample_{self.samples_collected % SAMPLES_PER_POSE + 1}_q{int(quality_score)}.jpg"
            sample_path = Path(FACE_IMAGES_PATH) / self.employee_id / sample_filename
            cv2.imwrite(str(sample_path), face_image)
            
            # Store metadata
            sample_metadata = {
                "pose": FACE_POSES[self.current_pose_index],
                "quality_score": quality_score,
                "detector": self.detector.primary_backend,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(sample_path)
            }
            
            if self.training_metadata.get("samples_metadata") is None:
                self.training_metadata["samples_metadata"] = []
            self.training_metadata["samples_metadata"].append(sample_metadata)
        
        # Check if we have enough samples for current pose
        if self.samples_collected % SAMPLES_PER_POSE == 0:
            self.current_pose_index += 1
        
        current_pose = FACE_POSES[min(self.current_pose_index, len(FACE_POSES) - 1)]
        progress = self.get_progress()
        avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0
        
        status_msg = f"Pose: {current_pose} | Progress: {progress:.1f}% | Samples: {self.samples_collected} | Avg Quality: {avg_quality:.1f}"
        
        return {
            "success": True,
            "message": status_msg,
            "current_pose": current_pose,
            "progress": progress,
            "samples_collected": self.samples_collected,
            "quality_score": quality_score,
            "avg_quality": avg_quality,
            "rejected_samples": len(self.rejected_samples)
        }
    
    def is_registration_complete(self) -> bool:
        """Check if registration is complete"""
        return self.samples_collected >= len(FACE_POSES) * SAMPLES_PER_POSE
    
    def get_current_instruction(self) -> str:
        """Get current pose instruction"""
        if self.current_pose_index < len(FACE_POSES):
            return FACE_POSES[self.current_pose_index]
        return "Registration complete"
    
    def get_progress(self) -> float:
        """Get registration progress"""
        total_samples = len(FACE_POSES) * SAMPLES_PER_POSE
        return (self.samples_collected / total_samples) * 100 if total_samples > 0 else 0
    
    def complete_registration(self) -> Dict[str, Any]:
        """
        Complete advanced face registration with comprehensive validation
        """
        if not self.is_registration_complete():
            return {
                "success": False,
                "message": "Not enough samples collected",
                "required_samples": len(FACE_POSES) * SAMPLES_PER_POSE,
                "collected_samples": self.samples_collected
            }
        
        if len(self.face_samples) < 10:
            return {
                "success": False,
                "message": "Too few valid samples",
                "valid_samples": len(self.face_samples),
                "minimum_required": 10
            }
        
        try:
            # Validate employee_id
            if not self.employee_id:
                return {
                    "success": False,
                    "message": "Employee ID is required",
                    "error_type": "validation"
                }
            
            # Register employee in database
            success, message = self.db.add_employee(
                self.employee_id, 
                self.employee_name
            )
            
            if not success:
                return {
                    "success": False,
                    "message": f"Database error: {message}",
                    "error_type": "database"
                }
            
            # Advanced face registration with metadata
            registration_result = self.recognizer.register_face_advanced(
                employee_id=self.employee_id,
                face_samples=self.face_samples,
                metadata=self.training_metadata
            )
            
            if registration_result["success"]:
                # Finalize training metadata
                self.training_metadata.update({
                    "end_time": datetime.now().isoformat(),
                    "total_samples": len(self.face_samples),
                    "avg_quality_score": np.mean(self.quality_scores) if self.quality_scores else 0,
                    "rejected_samples_count": len(self.rejected_samples),
                    "registration_success": True
                })
                
                success_message = f"✅ Advanced registration successful for {self.employee_name}"
                print(success_message)
                
                return {
                    "success": True,
                    "message": success_message,
                    "embeddings_stored": registration_result["embeddings_stored"],
                    "total_samples": len(self.face_samples),
                    "avg_quality": np.mean(self.quality_scores) if self.quality_scores else 0,
                    "rejected_samples": len(self.rejected_samples),
                    "model_used": self.recognizer.model_name,
                    "detector_used": self.detector.primary_backend,
                    "metadata": self.training_metadata
                }
            else:
                return {
                    "success": False,
                    "message": f"Face training failed: {registration_result.get('message', 'Unknown error')}",
                    "error_type": "training",
                    "details": registration_result
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Registration error: {str(e)}",
                "error_type": "exception",
                "error_details": str(e)
            }
    
    def reset_registration(self) -> Dict[str, Any]:
        """Reset registration process"""
        self.face_samples = []
        self.current_pose_index = 0
        self.samples_collected = 0
        self.quality_scores = []
        self.rejected_samples = []
        
        reset_message = "🔄 Advanced registration reset"
        print(reset_message)
        
        return {
            "success": True,
            "message": reset_message,
            "cleared_samples": len(self.face_samples),
            "cleared_rejections": len(self.rejected_samples)
        }
    
    def get_registration_stats(self) -> Dict[str, Any]:
        """Get detailed registration statistics"""
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "current_pose": self.get_current_instruction(),
            "pose_index": self.current_pose_index,
            "samples_collected": self.samples_collected,
            "total_target": len(FACE_POSES) * SAMPLES_PER_POSE,
            "progress_percentage": self.get_progress(),
            "quality_scores": self.quality_scores,
            "avg_quality": np.mean(self.quality_scores) if self.quality_scores else 0,
            "rejected_samples": self.rejected_samples,
            "rejection_count": len(self.rejected_samples),
            "is_complete": self.is_registration_complete(),
            "model_config": self.recognizer.get_model_info(),
            "detector_config": self.detector.get_detector_info()
        }
    
    def draw_advanced_registration_ui(self, frame: np.ndarray, status_msg: str = "") -> np.ndarray:
        """
        Draw advanced registration UI with quality indicators
        """
        # Get current face detection
        faces = self.detector.detect_faces_with_attributes(frame)
        h, w = frame.shape[:2]
        
        # Draw detection ROI
        roi_w = int(w * 0.6)
        roi_h = int(h * 0.7)
        roi_x1 = (w - roi_w) // 2
        roi_y1 = (h - roi_h) // 2
        roi_x2 = roi_x1 + roi_w
        roi_y2 = roi_y1 + roi_h
        
        # ROI rectangle with quality-based color
        roi_color = (255, 255, 0)  # Default yellow
        if len(faces) == 1:
            face_quality = faces[0].get("quality", {}).get("score", 0)
            if face_quality > 80:
                roi_color = (0, 255, 0)  # Green for good quality
            elif face_quality > 60:
                roi_color = (0, 255, 255)  # Cyan for medium quality
            else:
                roi_color = (0, 0, 255)  # Red for poor quality
        
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 3)
        
        # Draw detected faces with detailed info
        for i, face_data in enumerate(faces):
            bbox = face_data["bbox"]
            x1, y1, x2, y2 = bbox
            conf = face_data["confidence"]
            
            # Check if face is within ROI
            inside_roi = (x1 >= roi_x1 and y1 >= roi_y1 and 
                         x2 <= roi_x2 and y2 <= roi_y2)
            
            # Face rectangle color based on quality and position
            if inside_roi and conf >= FACE_CONFIDENCE_THRESHOLD:
                face_color = (0, 255, 0)  # Green for valid
            else:
                face_color = (0, 0, 255)  # Red for invalid
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), face_color, 2)
            
            # Face info text
            face_info = f"Face {i+1}: {conf:.2f}"
            
            # Add quality info
            quality_data = face_data.get("quality", {})
            if quality_data:
                quality_score = quality_data.get("score", 0)
                issues = quality_data.get("issues", [])
                
                face_info += f" | Q:{quality_score}"
                if issues:
                    face_info += f" Issues:{','.join(issues[:2])}"
            
            # Add anti-spoofing info
            if self.recognizer.enable_anti_spoofing:
                is_real = face_data.get("is_real", True)
                spoof_conf = face_data.get("spoofing_confidence", 0.0)
                if not is_real:
                    face_info += f" | SPOOF! {spoof_conf:.2f}"
            
            cv2.putText(frame, face_info, (x1, max(20, y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, face_color, 1)
        
        # Advanced instruction panel
        if self.current_pose_index < len(FACE_POSES):
            instruction = self.get_current_instruction()
            progress = self.get_progress()
            avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0
            
            # Background for text
            cv2.rectangle(frame, (10, 10), (450, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (450, 150), (255, 255, 255), 2)
            
            # Instruction text
            cv2.putText(frame, f"📸 Pose: {instruction}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Progress
            cv2.putText(frame, f"📊 Progress: {progress:.1f}%", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Samples and quality
            cv2.putText(frame, f"🎯 Samples: {self.samples_collected}/{len(FACE_POSES) * SAMPLES_PER_POSE}", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"⭐ Avg Quality: {avg_quality:.1f}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Rejections
            if self.rejected_samples:
                cv2.putText(frame, f"❌ Rejected: {len(self.rejected_samples)}", 
                           (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Status message
            if status_msg:
                cv2.putText(frame, f"Status: {status_msg}", (20, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Model info overlay
        model_info = f"Model: {self.recognizer.model_name} | Detector: {self.detector.primary_backend}"
        cv2.putText(frame, model_info, (w - 300, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame