import cv2
import numpy as np
from pathlib import Path
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from deepface import DeepFace
from config.settings import FACE_RECOGNITION_THRESHOLD, FACE_RECOGNITION_MODEL, MODELS_PATH
from config.advanced_config import (
    RECOGNITION_MODELS, DETECTION_BACKENDS, DISTANCE_METRICS,
    ANTI_SPOOFING_CONFIG, ENSEMBLE_CONFIG, FALLBACK_CONFIG
)
from models.database import DatabaseManager
from config.constants import FACE_QUALITY_REQUIREMENTS

class AdvancedFaceRecognizer:
    """
    Enhanced Face Recognizer using DeepFace master with multi-model support
    """
    def __init__(self, 
                 model_name: str = "ArcFace",
                 detector_backend: str = "retinaface",
                 distance_metric: str = "cosine",
                 enable_anti_spoofing: bool = True,
                 enable_ensemble: bool = True):
        
        self.threshold = FACE_RECOGNITION_THRESHOLD
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.enable_anti_spoofing = enable_anti_spoofing
        self.enable_ensemble = enable_ensemble
        self.db = DatabaseManager()
        self.last_error = None
        
        # Setup model paths
        try:
            Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
            os.environ["DEEPFACE_HOME"] = str(MODELS_PATH)
        except Exception:
            pass
        
        # Build models on initialization
        self._build_models()
        
        # Available models for fallback
        self.available_models = [
            "ArcFace", "Facenet512", "VGG-Face", "GhostFaceNet", 
            "SFace", "OpenFace", "DeepFace", "DeepID", "Dlib"
        ]
        
        # Available detectors
        self.available_detectors = [
            "retinaface", "yolov8n", "yolov8m", "yolov8l", 
            "mtcnn", "mediapipe", "dlib", "opencv", "ssd"
        ]
        
        # Ensemble models if enabled
        if self.enable_ensemble:
            self.ensemble_models = ENSEMBLE_CONFIG["models"]
            self.ensemble_weights = ENSEMBLE_CONFIG["weights"]
        else:
            self.ensemble_models = [model_name]
            self.ensemble_weights = [1.0]
        
        print(f"✅ AdvancedFaceRecognizer initialized with {model_name} + {detector_backend}")
        if enable_ensemble:
            print(f"   Ensemble models: {self.ensemble_models}")
        if enable_anti_spoofing:
            print(f"   Anti-spoofing: Enabled")
    
    def _build_models(self):
        """Build and cache models for faster inference"""
        try:
            # Build primary model
            self.model = DeepFace.build_model(model_name=self.model_name)
            print(f"✅ Built recognition model: {self.model_name}")
            
            # Build detector if needed (only for backends that need model building)
            # Note: retinaface, yolov8n, yolov11n are detector backends, not recognition models
            valid_detector_backends = ["retinaface", "yolov8n", "yolov8m", "yolov8l", "yolov11n", "yolov11s", "yolov11m", "yolov11l"]
            if self.detector_backend in valid_detector_backends:
                # These backends don't need separate model building - they're handled by detector
                self.detector_model = None
                print(f"ℹ️  Detector backend '{self.detector_backend}' will be handled by AdvancedFaceDetector")
            elif self.detector_backend not in ["opencv", "ssd"]:
                # Try to build as recognition model only if it's not a known detector backend
                try:
                    self.detector_model = DeepFace.build_model(model_name=self.detector_backend)
                    print(f"✅ Built detection model: {self.detector_backend}")
                except Exception as e:
                    print(f"⚠️  Could not build detector model '{self.detector_backend}': {e}")
                    self.detector_model = None
            else:
                self.detector_model = None
                
            # Build anti-spoofing model if enabled
            if self.enable_anti_spoofing:
                try:
                    self.spoofing_model = DeepFace.build_model(model_name="FasNet")
                    print("✅ Built anti-spoofing model: FasNet")
                except Exception as e:
                    print(f"⚠️  Could not build anti-spoofing model: {e}")
                    self.spoofing_model = None
            
        except Exception as e:
            self.last_error = f"Model building failed: {str(e)}"
            print(f"❌ {self.last_error}")
            self.model = None
            self.detector_model = None
            self.spoofing_model = None
    
    def extract_embedding(self, face_image: np.ndarray, 
                         model_name: Optional[str] = None,
                         enforce_detection: bool = False) -> Optional[List[float]]:
        """
        Extract face embedding with fallback models
        """
        if model_name is None:
            model_name = self.model_name
        
        models_to_try = [model_name] + [m for m in self.available_models if m != model_name]
        
        for current_model in models_to_try:
            try:
                # Convert numpy array to proper format for DeepFace
                if isinstance(face_image, np.ndarray):
                    # Ensure image is in RGB format
                    if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    else:
                        img = face_image
                else:
                    img = face_image
                
                # Extract embedding using DeepFace
                result = DeepFace.represent(
                    img_path=img,
                    model_name=current_model,
                    detector_backend=self.detector_backend,
                    enforce_detection=enforce_detection,
                    align=True
                )
                
                if result and isinstance(result, list) and len(result) > 0:
                    result_item = result[0]
                    if isinstance(result_item, dict) and "embedding" in result_item:
                        embedding = result_item["embedding"]
                        if current_model != model_name:
                            print(f"ℹ️  Used fallback model: {current_model}")
                        return embedding
                    
            except Exception as e:
                self.last_error = f"{current_model} embedding extraction failed: {str(e)}"
                print(f"⚠️  {self.last_error}")
                continue
        
        print("❌ All embedding extraction attempts failed")
        return None
    
    def extract_ensemble_embedding(self, face_image: np.ndarray) -> Optional[List[float]]:
        """
        Extract ensemble embedding using multiple models
        """
        if not self.enable_ensemble:
            return self.extract_embedding(face_image)
        
        embeddings = []
        weights = []
        
        for model_name, weight in zip(self.ensemble_models, self.ensemble_weights):
            try:
                embedding = self.extract_embedding(face_image, model_name)
                if embedding:
                    embeddings.append(embedding)
                    weights.append(weight)
            except Exception as e:
                print(f"⚠️  Ensemble model {model_name} failed: {e}")
                continue
        
        if not embeddings:
            return None
        
        # Get target dimension (maximum dimension from all embeddings)
        target_dim = max(len(emb) for emb in embeddings)
        
        # Pad all embeddings to target dimension
        padded_embeddings = []
        for embedding in embeddings:
            emb_array = np.array(embedding)
            if len(emb_array) < target_dim:
                # Pad with zeros
                padded_emb = np.pad(emb_array, (0, target_dim - len(emb_array)), mode='constant')
            else:
                padded_emb = emb_array
            padded_embeddings.append(padded_emb)
        
        # Weighted average of embeddings
        if len(padded_embeddings) == 1:
            return padded_embeddings[0].tolist()
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        ensemble_embedding = np.zeros(target_dim)
        for embedding, weight in zip(padded_embeddings, normalized_weights):
            ensemble_embedding += embedding * weight
        
        return ensemble_embedding.tolist()
    
    def recognize_face(self, face_image: np.ndarray, 
                      confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Recognize face with anti-spoofing and advanced validation
        """
        if confidence_threshold is None:
            confidence_threshold = self.threshold
        
        try:
            # Anti-spoofing check
            if self.enable_anti_spoofing:
                is_real, spoofing_confidence = self.detect_spoofing(face_image)
                if not is_real:
                    return {
                        "success": False,
                        "message": "Spoofing detected",
                        "spoofing_confidence": spoofing_confidence,
                        "is_real": False
                    }
            
            # Extract ensemble embedding
            embedding = self.extract_ensemble_embedding(face_image)
            if embedding is None:
                return {
                    "success": False,
                    "message": "Failed to extract face embedding"
                }
            
            # Get all registered faces from database
            registered_faces = self.db.get_all_face_embeddings()
            if not registered_faces:
                return {
                    "success": False,
                    "message": "No registered faces found"
                }
            
            # Find best match
            best_match = None
            best_similarity = 0.0
            
            for registered_face in registered_faces:
                try:
                    registered_embedding = json.loads(registered_face['embedding'])
                    similarity = self.calculate_similarity(embedding, registered_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = registered_face
                        
                except Exception as e:
                    print(f"⚠️  Error processing registered face {registered_face.get('employee_id', 'unknown')}: {e}")
                    continue
            
            # Check if match is good enough
            if best_similarity >= confidence_threshold and best_match:
                employee = self.db.get_employee(best_match['employee_id'])
                
                result = {
                    "success": True,
                    "message": "Face recognized successfully",
                    "employee_id": best_match['employee_id'],
                    "employee_name": employee['name'] if employee else "Unknown",
                    "confidence": best_similarity,
                    "model_used": self.model_name,
                    "detector_used": self.detector_backend,
                    "is_real": True if not self.enable_anti_spoofing else is_real
                }
                
                if self.enable_anti_spoofing:
                    result["spoofing_confidence"] = spoofing_confidence
                
                return result
            else:
                return {
                    "success": False,
                    "message": "No matching face found",
                    "best_confidence": best_similarity,
                    "threshold": confidence_threshold,
                    "is_real": True if not self.enable_anti_spoofing else is_real
                }
                
        except Exception as e:
            self.last_error = f"Face recognition failed: {str(e)}"
            print(f"❌ {self.last_error}")
            return {
                "success": False,
                "message": "Face recognition error",
                "error": str(e)
            }
    
    def detect_spoofing(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect face spoofing using anti-spoofing models
        """
        if not self.enable_anti_spoofing:
            return True, 1.0
        
        try:
            # Use DeepFace for anti-spoofing detection
            result = DeepFace.spoofing_detection(
                img_path=face_image,
                model_name="FasNet",
                enforce_detection=False
            )
            
            if result and isinstance(result, list) and len(result) > 0:
                # FasNet returns spoofing probability
                result_item = result[0]
                if isinstance(result_item, dict):
                    spoofing_confidence = result_item.get("spoofing_probability", 0.0)
                    is_real = spoofing_confidence < 0.5  # Threshold for real face
                    
                    return is_real, 1.0 - spoofing_confidence
            
            return True, 1.0  # Default to real if detection fails
            
        except Exception as e:
            print(f"⚠️  Anti-spoofing detection failed: {e}")
            return True, 1.0  # Default to real if detection fails
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate similarity between two face embeddings
        """
        try:
            if self.distance_metric == "cosine":
                # Cosine similarity
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                
                # Normalize vectors
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                
                # Calculate cosine similarity
                similarity = np.dot(vec1_norm, vec2_norm)
                return float(similarity)
            
            elif self.distance_metric == "euclidean":
                # Euclidean distance converted to similarity
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                
                distance = np.linalg.norm(vec1 - vec2)
                # Convert distance to similarity (inverse relationship)
                similarity = 1.0 / (1.0 + distance)
                return float(similarity)
            
            elif self.distance_metric == "euclidean_l2":
                # L2 normalized Euclidean distance
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                
                # Normalize vectors
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(vec1_norm - vec2_norm)
                # Convert to similarity
                similarity = 1.0 - (distance / 2.0)  # Normalize to 0-1
                return max(0.0, min(1.0, float(similarity)))
            
            else:
                # Default to cosine similarity
                return self.calculate_similarity(embedding1, embedding2)
                
        except Exception as e:
            print(f"⚠️  Similarity calculation failed: {e}")
            return 0.0
    
    def validate_face_quality(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate face quality for recognition
        """
        try:
            if face_image is None or face_image.size == 0:
                return False, "No face image provided"
            
            h, w = face_image.shape[:2]
            
            # Check minimum size (pixel area)
            min_face_size = FACE_QUALITY_REQUIREMENTS.get("min_face_size", 8000)
            face_area = h * w
            if face_area < min_face_size:
                return False, f"Face too small: {face_area}px, minimum: {min_face_size}px"
            
            # Convert to grayscale for quality checks
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Blur detection
            min_blur_score = FACE_QUALITY_REQUIREMENTS.get("min_blur_score", 60)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < min_blur_score:
                return False, f"Face too blurry: {blur_score:.1f}, minimum: {min_blur_score}"
            
            # Brightness check
            brightness = np.mean(gray)
            min_brightness = FACE_QUALITY_REQUIREMENTS.get("min_brightness", 40)
            max_brightness = FACE_QUALITY_REQUIREMENTS.get("max_brightness", 210)
            
            if brightness < min_brightness:
                return False, f"Face too dark: {brightness:.1f}, minimum: {min_brightness}"
            if brightness > max_brightness:
                return False, f"Face too bright: {brightness:.1f}, maximum: {max_brightness}"
            
            return True, "Face quality acceptable"
            
        except Exception as e:
            return False, f"Quality validation error: {str(e)}"
    
    def get_face_quality_score(self, face_image: np.ndarray) -> Tuple[float, List[str]]:
        """
        Calculate face quality score (0-100)
        """
        try:
            if face_image is None or face_image.size == 0:
                return 0.0, ["no_face"]
            
            h, w = face_image.shape[:2]
            issues = []
            scores = []
            
            # Size score (min_face_size is in pixels area)
            min_face_area = FACE_QUALITY_REQUIREMENTS.get("min_face_size", 8000)
            face_area = h * w
            area_ratio = face_area / min_face_area
            size_score = min(100.0, area_ratio * 50)
            scores.append(size_score)
            
            if area_ratio < 1.0:
                issues.append("too_small")
            
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Blur score
            min_blur_score = FACE_QUALITY_REQUIREMENTS.get("min_blur_score", 60)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality = min(100.0, (blur_score / min_blur_score) * 100)
            scores.append(blur_quality)
            
            if blur_score < min_blur_score:
                issues.append("blurry")
            
            # Brightness score
            brightness = np.mean(gray)
            min_brightness = FACE_QUALITY_REQUIREMENTS.get("min_brightness", 40)
            max_brightness = FACE_QUALITY_REQUIREMENTS.get("max_brightness", 210)
            
            # Optimal brightness around 127
            brightness_quality = 100.0 - (abs(brightness - 127) / 127) * 100
            scores.append(brightness_quality)
            
            if brightness < min_brightness or brightness > max_brightness:
                issues.append("poor_lighting")
            
            # Contrast score
            contrast = np.std(gray)
            min_contrast = FACE_QUALITY_REQUIREMENTS.get("min_contrast", 20)
            contrast_quality = min(100.0, (contrast / min_contrast) * 100)
            scores.append(contrast_quality)
            
            if contrast < min_contrast:
                issues.append("low_contrast")
            
            # Overall quality score
            overall_score = float(np.mean(scores))
            
            return overall_score, issues
            
        except Exception as e:
            print(f"⚠️  Quality score calculation failed: {e}")
            return 0.0, ["calculation_error"]
    
    def register_face_advanced(self, employee_id: str, face_samples: List[np.ndarray], 
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Advanced face registration with multiple samples and metadata
        """
        try:
            if not face_samples:
                return {"success": False, "message": "No face samples provided"}
            
            embeddings = []
            quality_scores = []
            
            # Process each sample
            for i, face_sample in enumerate(face_samples):
                try:
                    # Validate quality
                    is_valid, quality_msg = self.validate_face_quality(face_sample)
                    quality_score, quality_issues = self.get_face_quality_score(face_sample)
                    
                    if not is_valid:
                        print(f"⚠️  Sample {i+1} rejected: {quality_msg}")
                        continue
                    
                    # Extract ensemble embedding
                    embedding = self.extract_ensemble_embedding(face_sample)
                    if embedding:
                        embeddings.append(embedding)
                        quality_scores.append(quality_score or 0.0)
                        
                except Exception as e:
                    print(f"⚠️  Error processing sample {i+1}: {e}")
                    continue
            
            if not embeddings:
                return {"success": False, "message": "No valid embeddings extracted"}
            
            # Calculate average embedding
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            # Store in database
            success = self.db.add_face_embedding(
                employee_id=employee_id,
                embedding=avg_embedding
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Advanced registration successful with {len(embeddings)} samples",
                    "embeddings_stored": len(embeddings),
                    "avg_quality": avg_quality,
                    "model_used": self.model_name
                }
            else:
                return {"success": False, "message": "Database storage failed"}
                
        except Exception as e:
            return {"success": False, "message": f"Advanced registration error: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration"""
        return {
            "recognition_model": self.model_name,
            "detector_backend": self.detector_backend,
            "distance_metric": self.distance_metric,
            "threshold": self.threshold,
            "anti_spoofing": self.enable_anti_spoofing,
            "ensemble_enabled": self.enable_ensemble,
            "ensemble_models": self.ensemble_models,
            "ensemble_weights": self.ensemble_weights,
            "available_models": self.available_models,
            "available_detectors": self.available_detectors,
            "model_info": RECOGNITION_MODELS.get(self.model_name, {}),
            "detector_info": DETECTION_BACKENDS.get(self.detector_backend, {}),
            "last_error": self.last_error
        }