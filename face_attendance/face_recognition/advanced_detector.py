import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from deepface import DeepFace
from deepface.modules import detection
from config.settings import FACE_CONFIDENCE_THRESHOLD
from .advanced_analyzer import AdvancedFaceAnalyzer
from .model_preloader import preloader, initialize_models
from utils.enhanced_visualization import draw_enhanced_face_border, draw_registration_border

class AdvancedFaceDetector:
    """
    Advanced Face Detector with multi-backend support using DeepFace master
    """
    def __init__(self, 
                 primary_backend: str = "retinaface",
                 fallback_backends: Optional[List[str]] = None,
                 confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD,
                 enable_analysis: bool = True):
        
        self.primary_backend = primary_backend
        self.confidence_threshold = confidence_threshold
        self.last_error = None
        self.enable_analysis = enable_analysis
        self.analyzer = AdvancedFaceAnalyzer() if enable_analysis else None
        
        # Initialize model preloader untuk menghindari download - hanya jika perlu
        try:
            from .model_preloader import check_existing_models
            existing_models = check_existing_models()
            print(f"📋 Found {len(existing_models)} existing models")
            
            # Hanya jalankan preloader jika model belum lengkap
            required_models = {'retinaface', 'arcface', 'facenet512', 'age', 'gender', 'emotion', 'race'}
            missing_models = required_models - existing_models
            
            if missing_models:
                print(f"🔄 Menginisialisasi model preloader untuk model yang kurang: {missing_models}")
                initialize_models()
                print("✅ Model preloader berhasil diinisialisasi")
            else:
                print("✅ Semua model yang dibutuhkan sudah tersedia, skip preloader")
                
        except Exception as e:
            print(f"⚠️  Warning: Gagal menginisialisasi preloader: {e}")
            # Coba jalankan preloader sebagai fallback
            try:
                print("🔄 Mencoba preloader sebagai fallback...")
                initialize_models()
            except Exception as e2:
                print(f"⚠️  Fallback preloader juga gagal: {e2}")
        
        # Available backends in order of preference - dari deepface-master
        self.available_backends = [
            "retinaface",    # Most accurate - dari deepface-master
            "yolov8n",       # YOLOv8 face detection - dari deepface-master
            "yolov8m",       # Medium YOLOv8 - dari deepface-master
            "yolov8l",       # Large YOLOv8 - dari deepface-master
            "yolov11n",      # YOLOv11 terbaru - dari deepface-master
            "yolov11s",      # Small YOLOv11 - dari deepface-master
            "yolov11m",      # Medium YOLOv11 - dari deepface-master
            "yolov11l",      # Large YOLOv11 - dari deepface-master
            "yolov12n",      # YOLOv12 paling baru - dari deepface-master
            "yolov12s",      # Small YOLOv12 - dari deepface-master
            "mtcnn",         # Good balance - dari deepface-master
            "mediapipe",     # Fast - dari deepface-master
            "dlib",          # Traditional but reliable - dari deepface-master
            "opencv",        # Fastest but less accurate - dari deepface-master
            "ssd",           # SSD detector - dari deepface-master
            "yunet",         # YuNet detector - dari deepface-master
            "centerface",    # CenterFace - dari deepface-master
        ]
        
        # Set fallback backends
        if fallback_backends is None:
            self.fallback_backends = [b for b in self.available_backends if b != primary_backend]
        else:
            self.fallback_backends = fallback_backends
        
        print(f"✅ AdvancedFaceDetector initialized with backend: {primary_backend}")
        if enable_analysis:
            print("✅ Face analysis enabled")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame using primary backend with fallback support
        """
        try:
            faces = self._detect_faces_single_backend(frame, self.primary_backend, self.confidence_threshold)
            
            if faces:
                return faces
            
            # Try fallback backends
            for backend in self.fallback_backends:
                try:
                    faces = self._detect_faces_single_backend(frame, backend, self.confidence_threshold)
                    if faces:
                        print(f"ℹ️  Used fallback backend: {backend}")
                        return faces
                except Exception as e:
                    self.last_error = f"{backend} detection failed: {str(e)}"
                    print(f"⚠️  {self.last_error}")
                    continue
            
            print("❌ All detection backends failed")
            return []
            
        except Exception as e:
            self.last_error = f"Face detection failed: {str(e)}"
            print(f"❌ {self.last_error}")
            return []
    
    def _detect_faces_single_backend(self, frame: np.ndarray, 
                                       backend: str,
                                       confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Deteksi wajah menggunakan single backend dengan error handling yang lebih baik
        """
        try:
            # Gunakan preloaded detector jika tersedia
            detector = preloader.get_detector(backend)
            if detector is None:
                # Fallback ke DeepFace.extract_faces jika belum preload
                print(f"⚠️  Detector {backend} belum di-preload, menggunakan DeepFace...")
                return self._detect_faces_deepface_fallback(frame, backend, confidence_threshold)
        except Exception as e:
            print(f"⚠️  Error dengan detector langsung {backend}: {e}")
            return self._detect_faces_deepface_fallback(frame, backend, confidence_threshold)
        
        try:
            # Gunakan detector langsung
            results = []
            
            # Konversi frame ke format yang tepat
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB format
                img = frame
            else:
                # Convert BGR to RGB if needed
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Deteksi wajah menggunakan detector langsung
            try:
                detections = detector.detect_faces(img)
            except Exception as det_error:
                print(f"⚠️  Error in detect_faces for {backend}: {det_error}")
                # Try alternative approach
                try:
                    # Use DeepFace.extract_faces as fallback
                    faces = DeepFace.extract_faces(
                        img_path=img,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    results = []
                    for face_data in faces:
                        if face_data.get('confidence', 0) >= confidence_threshold:
                            facial_area = face_data.get('facial_area', {})
                            x, y, w, h = facial_area.get('x', 0), facial_area.get('y', 0), facial_area.get('w', 100), facial_area.get('h', 100)
                            bbox = (x, y, x + w, y + h)
                            
                            # Crop face region
                            face_crop = img[y:y+h, x:x+w]
                            
                            result = {
                                "bbox": bbox,
                                "confidence": face_data.get('confidence', 0.9),
                                "face": face_crop,
                                "backend": backend,
                                "facial_area": facial_area
                            }
                            
                            # Tambahkan analisis wajah jika analyzer aktif
                            if self.enable_analysis and self.analyzer:
                                try:
                                    analysis = self.analyzer.analyze_face(face_crop)
                                    result.update(analysis)
                                except Exception as e:
                                    print(f"⚠️  Analisis wajah gagal: {e}")
                            
                            results.append(result)
                    
                    return results
                    
                except Exception as extract_error:
                    print(f"⚠️  DeepFace.extract_faces also failed: {extract_error}")
                    return []
            
            if detections:
                for detection in detections:
                    try:
                        # Handle berbagai format detection
                        if hasattr(detection, 'confidence') and detection.confidence >= confidence_threshold:
                            # Ekstrak informasi dari detection
                            facial_area = detection.facial_area if hasattr(detection, 'facial_area') else detection
                            
                            # Konversi ke format bbox
                            if hasattr(facial_area, 'x') and hasattr(facial_area, 'y'):
                                x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h
                                bbox = (x, y, x + w, y + h)
                            else:
                                # Fallback ke format tuple/list
                                bbox = tuple(facial_area) if hasattr(facial_area, '__iter__') else (0, 0, 100, 100)
                            
                            # Buat hasil deteksi
                            result = {
                                "bbox": bbox,
                                "confidence": float(detection.confidence),
                                "backend": backend
                            }
                            
                            # Tambahkan facial area jika tersedia
                            if hasattr(facial_area, 'x'):
                                result["facial_area"] = {
                                    "x": int(facial_area.x),
                                    "y": int(facial_area.y),
                                    "w": int(facial_area.w),
                                    "h": int(facial_area.h)
                                }
                            
                            results.append(result)
                        elif isinstance(detection, dict) and detection.get('confidence', 0) >= confidence_threshold:
                            # Handle format dictionary
                            result = {
                                "bbox": detection.get('bbox', (0, 0, 100, 100)),
                                "confidence": float(detection.get('confidence', 0)),
                                "backend": backend
                            }
                            results.append(result)
                    except Exception as det_error:
                        print(f"⚠️  Error processing individual detection: {det_error}")
                        continue
                
                return results
            else:
                return []
                
        except Exception as e:
            print(f"⚠️  Error dengan detector {backend}: {e}")
            return self._detect_faces_deepface_fallback(frame, backend, confidence_threshold)
            print(f"⚠️  Error dengan detector langsung {backend}: {e}")
            # Fallback ke DeepFace.extract_faces
            return self._detect_faces_deepface_fallback(frame, backend, confidence_threshold)
    
    def _detect_faces_deepface_fallback(self, frame: np.ndarray, 
                                   backend: str,
                                   confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Fallback method menggunakan DeepFace.extract_faces
        """
        try:
            # Gunakan DeepFace.extract_faces sebagai fallback
            faces_data = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=backend,
                enforce_detection=False,
                align=True,
                expand_percentage=0
            )
            
            results = []
            
            if isinstance(faces_data, list):
                for face_data in faces_data:
                    if isinstance(face_data, dict) and "confidence" in face_data and face_data["confidence"] >= confidence_threshold:
                        # Ekstrak bounding box dari facial area
                        facial_area = face_data.get("facial_area")
                        
                        if facial_area:
                            # Konversi ke format bbox standar (x1, y1, x2, y2)
                            if isinstance(facial_area, dict):
                                x = facial_area.get("x", 0)
                                y = facial_area.get("y", 0)
                                w = facial_area.get("w", 0)
                                h = facial_area.get("h", 0)
                                bbox = (x, y, x + w, y + h)
                            elif isinstance(facial_area, (list, tuple)) and len(facial_area) == 4:
                                bbox = tuple(facial_area)
                            else:
                                bbox = facial_area
                        
                        # Siapkan hasil deteksi
                        result = {
                            "bbox": bbox,
                            "confidence": face_data.get("confidence", 0.0),
                            "face": face_data.get("face"),  # Gambar wajah yang sudah di-crop
                            "backend": backend,
                            "facial_area": facial_area
                        }
                        
                        # Tambahkan landmarks jika tersedia
                        if isinstance(face_data, dict) and "landmarks" in face_data:
                            result["landmarks"] = face_data["landmarks"]
                        
                        # Tambahkan analisis wajah jika analyzer aktif
                        if self.enable_analysis and self.analyzer:
                            try:
                                face_crop = face_data.get("face")
                                if face_crop is not None:
                                    analysis = self.analyzer.analyze_face(face_crop)
                                    result.update(analysis)
                            except Exception as e:
                                print(f"⚠️  Analisis wajah gagal: {e}")
                        
                        results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ DeepFace fallback juga gagal untuk {backend}: {e}")
            return []
    
    def extract_face_for_analysis(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                  margin: int = 20, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Extract face yang optimal untuk analisis dengan preprocessing
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Add margin for better face context
            if margin > 0:
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)
            
            # Extract face region
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                return None
            
            # Preprocessing untuk analisis yang lebih baik
            # 1. Resize dengan interpolasi berkualitas tinggi
            face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_CUBIC)
            
            # 2. Konversi ke RGB jika perlu (DeepFace butuh RGB)
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                # Asumsikan frame dalam BGR (OpenCV default), konversi ke RGB
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_resized
            
            # 3. Enhancement: contrast adjustment
            # Convert to LAB color space for better contrast adjustment
            lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back and convert to RGB
            enhanced_lab = cv2.merge([l, a, b])
            face_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # 4. Normalisasi ke range 0-1 untuk model deep learning
            face_normalized = face_enhanced.astype(np.float32) / 255.0
            
            # 5. Konversi kembali ke uint8 untuk DeepFace
            face_final = (face_normalized * 255).astype(np.uint8)
            
            return face_final
            
        except Exception as e:
            self.last_error = f"Face extraction for analysis failed: {str(e)}"
            print(f"❌ {self.last_error}")
            return None
    
    def detect_faces_with_attributes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces with additional attributes like landmarks, pose, etc.
        """
        print(f"🔍 detect_faces_with_attributes called - enable_analysis: {self.enable_analysis}, analyzer: {self.analyzer}")
        
        faces = self.detect_faces(frame)
        print(f"📊 Found {len(faces)} faces")
        
        # Add additional attributes if analyzer is available
        if self.enable_analysis and self.analyzer:
            print("🧠 Starting face attribute analysis...")
            for i, face in enumerate(faces):
                try:
                    # Extract face yang lebih berkualitas untuk analisis
                    face_crop = self.extract_face_for_analysis(frame, face["bbox"])
                    print(f"  Face {i+1}: face_crop quality check - shape: {face_crop.shape if face_crop is not None else 'None'}, min: {face_crop.min() if face_crop is not None else 'N/A'}, max: {face_crop.max() if face_crop is not None else 'N/A'}")
                    
                    if face_crop is not None:
                        # Analyze face attributes
                        print(f"  Face {i+1}: Starting analysis with enhanced face crop...")
                        attributes = self.analyzer.analyze_face(face_crop)
                        print(f"  Face {i+1}: Analysis result: {attributes}")
                        
                        # Store analysis in dedicated 'analysis' key
                        if attributes:
                            face["analysis"] = {
                                "age": attributes.get("age", 0),
                                "gender": attributes.get("gender", "unknown"),
                                "emotion": attributes.get("emotion", "neutral"),
                                "race": attributes.get("race", "unknown"),
                                "gender_confidence": attributes.get("gender_confidence", 0.0),
                                "emotion_confidence": attributes.get("emotion_confidence", 0.0),
                                "race_confidence": attributes.get("race_confidence", 0.0)
                            }
                            
                            # Also add emotion separately for compatibility
                            if "emotion" in attributes:
                                face["emotion"] = attributes["emotion"]
                        
                        print(f"✅ Face {i+1} analysis completed: {face.get('analysis', {})}")
                    else:
                        print(f"⚠️  Face {i+1}: Failed to extract quality face crop")
                        
                except Exception as e:
                    print(f"❌ Face {i+1} analysis failed: {e}")
                    face["analysis"] = {}  # Empty analysis on failure
        else:
            print("⚠️  Face analysis skipped - analyzer not available")
        
        return faces
    
    def get_available_backends(self) -> List[str]:
        """
        Get list of available detection backends
        """
        return self.available_backends.copy()
    
    def get_current_backend(self) -> str:
        """
        Get current active backend
        """
        return self.primary_backend
    
    def get_last_error(self) -> Optional[str]:
        """
        Get last error message
        """
        return self.last_error