"""
Advanced Face Recognition Module using DeepFace Master

This module provides enhanced face recognition capabilities with:
- Multi-model support (ArcFace, FaceNet, GhostFaceNet, etc.)
- Multi-backend detection (RetinaFace, YOLOv8, MTCNN, MediaPipe)
- Anti-spoofing detection
- Advanced quality validation
- Vector database support
- Real-time streaming
- Ensemble recognition
- Fallback mechanisms

Usage:
    from face_recognition import AdvancedFaceRecognizer, AdvancedFaceDetector, AdvancedFaceTrainer
    
    # Initialize components
    recognizer = AdvancedFaceRecognizer(
        model_name="ArcFace",
        detector_backend="retinaface",
        enable_anti_spoofing=True
    )
    
    detector = AdvancedFaceDetector(
        primary_backend="retinaface",
        confidence_threshold=0.7
    )
    
    trainer = AdvancedFaceTrainer(
        recognition_model="ArcFace",
        detection_backend="retinaface"
    )
"""

from .advanced_recognizer import AdvancedFaceRecognizer
from .advanced_detector import AdvancedFaceDetector
from .advanced_trainer import AdvancedFaceTrainer

__all__ = [
    "AdvancedFaceRecognizer",
    "AdvancedFaceDetector", 
    "AdvancedFaceTrainer"
]

__version__ = "2.0.0"
__author__ = "Face Attendance System"
__description__ = "Advanced Face Recognition Module using DeepFace Master"