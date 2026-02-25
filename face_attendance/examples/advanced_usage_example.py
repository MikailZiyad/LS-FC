#!/usr/bin/env python3
"""
Advanced Face Recognition Usage Example
Demonstrates the new capabilities with DeepFace Master integration
"""

import cv2
import numpy as np
from pathlib import Path
import time
from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
from face_recognition.advanced_detector import AdvancedFaceDetector
from face_recognition.advanced_trainer import AdvancedFaceTrainer
from config.advanced_config import ADVANCED_CONFIG

def demo_advanced_recognition():
    """Demonstrate advanced face recognition capabilities"""
    print("🚀 Advanced Face Recognition Demo")
    print("=" * 50)
    
    # Initialize recognizer with ensemble and anti-spoofing
    recognizer = AdvancedFaceRecognizer(
        model_name="ArcFace",
        detector_backend="retinaface",
        enable_ensemble=True,
        enable_anti_spoofing=True
    )
    
    print(f"✅ Recognizer initialized: {recognizer.model_name}")
    print(f"   Detector: {recognizer.detector_backend}")
    print(f"   Ensemble: {recognizer.enable_ensemble}")
    print(f"   Anti-spoofing: {recognizer.enable_anti_spoofing}")
    
    # Get model info
    model_info = recognizer.get_model_info()
    print(f"   Available models: {len(model_info['available_models'])}")
    print(f"   Available detectors: {len(model_info['available_detectors'])}")
    
    return recognizer

def demo_advanced_detection():
    """Demonstrate advanced face detection with fallback"""
    print("\n🔍 Advanced Face Detection Demo")
    print("=" * 50)
    
    detector = AdvancedFaceDetector(
        primary_backend="retinaface",
        fallback_backends=["yolov8n", "mtcnn", "mediapipe"]
    )
    
    print(f"✅ Detector initialized: {detector.primary_backend}")
    print(f"   Fallback backends: {detector.fallback_backends}")
    
    return detector

def demo_advanced_training():
    """Demonstrate advanced training capabilities"""
    print("\n📚 Advanced Face Training Demo")
    print("=" * 50)
    
    trainer = AdvancedFaceTrainer(
        recognition_model="ArcFace",
        detection_backend="retinaface",
        enable_multi_pose=True,
        enable_quality_validation=True
    )
    
    print(f"✅ Trainer initialized: {trainer.recognizer.model_name}")
    print(f"   Multi-pose: {trainer.enable_multi_pose}")
    print(f"   Quality validation: {trainer.enable_quality_validation}")
    
    return trainer

def demo_face_quality_validation():
    """Demonstrate face quality validation"""
    print("\n✨ Face Quality Validation Demo")
    print("=" * 50)
    
    recognizer = AdvancedFaceRecognizer()
    
    # Create sample test images
    test_cases = [
        ("Too small", np.zeros((50, 50, 3), dtype=np.uint8)),
        ("Too dark", np.zeros((100, 100, 3), dtype=np.uint8) + 20),
        ("Too bright", np.zeros((100, 100, 3), dtype=np.uint8) + 230),
        ("Good quality", np.zeros((100, 100, 3), dtype=np.uint8) + 100),
    ]
    
    for name, image in test_cases:
        is_valid, message = recognizer.validate_face_quality(image)
        score, issues = recognizer.get_face_quality_score(image)
        print(f"   {name}: {'✅' if is_valid else '❌'} {message} (score: {score:.1f})")

def demo_ensemble_recognition():
    """Demonstrate ensemble recognition"""
    print("\n🎯 Ensemble Recognition Demo")
    print("=" * 50)
    
    # Initialize with ensemble
    recognizer = AdvancedFaceRecognizer(
        enable_ensemble=True,
        ensemble_models=["ArcFace", "Facenet512", "GhostFaceNet"],
        ensemble_weights=[0.5, 0.3, 0.2]
    )
    
    print(f"✅ Ensemble recognizer initialized")
    print(f"   Models: {recognizer.ensemble_models}")
    print(f"   Weights: {recognizer.ensemble_weights}")
    
    return recognizer

def demo_anti_spoofing():
    """Demonstrate anti-spoofing detection"""
    print("\n🛡️ Anti-Spoofing Demo")
    print("=" * 50)
    
    recognizer = AdvancedFaceRecognizer(enable_anti_spoofing=True)
    
    # Create sample test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8) + 128
    
    is_real, confidence = recognizer.detect_spoofing(test_image)
    print(f"   Anti-spoofing result: {'✅ Real face' if is_real else '❌ Spoof detected'}")
    print(f"   Confidence: {confidence:.3f}")
    
    return recognizer

def demo_configuration_options():
    """Demonstrate configuration options"""
    print("\n⚙️ Configuration Options Demo")
    print("=" * 50)
    
    print("Available Recognition Models:")
    from config.advanced_config import RECOGNITION_MODELS
    for name, info in RECOGNITION_MODELS.items():
        if info.get("recommended"):
            print(f"   ⭐ {name}: {info['description']} (Accuracy: {info['accuracy']}%)")
    
    print("\nAvailable Detection Backends:")
    from config.advanced_config import DETECTION_BACKENDS
    for name, info in DETECTION_BACKENDS.items():
        if info.get("recommended"):
            print(f"   ⭐ {name}: {info['description']}")
    
    print("\nDistance Metrics:")
    from config.advanced_config import DISTANCE_METRICS
    for metric, info in DISTANCE_METRICS.items():
        print(f"   📏 {metric}: {info['description']}")

def demo_real_time_processing():
    """Demonstrate real-time processing capabilities"""
    print("\n🎥 Real-time Processing Demo")
    print("=" * 50)
    
    recognizer = AdvancedFaceRecognizer(
        model_name="ArcFace",
        detector_backend="yolov8n",  # Fast detection for real-time
        enable_ensemble=False,  # Disable for speed
        enable_anti_spoofing=False  # Disable for speed
    )
    
    print(f"✅ Real-time configuration:")
    print(f"   Model: {recognizer.model_name} (optimized for speed)")
    print(f"   Detector: {recognizer.detector_backend} (fast)")
    print(f"   Ensemble: Disabled (for speed)")
    print(f"   Anti-spoofing: Disabled (for speed)")
    
    return recognizer

def main():
    """Main demo function"""
    print("🚀 DeepFace Master Advanced Features Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        recognizer = demo_advanced_recognition()
        detector = demo_advanced_detection()
        trainer = demo_advanced_training()
        
        demo_face_quality_validation()
        ensemble_recognizer = demo_ensemble_recognition()
        anti_spoofing_recognizer = demo_anti_spoofing()
        
        demo_configuration_options()
        real_time_recognizer = demo_real_time_processing()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Usage Tips:")
        print("   • Use ensemble recognition for higher accuracy")
        print("   • Use anti-spoofing for security applications")
        print("   • Use quality validation for better registration")
        print("   • Use fallback backends for robust detection")
        print("   • Optimize configuration for speed vs accuracy")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()