#!/usr/bin/env python3
"""
Simple test for DeepFace master integration
"""

try:
    # Test basic DeepFace import
    from deepface import DeepFace
    print("✅ DeepFace imported successfully")
    
    # Test DeepFace modules
    from deepface.modules import modeling, representation, verification
    print("✅ DeepFace modules imported successfully")
    
    # Test our custom modules
    from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
    print("✅ AdvancedFaceRecognizer imported successfully")
    
    # Test model building
    print("Testing model building...")
    recognizer = AdvancedFaceRecognizer()
    print("✅ AdvancedFaceRecognizer created successfully")
    
    print("\n✅ All tests passed! DeepFace master integration is working.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()