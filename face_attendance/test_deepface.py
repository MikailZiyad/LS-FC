#!/usr/bin/env python3
"""
Test script for DeepFace master integration
"""

def test_imports():
    """Test if all advanced modules can be imported"""
    try:
        from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
        print("✅ AdvancedFaceRecognizer imported successfully")
        
        from face_recognition.advanced_trainer import AdvancedFaceTrainer
        print("✅ AdvancedFaceTrainer imported successfully")
        
        from face_recognition.advanced_detector import AdvancedFaceDetector
        print("✅ AdvancedFaceDetector imported successfully")
        
        from ui.advanced_attendance_ui import AdvancedAttendanceUI
        print("✅ AdvancedAttendanceUI imported successfully")
        
        from ui.advanced_registration_ui import AdvancedRegistrationUI
        print("✅ AdvancedRegistrationUI imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without camera"""
    try:
        # Test recognizer creation
        from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
        recognizer = AdvancedFaceRecognizer()
        print("✅ AdvancedFaceRecognizer created successfully")
        
        # Test trainer creation
        from face_recognition.advanced_trainer import AdvancedFaceTrainer
        trainer = AdvancedFaceTrainer()
        print("✅ AdvancedFaceTrainer created successfully")
        
        # Test detector creation
        from face_recognition.advanced_detector import AdvancedFaceDetector
        detector = AdvancedFaceDetector()
        print("✅ AdvancedFaceDetector created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing DeepFace Master Integration...")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing imports...")
    import_success = test_imports()
    
    # Test basic functionality
    print("\n2. Testing basic functionality...")
    func_success = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if import_success and func_success:
        print("✅ All tests passed! DeepFace master integration is working.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\nTest completed.")