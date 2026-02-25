#!/usr/bin/env python3
# Smoke tests for attendance and registration flows without GUI
import sys
import os
import time
from datetime import datetime

import cv2
import numpy as np

# Ensure project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from face_recognition.advanced_detector import AdvancedFaceDetector
from face_recognition.advanced_recognizer import AdvancedFaceRecognizer
from face_recognition.advanced_trainer import AdvancedFaceTrainer
from models.database import DatabaseManager


def open_camera_with_fallback():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    indices = [CAMERA_INDEX, 0, 1, 2]
    tried = []
    for be in backends:
        for idx in indices:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, be)
                tried.append(f"{idx}:{be}")
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    ret, _ = cap.read()
                    if ret:
                        return cap, tried
                if cap:
                    cap.release()
            except Exception:
                if cap:
                    cap.release()
                continue
    return None, tried


def get_one_frame():
    cap, tried = open_camera_with_fallback()
    if cap and cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame, tried
    return None, tried


def test_attendance_flow():
    print("🧪 Testing Attendance Flow (smoke)")
    db = DatabaseManager()
    detector = AdvancedFaceDetector(primary_backend="retinaface", enable_analysis=True)
    recognizer = AdvancedFaceRecognizer(model_name="ArcFace", detector_backend="retinaface",
                                        enable_anti_spoofing=True, enable_ensemble=True)

    frame, tried = get_one_frame()
    if frame is None:
        print(f"⚠️  Tidak bisa mendapatkan frame kamera. Tried: {', '.join(tried)}")
        print("ℹ️  Lewati deteksi langsung; memastikan fungsi fail-safe berjalan.")
        # Use blank image to ensure no exceptions
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    faces = detector.detect_faces_with_attributes(frame)
    print(f"🔎 Faces detected: {len(faces)}")

    if faces:
        best = max(faces, key=lambda x: x.get("confidence", 0))
        x1, y1, x2, y2 = best["bbox"]
        face_roi = frame[y1:y2, x1:x2]
        result = recognizer.recognize_face(face_roi)
        print(f"🎯 Recognition result: {result}")

        if result.get("success"):
            employee_id = result["employee_id"]
            saved = db.record_attendance(employee_id, confidence_score=result.get("confidence"))
            print(f"🗃️  Attendance recorded: {saved}")
    else:
        print("ℹ️  Tidak ada wajah terdeteksi; jalur attendance aman tanpa exception.")

    print("✅ Attendance smoke test selesai")
    return True


def test_registration_flow():
    print("\n🧪 Testing Registration Flow (smoke)")
    trainer = AdvancedFaceTrainer(recognition_model="ArcFace", detection_backend="retinaface")

    employee_id = f"SMOKETEST_{int(time.time())}"
    employee_name = "Smoke Test User"
    trainer.start_registration(employee_id, employee_name, {"source": "smoke_test"})

    collected = 0
    attempts = 0
    max_attempts = 10

    while collected < 3 and attempts < max_attempts:
        frame, _ = get_one_frame()
        if frame is None:
            # Use blank frame to ensure function returns gracefully
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        res = trainer.capture_frame(frame, auto_quality_check=False)
        print(f"📷 Capture attempt {attempts+1}: {res.get('message', res)}")
        if res.get("success"):
            collected += 1
        attempts += 1

    # Intentionally call complete_registration early to test validation paths
    complete = trainer.complete_registration()
    print(f"🧾 Complete registration result: {complete}")
    assert isinstance(complete, dict)
    print("✅ Registration smoke test selesai")
    return True


if __name__ == "__main__":
    ok1 = test_attendance_flow()
    ok2 = test_registration_flow()
    print("\n📊 Smoke Test Summary")
    print(f"  Attendance flow: {'✅' if ok1 else '❌'}")
    print(f"  Registration flow: {'✅' if ok2 else '❌'}")

