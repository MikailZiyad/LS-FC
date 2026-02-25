import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATABASE_PATH = BASE_DIR / "data" / "face_attendance.db"

FACE_IMAGES_PATH = BASE_DIR / "data" / "employees"
ATTENDANCE_DATA_PATH = BASE_DIR / "data" / "attendance"
MODELS_PATH = BASE_DIR / "data" / "models"

FACE_CONFIDENCE_THRESHOLD = 0.5
FACE_RECOGNITION_THRESHOLD = 0.25
REQUIRED_FACE_SAMPLES = 15

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FACE_DETECTION_MODEL = "yolov8"
FACE_RECOGNITION_MODEL = "Facenet"

LOG_FILE = BASE_DIR / "logs" / "attendance.log"

REPORTS_PATH = BASE_DIR / "reports"
