"""
Model Preloader untuk menghindari download model dari internet
Menggunakan model lokal yang sudah ada di deepface-master
"""

import os
import sys
from typing import Dict, Any, Optional, List
import warnings

# Tambahkan path deepface-master ke sys.path
deepface_master_path = r"c:\Users\Asus\Documents\Kodingan\yolo\deepface-master\deepface-master"
if deepface_master_path not in sys.path:
    sys.path.insert(0, deepface_master_path)

# Set environment variable untuk custom weights path
os.environ["DEEPFACE_HOME"] = r"c:\Users\Asus\Documents\Kodingan\yolo\face_attendance\data\models"

# Override download function untuk menggunakan model lokal
from deepface.commons import weight_utils
import gdown

# Simpan fungsi original
original_download = weight_utils.download_weights_if_necessary

def mock_download_weights_if_necessary(
    file_name: str, 
    source_url: str, 
    compress_type: Optional[str] = None
) -> str:
    """
    Mock download function yang menggunakan model lokal jika tersedia
    """
    home = r"c:\Users\Asus\Documents\Kodingan\yolo\face_attendance\data\models"
    target_file = os.path.normpath(os.path.join(home, ".deepface", "weights", file_name))
    
    # Cek apakah file sudah ada
    if os.path.isfile(target_file):
        print(f"✅ Model {file_name} sudah tersedia di {target_file}")
        return target_file
    
    # Cek apakah ada file dengan nama mirip di folder weights
    weights_dir = os.path.dirname(target_file)
    if os.path.exists(weights_dir):
        for existing_file in os.listdir(weights_dir):
            # Cek apakah nama file mengandung nama model yang dicari
            model_name = file_name.replace('_weights.h5', '').replace('-face.pt', '').lower()
            existing_model = existing_file.replace('_weights.h5', '').replace('-face.pt', '').lower()
            
            if model_name in existing_model or existing_model in model_name:
                print(f"✅ Menggunakan model existing: {existing_file} untuk {file_name}")
                return os.path.join(weights_dir, existing_file)
    
    # Jika tidak ada sama sekali, skip download dan beri warning
    print(f"⚠️  Model {file_name} tidak ditemukan secara lokal, melewati download...")
    # Return path dummy untuk menghindari error
    return target_file

# Override fungsi
weight_utils.download_weights_if_necessary = mock_download_weights_if_necessary

class ModelPreloader:
    """
    Preload semua model yang dibutuhkan untuk menghindari delay saat pertama kali digunakan
    """
    
    def __init__(self):
        self.preloaded_models = {}
        self.detector_models = {}
        self.recognition_models = {}
        self.analysis_models = {}
    
    def preload_detection_models(self, backends: Optional[list] = None):
        """
        Preload model deteksi wajah - hanya jika file model sudah ada
        """
        if backends is None:
            backends = ["retinaface", "yolov8n", "yolov11n"]  # Hanya backend utama
        
        print("🔄 Preloading detection models...")
        
        for backend in backends:
            try:
                # Skip jika file model belum ada (untuk YOLO)
                if backend in ["yolov8n", "yolov8m", "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov12n", "yolov12s"]:
                    from deepface.commons import weight_utils
                    
                    # Nama file model untuk YOLO
                    model_file = f"{backend}-face.pt"
                    home = r"c:\Users\Asus\Documents\Kodingan\yolo\face_attendance\data\models"
                    target_file = os.path.normpath(os.path.join(home, ".deepface", "weights", model_file))
                    
                    # Skip jika file belum ada atau masih partial download
                    if not os.path.isfile(target_file) or target_file.endswith('.part'):
                        print(f"⏭️  Skip {backend} - model file belum tersedia")
                        continue
                
                # Load detector model
                from deepface.modules import modeling
                detector = modeling.build_model("face_detector", backend)
                self.detector_models[backend] = detector
                print(f"✅ Preloaded detector: {backend}")
                
            except Exception as e:
                print(f"⚠️  Gagal preload detector {backend}: {e}")
    
    def preload_recognition_models(self, models: Optional[list] = None):
        """
        Preload model pengenalan wajah
        """
        if models is None:
            models = ["ArcFace", "Facenet512"]
        
        print("🔄 Preloading recognition models...")
        
        for model_name in models:
            try:
                from deepface.modules import modeling
                
                # Load recognition model
                model = modeling.build_model("facial_recognition", model_name)
                self.recognition_models[model_name] = model
                print(f"✅ Preloaded recognition model: {model_name}")
                
            except Exception as e:
                print(f"⚠️  Gagal preload recognition model {model_name}: {e}")
    
    def preload_analysis_models(self, analysis_types: Optional[list] = None):
        """
        Preload model analisis (gender, age, emotion, race)
        """
        if analysis_types is None:
            analysis_types = ["Gender", "Age", "Emotion", "Race"]
        
        print("🔄 Preloading analysis models...")
        
        for analysis_type in analysis_types:
            try:
                from deepface.modules import modeling
                
                # Load analysis model
                model = modeling.build_model("facial_attribute", analysis_type)
                self.analysis_models[analysis_type] = model
                print(f"✅ Preloaded analysis model: {analysis_type}")
                
            except Exception as e:
                print(f"⚠️  Gagal preload analysis model {analysis_type}: {e}")
    
    def preload_all(self, 
                   detection_backends: Optional[list] = None,
                   recognition_models: Optional[list] = None,
                   analysis_types: Optional[list] = None):
        """
        Preload semua model yang dibutuhkan
        """
        print("🚀 Memulai preload semua model...")
        
        self.preload_detection_models(detection_backends)
        self.preload_recognition_models(recognition_models)
        self.preload_analysis_models(analysis_types)
        
        print("✅ Selesai preload semua model!")
    
    def get_detector(self, backend: str):
        """
        Get preloaded detector model
        """
        return self.detector_models.get(backend)
    
    def get_recognition_model(self, model_name: str):
        """
        Get preloaded recognition model
        """
        return self.recognition_models.get(model_name)
    
    def get_analysis_model(self, analysis_type: str):
        """
        Get preloaded analysis model
        """
        return self.analysis_models.get(analysis_type)

# Global preloader instance
preloader = ModelPreloader()

def check_existing_models():
    """
    Cek model yang sudah ada di folder weights
    """
    home = r"c:\Users\Asus\Documents\Kodingan\yolo\face_attendance\data\models"
    weights_dir = os.path.normpath(os.path.join(home, ".deepface", "weights"))
    
    existing_models = set()
    
    if os.path.exists(weights_dir):
        for filename in os.listdir(weights_dir):
            if filename.endswith('.h5') or filename.endswith('.pt'):
                # Extract model name from filename
                if 'arcface' in filename.lower():
                    existing_models.add('arcface')
                elif 'facenet512' in filename.lower():
                    existing_models.add('facenet512')
                elif 'facenet' in filename.lower():
                    existing_models.add('facenet')
                elif 'ghostface' in filename.lower():
                    existing_models.add('ghostfacenet')
                elif 'age' in filename.lower():
                    existing_models.add('age')
                elif 'gender' in filename.lower():
                    existing_models.add('gender')
                elif 'emotion' in filename.lower() or 'expression' in filename.lower():
                    existing_models.add('emotion')
                elif 'race' in filename.lower():
                    existing_models.add('race')
                elif 'retinaface' in filename.lower():
                    existing_models.add('retinaface')
                elif 'yolov8n' in filename.lower():
                    existing_models.add('yolov8n')
                elif 'yolov11n' in filename.lower():
                    existing_models.add('yolov11n')
    
    return existing_models

def initialize_models():
    """
    Initialize dan preload semua model yang dibutuhkan
    """
    # Cek model yang sudah ada dulu
    existing_models = check_existing_models()
    print(f"📋 Found {len(existing_models)} existing models")
    
    # Hanya preload model yang belum ada
    detection_needed = ["retinaface", "yolov8n", "yolov11n"]
    recognition_needed = ["ArcFace", "Facenet512"]
    analysis_needed = ["Gender", "Age", "Emotion", "Race"]
    
    # Filter hanya yang belum ada
    detection_to_load = [m for m in detection_needed if m.lower() not in existing_models]
    recognition_to_load = [m for m in recognition_needed if m.lower() not in existing_models]
    analysis_to_load = [m for m in analysis_needed if m.lower() not in existing_models]
    
    if detection_to_load or recognition_to_load or analysis_to_load:
        print(f"🔄 Loading missing models: detection={detection_to_load}, recognition={recognition_to_load}, analysis={analysis_to_load}")
        preloader.preload_all(
            detection_backends=detection_to_load,
            recognition_models=recognition_to_load,
            analysis_types=analysis_to_load
        )
    else:
        print("✅ All required models are already available!")

if __name__ == "__main__":
    initialize_models()