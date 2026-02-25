# Advanced Face Recognition Configuration for DeepFace Master

# Recognition Models Configuration
RECOGNITION_MODELS = {
    "ArcFace": {
        "name": "ArcFace",
        "description": "ArcFace - Best accuracy for face recognition",
        "recommended": True,
        "accuracy": 99.4,
        "speed": "medium",
        "memory_usage": "high",
        "use_cases": ["security", "access_control", "attendance"]
    },
    "Facenet": {
        "name": "Facenet",
        "description": "FaceNet - Google's face recognition model",
        "recommended": True,
        "accuracy": 99.2,
        "speed": "fast",
        "memory_usage": "medium",
        "use_cases": ["general", "mobile", "real_time"]
    },
    "Facenet512": {
        "name": "Facenet512",
        "description": "FaceNet512 - Higher dimensional embeddings",
        "recommended": True,
        "accuracy": 99.3,
        "speed": "medium",
        "memory_usage": "high",
        "use_cases": ["high_accuracy", "large_database"]
    },
    "GhostFaceNet": {
        "name": "GhostFaceNet",
        "description": "GhostFaceNet - Lightweight and fast",
        "recommended": True,
        "accuracy": 98.8,
        "speed": "fast",
        "memory_usage": "low",
        "use_cases": ["mobile", "edge", "real_time"]
    },
    "VGG-Face": {
        "name": "VGG-Face",
        "description": "VGG-Face - Traditional deep learning model",
        "recommended": False,
        "accuracy": 97.3,
        "speed": "slow",
        "memory_usage": "high",
        "use_cases": ["legacy", "research"]
    },
    "OpenFace": {
        "name": "OpenFace",
        "description": "OpenFace - Open source face recognition",
        "recommended": False,
        "accuracy": 96.8,
        "speed": "medium",
        "memory_usage": "medium",
        "use_cases": ["research", "education"]
    },
    "DeepFace": {
        "name": "DeepFace",
        "description": "DeepFace - Facebook's face recognition",
        "recommended": False,
        "accuracy": 97.4,
        "speed": "slow",
        "memory_usage": "high",
        "use_cases": ["research", "benchmarking"]
    },
    "DeepID": {
        "name": "DeepID",
        "description": "DeepID - Deep learning face identification",
        "recommended": False,
        "accuracy": 97.1,
        "speed": "medium",
        "memory_usage": "medium",
        "use_cases": ["research", "comparison"]
    },
    "Dlib": {
        "name": "Dlib",
        "description": "Dlib - Traditional face recognition",
        "recommended": False,
        "accuracy": 96.2,
        "speed": "fast",
        "memory_usage": "low",
        "use_cases": ["legacy", "embedded"]
    }
}

# Detection Backends Configuration
DETECTION_BACKENDS = {
    "retinaface": {
        "name": "retinaface",
        "description": "RetinaFace - Most accurate face detection",
        "recommended": True,
        "accuracy": 99.1,
        "speed": "medium",
        "memory_usage": "medium",
        "face_landmarks": True,
        "use_cases": ["security", "high_accuracy", "attendance"]
    },
    "yolov8n": {
        "name": "yolov8n",
        "description": "YOLOv8 Nano - Fast and accurate",
        "recommended": True,
        "accuracy": 98.5,
        "speed": "fast",
        "memory_usage": "low",
        "face_landmarks": False,
        "use_cases": ["real_time", "mobile", "edge"]
    },
    "yolov8m": {
        "name": "yolov8m",
        "description": "YOLOv8 Medium - Balanced accuracy and speed",
        "recommended": True,
        "accuracy": 98.8,
        "speed": "medium",
        "memory_usage": "medium",
        "face_landmarks": False,
        "use_cases": ["general", "attendance", "access_control"]
    },
    "yolov8l": {
        "name": "yolov8l",
        "description": "YOLOv8 Large - High accuracy",
        "recommended": True,
        "accuracy": 99.0,
        "speed": "slow",
        "memory_usage": "high",
        "face_landmarks": False,
        "use_cases": ["high_accuracy", "security"]
    },
    "mtcnn": {
        "name": "mtcnn",
        "description": "MTCNN - Multi-task CNN face detection",
        "recommended": True,
        "accuracy": 98.2,
        "speed": "medium",
        "memory_usage": "medium",
        "face_landmarks": True,
        "use_cases": ["general", "attendance", "mobile"]
    },
    "mediapipe": {
        "name": "mediapipe",
        "description": "MediaPipe - Google's face detection",
        "recommended": True,
        "accuracy": 97.8,
        "speed": "fast",
        "memory_usage": "low",
        "face_landmarks": True,
        "use_cases": ["real_time", "mobile", "webcam"]
    },
    "dlib": {
        "name": "dlib",
        "description": "Dlib - Traditional face detection",
        "recommended": False,
        "accuracy": 96.5,
        "speed": "medium",
        "memory_usage": "low",
        "face_landmarks": True,
        "use_cases": ["legacy", "embedded", "research"]
    },
    "opencv": {
        "name": "opencv",
        "description": "OpenCV - Haar cascade face detection",
        "recommended": False,
        "accuracy": 94.2,
        "speed": "fast",
        "memory_usage": "very_low",
        "face_landmarks": False,
        "use_cases": ["legacy", "resource_constrained"]
    },
    "ssd": {
        "name": "ssd",
        "description": "SSD - Single Shot MultiBox Detector",
        "recommended": False,
        "accuracy": 96.8,
        "speed": "fast",
        "memory_usage": "medium",
        "face_landmarks": False,
        "use_cases": ["real_time", "mobile"]
    }
}

# Distance Metrics Configuration
DISTANCE_METRICS = {
    "cosine": {
        "name": "cosine",
        "description": "Cosine similarity - Recommended for face embeddings",
        "recommended": True,
        "range": "0-1",
        "threshold": 0.4,
        "use_cases": ["face_recognition", "embedding_comparison"]
    },
    "euclidean": {
        "name": "euclidean",
        "description": "Euclidean distance - Traditional distance metric",
        "recommended": False,
        "range": "0-infinity",
        "threshold": 10.0,
        "use_cases": ["general", "traditional_ml"]
    },
    "euclidean_l2": {
        "name": "euclidean_l2",
        "description": "L2 normalized Euclidean distance",
        "recommended": False,
        "range": "0-2",
        "threshold": 0.8,
        "use_cases": ["normalized_embeddings", "research"]
    }
}

# Anti-Spoofing Configuration
ANTI_SPOOFING_CONFIG = {
    "enabled": True,
    "model_name": "FasNet",
    "confidence_threshold": 0.5,
    "fallback_models": ["Silent-Face", "FAS-ResNet"],
    "use_cases": ["security", "access_control", "attendance"]
}

# Face Quality Requirements (Advanced)
ADVANCED_FACE_QUALITY_REQUIREMENTS = {
    "min_quality_score": 60,
    "max_blur_score": 100,
    "min_brightness": 40,
    "max_brightness": 210,
    "min_contrast": 20,
    "min_face_size": (80, 80),
    "max_face_size": (400, 400),
    "required_landmarks": ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"],
    "min_landmark_visibility": 0.8,
    "max_head_pose_angle": 30,  # degrees
    "max_occlusion": 0.2,  # 20% max occlusion
    "min_eye_distance": 30,  # pixels
    "max_eye_distance": 200  # pixels
}

# Multi-Model Ensemble Configuration
ENSEMBLE_CONFIG = {
    "enabled": True,
    "models": ["ArcFace", "Facenet", "GhostFaceNet"],
    "weights": [0.5, 0.3, 0.2],  # Model weights for ensemble
    "voting_method": "weighted_average",  # weighted_average, majority_vote
    "confidence_threshold": 0.7,
    "max_ensemble_size": 3,
    "use_cases": ["high_security", "critical_applications"]
}

# Vector Database Configuration (for large scale)
VECTOR_DATABASE_CONFIG = {
    "enabled": False,  # Enable for large databases (>1000 employees)
    "provider": "faiss",  # faiss, pinecone, weaviate, milvus
    "index_type": "IVFFlat",  # IVFFlat, HNSW, Flat
    "nlist": 100,  # Number of clusters for IVFFlat
    "nprobe": 10,  # Number of clusters to search
    "dimension": 512,  # Embedding dimension (depends on model)
    "metric": "cosine",  # cosine, euclidean, inner_product
    "use_gpu": True,  # Use GPU for vector search
    "batch_size": 32,
    "max_results": 10
}

# Real-time Streaming Configuration
STREAMING_CONFIG = {
    "enabled": True,
    "frame_skip": 2,  # Process every Nth frame
    "detection_interval": 5,  # Run detection every N frames
    "tracking_enabled": True,
    "tracker_type": "CSRT",  # CSRT, KCF, MOSSE
    "max_tracking_age": 30,  # frames
    "min_tracking_confidence": 0.3,
    "batch_processing": False,  # Process multiple frames at once
    "use_gpu": True,
    "max_fps": 30,
    "buffer_size": 10
}

# Performance Optimization Configuration
PERFORMANCE_CONFIG = {
    "model_cache_size": 3,  # Number of models to keep in memory
    "embedding_cache_size": 1000,  # Number of embeddings to cache
    "batch_size": 32,  # Batch size for processing
    "use_multiprocessing": True,
    "num_workers": 4,  # Number of worker processes
    "gpu_memory_fraction": 0.8,  # GPU memory usage limit
    "cpu_threads": 8,  # Number of CPU threads
    "optimize_for": "accuracy"  # accuracy, speed, balanced
}

# Model Fallback Configuration
FALLBACK_CONFIG = {
    "enabled": True,
    "fallback_order": {
        "recognition": ["ArcFace", "Facenet", "GhostFaceNet", "Facenet512"],
        "detection": ["retinaface", "yolov8m", "yolov8n", "mtcnn", "mediapipe"]
    },
    "max_fallback_attempts": 3,
    "fallback_threshold": 0.5,  # Confidence threshold for fallback
    "log_fallbacks": True,
    "notify_on_fallback": True
}

# Database Integration Configuration
DATABASE_CONFIG = {
    "embeddings_table": "face_embeddings_advanced",
    "employees_table": "employees_advanced",
    "attendance_table": "attendance_advanced",
    "metadata_table": "face_metadata",
    "use_json_columns": True,  # Store metadata as JSON
    "embedding_compression": True,  # Compress embeddings
    "backup_embeddings": True,  # Keep backup of embeddings
    "audit_log": True,  # Log all database operations
    "encryption": False  # Encrypt sensitive data
}

# Security and Privacy Configuration
SECURITY_CONFIG = {
    "encrypt_embeddings": False,  # Encrypt face embeddings
    "hash_employee_ids": False,  # Hash employee IDs
    "audit_logging": True,  # Log all recognition events
    "data_retention_days": 365,  # Keep data for 1 year
    "anonymize_data": False,  # Anonymize sensitive data
    "access_control": True,  # Role-based access control
    "rate_limiting": True,  # Prevent abuse
    "max_attempts_per_minute": 60
}

# Logging and Monitoring Configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_to_file": True,
    "log_file": "logs/face_recognition_advanced.log",
    "max_file_size": "10MB",
    "backup_count": 5,
    "log_performance_metrics": True,
    "log_errors": True,
    "log_recognition_events": True,
    "monitor_performance": True,
    "alert_thresholds": {
        "accuracy": 0.95,
        "speed": 100,  # ms
        "memory_usage": 0.8,  # 80% of available memory
        "error_rate": 0.01  # 1% error rate
    }
}

# Default Configuration
DEFAULT_ADVANCED_CONFIG = {
    "recognition_model": "ArcFace",
    "detection_backend": "retinaface",
    "distance_metric": "cosine",
    "confidence_threshold": 0.4,
    "enable_anti_spoofing": True,
    "enable_ensemble": True,
    "enable_vector_database": False,
    "enable_streaming": True,
    "optimize_for": "balanced",
    "use_gpu": True,
    # Registration defaults expected by AdvancedRegistrationUI
    "face_poses": ["front", "left", "right", "up", "smile"]
}

# Global configuration object (for backward compatibility)
ADVANCED_CONFIG = DEFAULT_ADVANCED_CONFIG
