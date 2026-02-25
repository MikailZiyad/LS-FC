FACE_POSES = [
    "Hadap lurus ke kamera",
    "Hadap sedikit ke kiri", 
    "Hadap sedikit ke kanan",
    "Lihat ke atas",
    "Posisi normal dengan senyum"
]

SAMPLES_PER_POSE = 3

FACE_QUALITY_REQUIREMENTS = {
    "min_brightness": 40,
    "max_brightness": 210,
    "min_blur_score": 60,
    "min_face_size": 8000
}

ATTENDANCE_STATUS = {
    "PRESENT": "Hadir",
    "LATE": "Terlambat", 
    "ABSENT": "Tidak Hadir"
}

TIME_LATE_THRESHOLD = "09:00:00"

DATABASE_TABLES = {
    "employees": "employees",
    "attendance": "attendance",
    "face_embeddings": "face_embeddings"
}
