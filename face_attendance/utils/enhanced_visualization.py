"""
Utility functions for drawing enhanced face borders with analysis information
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from face_recognition.advanced_analyzer import AdvancedFaceAnalyzer

def draw_enhanced_face_border(
    frame: np.ndarray,
    face_data: Dict[str, Any],
    user_info: Optional[Dict[str, Any]] = None,
    border_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Menggambar border box dengan informasi lengkap untuk wajah
    
    Args:
        frame: Frame video
        face_data: Data wajah dari detektor
        user_info: Informasi user (untuk absensi)
        border_thickness: Ketebalan border
        font_scale: Skala font
        font_thickness: Ketebalan font
        
    Returns:
        Frame yang sudah diberi border dan informasi
    """
    # Ekstrak informasi wajah
    bbox = face_data.get("bbox", (0, 0, 0, 0))
    x1, y1, x2, y2 = bbox
    confidence = face_data.get("confidence", 0.0)
    analysis = face_data.get("analysis", {})
    
    # Hitung warna berdasarkan confidence
    if confidence >= 0.8:
        border_color = (0, 255, 0)  # Hijau - tinggi
    elif confidence >= 0.6:
        border_color = (0, 255, 255)  # Kuning - sedang
    else:
        border_color = (0, 0, 255)  # Merah - rendah
    
    # Gambar border utama
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)
    
    # Siapkan teks informasi
    text_lines = []
    
    # Informasi user (untuk absensi)
    if user_info:
        user_name = user_info.get("name", "Unknown")
        user_id = user_info.get("user_id", "N/A")
        text_lines.append(f"Name: {user_name}")
        text_lines.append(f"ID: {user_id}")
    
    # Informasi analisis wajah
    if analysis:
        age = analysis.get("age", 0)
        gender = analysis.get("gender", "unknown")
        emotion = analysis.get("emotion", "neutral")
        race = analysis.get("race", "unknown")
        
        # Simbol gender
        gender_symbol = "♂" if gender.lower() == "man" else "♀" if gender.lower() == "woman" else "?"
        
        text_lines.extend([
            f"Age: {age}",
            f"Gender: {gender_symbol} {gender.title()}",
            f"Emotion: {emotion.title()}",
            f"Race: {race.title()}",
            f"Conf: {confidence:.2f}"
        ])
    else:
        # Informasi dasar jika tidak ada analisis
        text_lines.extend([
            f"Face Detected",
            f"Conf: {confidence:.2f}"
        ])
    
    # Gambar background untuk teks
    text_height = len(text_lines) * 25 + 10
    text_y = max(0, y1 - text_height)
    
    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, text_y), (x1 + 200, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Tulis teks
    for i, line in enumerate(text_lines):
        y_pos = text_y + 20 + (i * 20)
        cv2.putText(frame, line, (x1 + 5, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # Tambahkan indikator status kecil di pojok
    status_size = 15
    if confidence >= 0.8:
        status_color = (0, 255, 0)
    elif confidence >= 0.6:
        status_color = (0, 255, 255)
    else:
        status_color = (0, 0, 255)
    
    cv2.circle(frame, (x2 - status_size, y1 + status_size), status_size // 2, status_color, -1)
    
    return frame

def draw_registration_border(
    frame: np.ndarray,
    face_data: Dict[str, Any],
    capture_status: str = "waiting",
    border_thickness: int = 3,
    font_scale: float = 0.7,
    font_thickness: int = 2
) -> np.ndarray:
    """
    Border khusus untuk mode registrasi dengan informasi lengkap
    
    Args:
        frame: Frame video
        face_data: Data wajah dari detektor
        capture_status: Status capture (waiting, capturing, captured)
        border_thickness: Ketebalan border
        font_scale: Skala font
        font_thickness: Ketebalan font
        
    Returns:
        Frame yang sudah diberi border registrasi
    """
    # Ekstrak informasi
    bbox = face_data.get("bbox", (0, 0, 0, 0))
    x1, y1, x2, y2 = bbox
    analysis = face_data.get("analysis", {})
    
    # Warna berdasarkan status
    if capture_status == "waiting":
        border_color = (255, 165, 0)  # Orange
    elif capture_status == "capturing":
        border_color = (0, 255, 255)  # Kuning
    elif capture_status == "captured":
        border_color = (0, 255, 0)  # Hijau
    else:
        border_color = (255, 0, 0)  # Merah
    
    # Border utama dengan style khusus
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)
    
    # Corner decorations untuk estetika
    corner_length = 20
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), border_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), border_color, 3)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), border_color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), border_color, 3)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), border_color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), border_color, 3)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), border_color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), border_color, 3)
    
    # Informasi analisis wajah
    if analysis:
        info_lines = []
        
        # Age
        age = analysis.get("age", 0)
        info_lines.append(f"Age: {age}")
        
        # Gender dengan simbol
        gender = analysis.get("gender", "unknown")
        gender_symbol = "♂" if gender.lower() == "man" else "♀" if gender.lower() == "woman" else "?"
        info_lines.append(f"Gender: {gender_symbol} {gender.title()}")
        
        # Emotion
        emotion = analysis.get("emotion", "neutral")
        emotion_emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "surprise": "😲",
            "fear": "😨",
            "disgust": "🤢",
            "neutral": "😐"
        }.get(emotion.lower(), "😐")
        info_lines.append(f"Emotion: {emotion_emoji} {emotion.title()}")
        
        # Race
        race = analysis.get("race", "unknown")
        info_lines.append(f"Race: {race.title()}")
        
        # Status registrasi
        info_lines.append(f"Status: {capture_status.title()}")
        
        # Gambar background untuk teks
        text_width = max(len(line) for line in info_lines) * 12
        text_height = len(info_lines) * 25 + 10
        
        # Background di samping wajah
        bg_x1 = x2 + 10
        bg_y1 = y1
        bg_x2 = bg_x1 + text_width
        bg_y2 = bg_y1 + text_height
        
        # Background semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Tulis teks
        for i, line in enumerate(info_lines):
            y_pos = bg_y1 + 20 + (i * 25)
            cv2.putText(frame, line, (bg_x1 + 5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # Judul registrasi
    title = "FACE REGISTRATION"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    title_x = x1 + (x2 - x1 - title_size[0]) // 2
    title_y = y1 - 10
    
    cv2.putText(frame, title, (title_x, title_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
    
    return frame