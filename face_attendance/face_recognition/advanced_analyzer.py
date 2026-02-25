"""
Advanced Face Analyzer using DeepFace master
Menganalisis atribut wajah: gender, umur, emosi, ras
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from deepface import DeepFace
from deepface.modules.demography import analyze as deepface_analyze
from deepface.commons.logger import Logger

logger = Logger()

class AdvancedFaceAnalyzer:
    """
    Analyzer wajah canggih untuk ekstraksi atribut wajah
    """
    
    def __init__(self):
        self.last_analysis = None
        self.last_error = None
        
    def analyze_face_attributes(self, 
                                face_image: np.ndarray,
                                actions: List[str] = ["age", "gender", "emotion", "race"],
                                detector_backend: str = "retinaface",
                                enforce_detection: bool = False) -> Optional[Dict[str, Any]]:
        """
        Analisis atribut wajah lengkap dengan preprocessing yang lebih baik
        
        Args:
            face_image: Gambar wajah dalam format numpy array
            actions: Daftar atribut yang ingin dianalisis
            detector_backend: Backend deteksi wajah
            enforce_detection: Apakah harus memaksa deteksi wajah
            
        Returns:
            Dictionary berisi hasil analisis
        """
        try:
            # Preprocessing untuk hasil analisis yang lebih baik
            # 1. Validasi input
            if face_image is None or face_image.size == 0:
                print("❌ Face image is None or empty")
                return None
            
            # 2. Normalisasi tipe data gambar ke uint8
            analysis_image = face_image
            if analysis_image.dtype == np.float64 or analysis_image.dtype == np.float32:
                # Jika dalam rentang 0..1, scale ke 0..255
                if analysis_image.max() <= 1.0:
                    analysis_image = (analysis_image * 255.0)
                analysis_image = np.clip(analysis_image, 0, 255).astype(np.uint8)
            elif analysis_image.dtype != np.uint8:
                analysis_image = np.clip(analysis_image, 0, 255).astype(np.uint8)
            
            # 3. Konversi ke format warna yang tepat
            if len(analysis_image.shape) == 2 or (len(analysis_image.shape) == 3 and analysis_image.shape[2] == 1):
                # Grayscale ke BGR
                analysis_image = cv2.cvtColor(analysis_image, cv2.COLOR_GRAY2BGR)
            elif len(analysis_image.shape) == 3 and analysis_image.shape[2] == 3:
                # Biarkan sebagai BGR (umumnya dari OpenCV). Jika input RGB, DeepFace tetap dapat menangani.
                pass
            
            # 4. Validasi ukuran - pastikan cukup besar untuk analisis
            h, w = analysis_image.shape[:2]
            if h < 100 or w < 100:
                print(f"⚠️  Face image too small ({w}x{h}), resizing...")
                # Resize ke ukuran yang lebih besar
                analysis_image = cv2.resize(analysis_image, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # 5. Enhancement untuk analisis yang lebih baik
            # Convert to LAB for better contrast enhancement
            if len(analysis_image.shape) == 3:
                lab = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge back
                enhanced_lab = cv2.merge([l, a, b])
                analysis_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 6. Analisis menggunakan DeepFace dengan parameter yang lebih baik
            print(f"🔬 Starting DeepFace analysis with image shape: {analysis_image.shape}")
            analysis_result = deepface_analyze(
                img_path=analysis_image,
                actions=tuple(actions),
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
                align=True,
                silent=False  # Enable logging untuk debugging
            )
            
            print(f"🔬 DeepFace analysis result: {analysis_result}")
            
            if analysis_result and len(analysis_result) > 0:
                # Ambil hasil dari wajah pertama yang terdeteksi
                face_data = analysis_result[0]
                
                # Ekstrak informasi penting dengan validasi
                result = {
                    "age": max(0, int(face_data.get("age", 0))),  # Pastikan umur tidak negatif
                    "gender": face_data.get("dominant_gender", "unknown"),
                    "gender_confidence": float(face_data.get("gender", {}).get("confidence", 0.0)),
                    "emotion": face_data.get("dominant_emotion", "neutral"),
                    "emotion_confidence": float(face_data.get("emotion", {}).get("confidence", 0.0)),
                    "race": face_data.get("dominant_race", "unknown"),
                    "race_confidence": float(face_data.get("race", {}).get("confidence", 0.0)),
                    "region": face_data.get("region", {}),
                    "face_confidence": float(face_data.get("face_confidence", 0.0))
                }
                
                # Validasi hasil emosi - jika confidence terlalu rendah, gunakan neutral
                if result["emotion_confidence"] < 0.3:
                    print(f"⚠️  Emotion confidence too low ({result['emotion_confidence']:.3f}), defaulting to neutral")
                    result["emotion"] = "neutral"
                
                # Validasi hasil gender
                if result["gender_confidence"] < 0.3:
                    print(f"⚠️  Gender confidence too low ({result['gender_confidence']:.3f}), defaulting to unknown")
                    result["gender"] = "unknown"
                
                # Validasi umur - jika terlalu muda atau tua, mungkin error
                if result["age"] < 5 or result["age"] > 90:
                    print(f"⚠️  Age seems unrealistic ({result['age']}), checking confidence...")
                    # Jika confidence rendah, beri warning
                    if face_data.get("age", 0) < 0.5:
                        print("⚠️  Age prediction has low confidence")
                
                self.last_analysis = result
                print(f"✅ Analysis completed successfully: age={result['age']}, gender={result['gender']}, emotion={result['emotion']}")
                return result
            else:
                print("❌ No analysis results returned")
                return None
                
        except Exception as e:
            self.last_error = f"Face analysis failed: {str(e)}"
            print(f"❌ {self.last_error}")
            return None
    
    def analyze_face_with_consensus(self, face_image: np.ndarray, num_iterations: int = 3) -> Dict[str, Any]:
        """
        Analisis wajah dengan multiple predictions untuk hasil yang lebih akurat
        
        Args:
            face_image: Gambar wajah
            num_iterations: Jumlah prediksi yang akan dilakukan
            
        Returns:
            Hasil analisis dengan consensus
        """
        print(f"🔬 Starting consensus analysis with {num_iterations} iterations...")
        
        results = []
        for i in range(num_iterations):
            try:
                # Tambahkan sedikit noise atau augmentasi untuk variasi
                if i > 0 and face_image is not None:
                    # Random brightness adjustment
                    alpha = np.random.uniform(0.9, 1.1)
                    beta = np.random.uniform(-10, 10)
                    adjusted_image = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)
                    result = self.analyze_face_attributes(adjusted_image)
                else:
                    result = self.analyze_face_attributes(face_image)
                
                if result:
                    results.append(result)
                    print(f"  Iteration {i+1}: age={result.get('age', 0)}, emotion={result.get('emotion', 'unknown')}")
                
            except Exception as e:
                print(f"  Iteration {i+1} failed: {e}")
                continue
        
        if not results:
            print("❌ All consensus iterations failed")
            return {}
        
        # Consensus untuk umur - ambil median
        ages = [r.get("age", 0) for r in results]
        consensus_age = int(np.median(ages))
        
        # Consensus untuk gender - voting
        genders = [r.get("gender", "unknown") for r in results]
        gender_counts = {}
        for g in genders:
            gender_counts[g] = gender_counts.get(g, 0) + 1
        consensus_gender = max(gender_counts.items(), key=lambda x: x[1])[0] if gender_counts else "unknown"
        
        # Consensus untuk emosi - voting dengan confidence
        emotions = [r.get("emotion", "neutral") for r in results]
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        consensus_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        
        # Consensus untuk ras - voting
        races = [r.get("race", "unknown") for r in results]
        race_counts = {}
        for r in races:
            race_counts[r] = race_counts.get(r, 0) + 1
        consensus_race = max(race_counts.items(), key=lambda x: x[1])[0] if race_counts else "unknown"
        
        # Hitung confidence rata-rata
        avg_confidence = np.mean([r.get("face_confidence", 0.0) for r in results])
        
        consensus_result = {
            "age": consensus_age,
            "gender": consensus_gender,
            "emotion": consensus_emotion,
            "race": consensus_race,
            "face_confidence": avg_confidence,
            "consensus_count": len(results),
            "age_variance": np.var(ages),
            "raw_results": results  # Simpan hasil mentah untuk debugging
        }
        
        print(f"✅ Consensus result: age={consensus_age}, gender={consensus_gender}, emotion={consensus_emotion}, confidence={avg_confidence:.3f}")
        return consensus_result
    
    def analyze_face(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Wrapper method untuk analyze_face_attributes - versi sederhana dengan consensus
        
        Args:
            face_image: Gambar wajah dalam format numpy array
            
        Returns:
            Dictionary berisi hasil analisis
        """
        print(f"🔬 analyze_face called with image shape: {face_image.shape}")
        try:
            # Gunakan consensus approach untuk hasil yang lebih akurat
            consensus_result = self.analyze_face_with_consensus(face_image, num_iterations=3)
            
            if consensus_result:
                # Konversi ke format standar
                result = {
                    "age": consensus_result.get("age", 0),
                    "gender": consensus_result.get("gender", "unknown"),
                    "emotion": consensus_result.get("emotion", "neutral"),
                    "race": consensus_result.get("race", "unknown"),
                    "gender_confidence": consensus_result.get("face_confidence", 0.0),
                    "emotion_confidence": consensus_result.get("face_confidence", 0.0),
                    "race_confidence": consensus_result.get("face_confidence", 0.0),
                    "face_confidence": consensus_result.get("face_confidence", 0.0),
                    "consensus_info": consensus_result  # Simpan info consensus
                }
                print(f"🔬 analyze_face consensus result: {result}")
                return result
            else:
                # Fallback ke single analysis jika consensus gagal
                print("⚠️  Consensus analysis failed, falling back to single analysis")
                result = self.analyze_face_attributes(face_image) or {}
                print(f"🔬 analyze_face fallback result: {result}")
                return result
                
        except Exception as e:
            print(f"❌ analyze_face error: {e}")
            return {}
    
    def analyze_emotion(self, face_image: np.ndarray) -> Optional[str]:
        """
        Analisis emosi saja dari wajah
        
        Args:
            face_image: Gambar wajah dalam format numpy array
            
        Returns:
            Emosi dominan atau None jika gagal
        """
        result = self.analyze_face_attributes(face_image, actions=["emotion"])
        if result and "dominant_emotion" in result:
            return result["dominant_emotion"]
        return None
    
    def analyze_faces_in_frame(self, 
                              frame: np.ndarray,
                              detected_faces: List[Dict[str, Any]],
                              actions: List[str] = ["age", "gender", "emotion", "race"]) -> List[Dict[str, Any]]:
        """
        Analisis atribut untuk semua wajah yang terdeteksi dalam frame
        
        Args:
            frame: Frame video
            detected_faces: List wajah yang sudah terdeteksi
            actions: Atribut yang ingin dianalisis
            
        Returns:
            List hasil analisis untuk setiap wajah
        """
        results = []
        
        for face_data in detected_faces:
            try:
                # Ekstrak area wajah dari frame
                facial_area = face_data.get("facial_area", {})
                if facial_area:
                    x = int(facial_area.get("x", 0))
                    y = int(facial_area.get("y", 0))
                    w = int(facial_area.get("w", 0))
                    h = int(facial_area.get("h", 0))
                    
                    # Pastikan koordinat dalam batas frame
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(frame.shape[1], x + w)
                    y2 = min(frame.shape[0], y + h)
                    
                    # Ekstrak wajah dari frame
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Analisis wajah
                        analysis = self.analyze_face_attributes(
                            face_image=face_crop,
                            actions=actions,
                            enforce_detection=False
                        )
                        
                        if analysis:
                            # Gabungkan dengan data deteksi asli
                            enhanced_face = face_data.copy()
                            enhanced_face["analysis"] = analysis
                            results.append(enhanced_face)
                        else:
                            # Gunakan data asli jika analisis gagal
                            results.append(face_data)
                    else:
                        results.append(face_data)
                else:
                    results.append(face_data)
                    
            except Exception as e:
                logger.error(f"Error analyzing face: {str(e)}")
                # Simpan data asli jika terjadi error
                results.append(face_data)
        
        return results
    
    def format_analysis_text(self, analysis: Dict[str, Any]) -> str:
        """
        Format hasil analisis menjadi teks untuk ditampilkan
        
        Args:
            analysis: Hasil analisis wajah
            
        Returns:
            Teks yang sudah diformat
        """
        if not analysis:
            return "No analysis"
        
        # Format informasi dasar
        age = analysis.get("age", 0)
        gender = analysis.get("gender", "unknown")
        emotion = analysis.get("emotion", "neutral")
        race = analysis.get("race", "unknown")
        
        # Konversi ke format yang lebih pendek
        gender_symbol = "♂" if gender.lower() == "man" else "♀" if gender.lower() == "woman" else "?"
        
        # Buat teks yang informatif
        text_lines = [
            f"Age: {age}",
            f"Gender: {gender_symbol} {gender.title()}",
            f"Emotion: {emotion.title()}",
            f"Race: {race.title()}"
        ]
        
        return "\n".join(text_lines)
    
    def get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Dapatkan warna berdasarkan confidence
        
        Args:
            confidence: Nilai confidence (0-1)
            
        Returns:
            Tuple warna BGR
        """
        if confidence >= 0.8:
            return (0, 255, 0)  # Hijau - tinggi
        elif confidence >= 0.6:
            return (0, 255, 255)  # Kuning - sedang
        else:
            return (0, 0, 255)  # Merah - rendah
