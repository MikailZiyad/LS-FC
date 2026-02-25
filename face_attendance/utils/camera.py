import cv2
import threading
import time
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

class CameraManager:
    def __init__(self, camera_index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
    
    def start(self):
        if self.running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not self.cap.isOpened():
                print("❌ Kamera tidak dapat dibuka")
                return False
            
            # Warm up camera
            for _ in range(10):
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    break
                time.sleep(0.1)
            
            if self.frame is None:
                print("❌ Kamera tidak memberikan frame")
                return False
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"✅ Kamera berhasil dimulai ({self.width}x{self.height})")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def _capture_frames(self):
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame = frame
                    else:
                        print("⚠️  Gagal membaca frame dari kamera")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"⚠️  Error capturing frame: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        print("✅ Kamera dihentikan")
    
    def is_running(self):
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self):
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'index': self.camera_index
            }
        return None