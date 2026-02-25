#!/usr/bin/env python3
"""Test script untuk memverifikasi semua imports berfungsi dengan baik."""

print("🧪 Testing imports...")

try:
    import cv2
    print("✅ cv2 - OK")
except ImportError as e:
    print(f"❌ cv2 - ERROR: {e}")

try:
    import numpy as np
    print("✅ numpy - OK")
except ImportError as e:
    print(f"❌ numpy - ERROR: {e}")

try:
    import torch
    print("✅ torch - OK")
except ImportError as e:
    print(f"❌ torch - ERROR: {e}")

try:
    from deepface import DeepFace
    print("✅ deepface - OK")
except ImportError as e:
    print(f"❌ deepface - ERROR: {e}")

try:
    import customtkinter as ctk
    print("✅ customtkinter - OK")
except ImportError as e:
    print(f"❌ customtkinter - ERROR: {e}")

try:
    from ultralytics import YOLO
    print("✅ ultralytics - OK")
except ImportError as e:
    print(f"❌ ultralytics - ERROR: {e}")

try:
    from PIL import Image, ImageTk
    print("✅ PIL - OK")
except ImportError as e:
    print(f"❌ PIL - ERROR: {e}")

print("\n🎯 Test selesai!")
