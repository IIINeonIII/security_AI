import cv2
import time

import os

print("ROOT FILES:", os.listdir())


from config import CAMERA_INDEX

from modules.face.face_inference import recognize_face
from modules.voice.voice_inference import recognize_voice
from modules.auth.authentication import authenticate
from utils.logger import log_access_attempt

from scripts.build_face_database import load_face_database
from scripts.build_voice_database import load_voice_database


# =========================
# LOAD DATABASES
# =========================

face_embeddings, face_labels = load_face_database()
voice_embeddings, voice_labels = load_voice_database()


# =========================
# CAMERA MANAGER (AUTO-RECOVERY)
# =========================

class CameraManager:
    def __init__(self, index=CAMERA_INDEX, retries=3, delay=1):
        self.index = index
        self.retries = retries
        self.delay = delay
        self.cam = None
        self.connect()

    def connect(self):
        for attempt in range(self.retries):
            self.cam = cv2.VideoCapture(self.index)

            if self.cam.isOpened():
                print(f"[INFO] Camera connected (attempt {attempt + 1})")
                return True

            print(f"[WARNING] Camera failed (attempt {attempt + 1})")
            time.sleep(self.delay)

        print("[ERROR] Could not connect to camera.")
        self.cam = None
        return False

    def read_frame(self):
        if self.cam is None or not self.cam.isOpened():
            print("[WARNING] Camera lost. Reconnecting...")
            self.connect()

        if self.cam is None:
            return None

        ret, frame = self.cam.read()

        if not ret:
            print("[WARNING] Frame read failed. Reconnecting...")
            self.cam.release()
            self.connect()

            if self.cam is None:
                return None

            ret, frame = self.cam.read()

            if not ret:
                return None

        return frame

    def release(self):
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released.")


# =========================
# INITIALIZE CAMERA
# =========================

camera = CameraManager()


# =========================
# FACE CAPTURE
# =========================

def capture_face_image():
    frame = camera.read_frame()

    if frame is None:
        return None

    return frame


# =========================
# VOICE CAPTURE (TEMP)
# =========================

def capture_voice():
    audio_path = input("Enter path to voice file: ")
    return audio_path


# =========================
# MAIN SYSTEM LOOP
# =========================

def run_system():

    print("\n=== BIOMETRIC SECURITY SYSTEM ===\n")

    try:
        while True:

            input("\nPress ENTER to start authentication...")

            # -------------------------
            # Step 1: Capture inputs
            # -------------------------
            face_input = capture_face_image()
            voice_input = capture_voice()

            if face_input is None:
                print("[ERROR] No face captured.")
                continue

            # -------------------------
            # Step 2: Face inference
            # -------------------------
            face_result = recognize_face(
                face_input,
                face_embeddings,
                face_labels
            )

            # -------------------------
            # Step 3: Voice inference
            # -------------------------
            voice_result = recognize_voice(
                voice_input,
                voice_embeddings,
                voice_labels
            )

            # -------------------------
            # Step 4: Authentication
            # -------------------------
            auth_result = authenticate(face_result, voice_result)

            # -------------------------
            # Step 5: Logging
            # -------------------------
            log_access_attempt(
                user_label=face_result.get("label"),
                face_result=face_result,
                voice_result=voice_result,
                final_decision=auth_result["final_decision"]
            )

            # -------------------------
            # Step 6: Output
            # -------------------------
            print("\n--- RESULT ---")
            print("Face:", face_result)
            print("Voice:", voice_result)
            print("Decision:", auth_result["final_decision"])
            print("Confidence:", auth_result["confidence"])
            print("Mode:", auth_result["mode"])

    finally:
        # =========================
        # CLEANUP
        # =========================
        camera.release()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_system()