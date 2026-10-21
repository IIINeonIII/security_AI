import os
import numpy as np
from deepface import DeepFace
from speechbrain.inference.speaker import EncoderClassifier
import torch
import torchaudio

from utils.preprocessing import align_face, normalize_image, preprocess_audio


# =========================
# CONFIG
# =========================

FACE_DATA_PATH = "data/faces"
VOICE_DATA_PATH = "data/voices"

FACE_DB_EMB = "data/embeddings/face_embeddings.npy"
FACE_DB_LABELS = "data/embeddings/face_labels.npy"

VOICE_DB_EMB = "data/embeddings/voice_embeddings.npy"
VOICE_DB_LABELS = "data/embeddings/voice_labels.npy"

MODEL_NAME = "ArcFace"

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)


# =========================
# FACE ENROLLMENT
# =========================

def extract_face_embedding(image_path):
    """
    Extract ArcFace embedding from image.
    """

    try:
        image = align_face(image_path)
        image = normalize_image(image)

        result = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            enforce_detection=True
        )

        embedding = np.array(result[0]["embedding"])
        norm = np.linalg.norm(embedding)

        return embedding / norm if norm != 0 else embedding

    except Exception as e:
        print(f"[WARNING] Face error: {e}")
        return None


# =========================
# VOICE ENROLLMENT
# =========================

def extract_voice_embedding(audio_path):
    """
    Extract speaker embedding from audio.
    """

    try:
        signal = preprocess_audio(audio_path)

        with torch.no_grad():
            embedding = speaker_model.encode_batch(signal)

        return np.array(embedding.squeeze().cpu())

    except Exception as e:
        print(f"[WARNING] Voice error: {e}")
        return None


# =========================
# DATABASE UPDATE HELPERS
# =========================

def append_to_db(new_embeddings, new_labels, emb_path, label_path):
    """
    Append new data to existing database.
    """

    if os.path.exists(emb_path):
        old_embeddings = np.load(emb_path)
        new_embeddings = np.vstack([old_embeddings, new_embeddings])

    if os.path.exists(label_path):
        old_labels = np.load(label_path)
        new_labels = np.append(old_labels, new_labels)

    np.save(emb_path, new_embeddings)
    np.save(label_path, new_labels)


# =========================
# MAIN ENROLLMENT LOGIC
# =========================

def enroll_user(user_name):
    """
    Enroll a new user into the system.
    """

    face_embeddings = []
    voice_embeddings = []

    face_labels = []
    voice_labels = []

    # -------------------------
    # FACE PROCESSING
    # -------------------------
    face_path = os.path.join(FACE_DATA_PATH, user_name)

    if os.path.isdir(face_path):
        for file in os.listdir(face_path):
            emb = extract_face_embedding(os.path.join(face_path, file))
            if emb is not None:
                face_embeddings.append(emb)
                face_labels.append(user_name)
    else:
        print("[INFO] No face data found.")

    # -------------------------
    # VOICE PROCESSING
    # -------------------------
    voice_path = os.path.join(VOICE_DATA_PATH, user_name)

    if os.path.isdir(voice_path):
        for file in os.listdir(voice_path):
            emb = extract_voice_embedding(os.path.join(voice_path, file))
            if emb is not None:
                voice_embeddings.append(emb)
                voice_labels.append(user_name)
    else:
        print("[INFO] No voice data found.")

    # -------------------------
    # SAVE TO DATABASE
    # -------------------------

    if len(face_embeddings) > 0:
        append_to_db(
            np.array(face_embeddings),
            np.array(face_labels),
            FACE_DB_EMB,
            FACE_DB_LABELS
        )

    if len(voice_embeddings) > 0:
        append_to_db(
            np.array(voice_embeddings),
            np.array(voice_labels),
            VOICE_DB_EMB,
            VOICE_DB_LABELS
        )

    print(f"[INFO] User '{user_name}' enrolled successfully.")


# =========================
# RUN SCRIPT
# =========================

if __name__ == "__main__":
    name = input("Enter new user name: ")
    enroll_user(name)