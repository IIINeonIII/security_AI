import os
import numpy as np
from deepface import DeepFace

DATASET_PATH = "data/faces"
OUTPUT_PATH = "data/embeddings"
MODEL_NAME = "ArcFace"


# =========================
# EMBEDDING UTIL
# =========================

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


# =========================
# BUILD DATABASE (TRAINING STEP)
# =========================

def build_face_database():
    embeddings = []
    labels = []

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing {person_name}...")

        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)

            try:
                result = DeepFace.represent(
                    img_path=file_path,
                    model_name=MODEL_NAME,
                    enforce_detection=True
                )

                embedding = result[0]["embedding"]
                embedding = normalize_embedding(np.array(embedding))

                embeddings.append(embedding)
                labels.append(person_name)

            except Exception as e:
                print(f"[WARNING] Skipping {file_path}: {e}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print(f"[INFO] Generated embeddings shape: {embeddings.shape}")

    return embeddings, labels


# =========================
# SAVE DATABASE
# =========================

def save_face_database(embeddings, labels):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    np.save(os.path.join(OUTPUT_PATH, "face_embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_PATH, "face_labels.npy"), labels)

    print(f"[INFO] Saved {len(embeddings)} face embeddings")


# =========================
# LOAD DATABASE (FOR STREAMLIT)
# =========================

def load_face_database():
    embeddings = np.load(os.path.join(OUTPUT_PATH, "face_embeddings.npy"))
    labels = np.load(os.path.join(OUTPUT_PATH, "face_labels.npy"))

    print(f"[INFO] Loaded {len(embeddings)} face embeddings")

    return embeddings, labels


# =========================
# RUN AS SCRIPT
# =========================

if __name__ == "__main__":
    embeddings, labels = build_face_database()

    if len(embeddings) == 0:
        print("[ERROR] No embeddings created. Check dataset.")
    else:
        save_face_database(embeddings, labels)
