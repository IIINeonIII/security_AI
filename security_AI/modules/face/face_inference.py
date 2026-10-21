import numpy as np
from deepface import DeepFace

from utils.distance import find_best_match, is_match
from utils.preprocessing import align_face, normalize_image
from config import FACE_MODEL, FACE_THRESHOLD, DEBUG_MODE

MODEL_NAME = FACE_MODEL


def normalize_embedding(embedding):
    """
    Normalize embedding to unit length.
    Ensures cosine similarity works correctly.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def extract_embedding(image):
    """
    Extract face embedding from an image with preprocessing.

    Steps:
        1. Align face (fix rotation / pose)
        2. Normalize image
        3. Run ArcFace model
        4. Normalize embedding
    """

    try:
        # 🧠 Step 1: Preprocessing
        image = align_face(image)
        image = normalize_image(image)

        if DEBUG_MODE:
            print("[DEBUG] Image preprocessed for embedding extraction")

        # 🧠 Step 2: Extract embedding using DeepFace
        result = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            enforce_detection=True
        )

        embedding = np.array(result[0]["embedding"])

        if DEBUG_MODE:
            print(f"[DEBUG] Raw embedding shape: {embedding.shape}")

        # 🔥 Step 3: Normalize embedding
        embedding = normalize_embedding(embedding)

        if DEBUG_MODE:
            print("[DEBUG] Embedding normalized")

        return embedding

    except Exception as e:
        print(f"[WARNING] Face not detected: {e}")
        return None


def recognize_face(image, database_embeddings, labels, threshold=FACE_THRESHOLD):
    """
    Recognize a face and return structured result.

    Returns:
        {
            "label": str,
            "score": float,
            "status": "MATCH" | "UNKNOWN"
        }
    """

    if DEBUG_MODE:
        print("\n[DEBUG] --- Face Recognition Started ---")

    # 🧠 Step 1: Extract embedding
    query_embedding = extract_embedding(image)

    if query_embedding is None:
        if DEBUG_MODE:
            print("[DEBUG] No face detected in input")

        return {
            "label": "NO_FACE_DETECTED",
            "score": 0.0,
            "status": "UNKNOWN"
        }

    # 🔍 Step 2: Handle empty database
    if len(database_embeddings) == 0:
        if DEBUG_MODE:
            print("[DEBUG] Face database is empty")

        return {
            "label": "NO_DATABASE",
            "score": 0.0,
            "status": "UNKNOWN"
        }

    # 🔎 Step 3: Find best match
    best_label, best_score = find_best_match(
        query_embedding,
        database_embeddings,
        labels
    )

    if DEBUG_MODE:
        print(f"[DEBUG] Best match: {best_label}")
        print(f"[DEBUG] Similarity score: {best_score}")
        print(f"[DEBUG] Threshold: {threshold}")

    # ❌ Step 4: Fallback safety check
    if best_label is None:
        if DEBUG_MODE:
            print("[DEBUG] No valid match found")

        return {
            "label": "NO_DATABASE",
            "score": 0.0,
            "status": "UNKNOWN"
        }

    # ✅ Step 5: Decision
    status = "MATCH" if is_match(best_score, threshold) else "UNKNOWN"

    if DEBUG_MODE:
        print(f"[DEBUG] Final status: {status}")
        print("[DEBUG] --- Face Recognition Ended ---\n")

    # 📦 Step 6: Structured output
    return {
        "label": best_label,
        "score": float(best_score),
        "status": status
    }