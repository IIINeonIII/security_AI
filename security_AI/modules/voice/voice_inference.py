import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from utils.distance import find_best_match, is_match
from utils.preprocessing import preprocess_audio
from config import VOICE_MODEL, VOICE_THRESHOLD, DEBUG_MODE

MODEL_NAME = VOICE_MODEL

speaker_model = EncoderClassifier.from_hparams(
    source=MODEL_NAME
)


def normalize_embedding(embedding):
    """
    Normalize embedding to unit length.
    Ensures cosine similarity works correctly.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def extract_embedding(audio_path):
    """
    Extract speaker embedding with preprocessing.

    Steps:
        1. Load + trim silence + reduce noise
        2. Pass through ECAPA-TDNN model
        3. Return embedding vector
    """

    try:
        # 🎤 Step 1: Preprocess audio
        signal = preprocess_audio(audio_path)

        if DEBUG_MODE:
            print("[DEBUG] Audio preprocessed for embedding extraction")

        # 🎤 Step 2: Extract embedding
        with torch.no_grad():
            embedding = speaker_model.encode_batch(signal)

        embedding = np.array(embedding.squeeze().cpu())

        if DEBUG_MODE:
            print(f"[DEBUG] Raw voice embedding shape: {embedding.shape}")

        # 🔥 Step 3: Normalize embedding
        embedding = normalize_embedding(embedding)

        if DEBUG_MODE:
            print("[DEBUG] Voice embedding normalized")

        return embedding

    except Exception as e:
        print(f"[WARNING] Voice not processed: {e}")
        return None


def recognize_voice(audio_path, database_embeddings, labels, threshold=VOICE_THRESHOLD):
    """
    Recognize speaker and return structured result.

    Returns:
        {
            "label": str,
            "score": float,
            "status": "MATCH" | "UNKNOWN"
        }
    """

    if DEBUG_MODE:
        print("\n[DEBUG] --- Voice Recognition Started ---")

    # 🎤 Step 1: Extract embedding
    query_embedding = extract_embedding(audio_path)

    if query_embedding is None:
        if DEBUG_MODE:
            print("[DEBUG] No voice detected in input")

        return {
            "label": "NO_VOICE_DETECTED",
            "score": 0.0,
            "status": "UNKNOWN"
        }

    # 🔍 Step 2: Handle empty database
    if len(database_embeddings) == 0:
        if DEBUG_MODE:
            print("[DEBUG] Voice database is empty")

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

    # ❌ Step 4: Decision
    status = "MATCH" if is_match(best_score, threshold) else "UNKNOWN"

    if DEBUG_MODE:
        print(f"[DEBUG] Final status: {status}")
        print("[DEBUG] --- Voice Recognition Ended ---\n")

    # 📦 Step 5: Structured output
    return {
        "label": best_label,
        "score": float(best_score),
        "status": status
    }