import os
import numpy as np
import joblib

from config import (
    FACE_THRESHOLD,
    VOICE_THRESHOLD,
    FUSION_DECISION_THRESHOLD,
    FUSION_CONFIDENCE_FALLBACK,
    FACE_WEIGHT,
    VOICE_WEIGHT
)

# =========================
# SAFE MODEL LOADING
# =========================

MODEL_PATH = "data/embeddings/fusion_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("[INFO] Fusion model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to load fusion model: {e}")
        model = None
else:
    print("[WARNING] Fusion model not found. Running in DEMO MODE.")

# =========================
# HEURISTIC FALLBACK
# =========================

def heuristic_fallback(face_result, voice_result):
    """
    Backup rule-based system (config-driven).
    """

    face_score = face_result.get("score", 0.0)
    voice_score = voice_result.get("score", 0.0)

    if face_score > FACE_THRESHOLD and voice_score > VOICE_THRESHOLD:
        return "GRANTED"

    return "DENIED"

# =========================
# ML FUSION
# =========================

def ml_fusion(face_result, voice_result):
    """
    ML-based decision (Logistic Regression fusion model).
    """

    face_score = face_result.get("score", 0.0)
    voice_score = voice_result.get("score", 0.0)

    face_q = face_result.get("quality", face_score)
    voice_q = voice_result.get("quality", voice_score)

    features = np.array([[face_score, voice_score, face_q, voice_q]])

    # =========================
    # DEMO MODE (NO MODEL)
    # =========================
    if model is None:
        prob = (FACE_WEIGHT * face_score) + (VOICE_WEIGHT * voice_score)
        prob = max(0.0, min(1.0, prob))  # clamp for stability

        decision = "GRANTED" if prob > FUSION_DECISION_THRESHOLD else "DENIED"
        return decision, float(prob)

    # =========================
    # REAL ML MODE
    # =========================
    try:
        prob = model.predict_proba(features)[0][1]
        decision = "GRANTED" if prob > FUSION_DECISION_THRESHOLD else "DENIED"
        return decision, float(prob)

    except Exception as e:
        print(f"[WARNING] ML inference failed: {e}")
        fallback_decision = heuristic_fallback(face_result, voice_result)
        return fallback_decision, 0.5  # neutral confidence

# =========================
# ADAPTIVE FUSION
# =========================

def adaptive_fusion(face_result, voice_result):
    """
    Adaptive fallback fusion (quality-aware, modality-aware).
    """

    face_score = face_result.get("score", 0.0)
    voice_score = voice_result.get("score", 0.0)

    face_quality = face_result.get("quality", face_score)
    voice_quality = voice_result.get("quality", voice_score)

    face_available = face_score > FACE_THRESHOLD and face_quality > 0.5
    voice_available = voice_score > VOICE_THRESHOLD and voice_quality > 0.5

    if not face_available and not voice_available:
        return "DENIED", 0.0, "NO_INPUT"

    if face_available and not voice_available:
        return (
            "GRANTED" if face_score > FACE_THRESHOLD else "DENIED",
            float(face_score),
            "FACE_ONLY"
        )

    if voice_available and not face_available:
        return (
            "GRANTED" if voice_score > VOICE_THRESHOLD else "DENIED",
            float(voice_score),
            "VOICE_ONLY"
        )

    face_w = face_quality * FACE_WEIGHT
    voice_w = voice_quality * VOICE_WEIGHT

    total = face_w + voice_w
    face_w /= total
    voice_w /= total

    final_score = (face_w * face_score) + (voice_w * voice_score)

    decision = "GRANTED" if final_score > FUSION_DECISION_THRESHOLD else "DENIED"

    return decision, float(final_score), "ADAPTIVE_FUSION"

# =========================
# MAIN AUTH FUNCTION
# =========================

def authenticate(face_result, voice_result):
    """
    Hybrid ML + adaptive + fallback authentication system.
    """

    decision, confidence = ml_fusion(face_result, voice_result)

    # fallback if low confidence
    if confidence < FUSION_CONFIDENCE_FALLBACK:
        decision, confidence, mode = adaptive_fusion(face_result, voice_result)

        return {
            "final_decision": decision,
            "confidence": confidence,
            "mode": mode,
            "identity": face_result.get("identity", "UNKNOWN")
        }

    return {
        "final_decision": decision,
        "confidence": confidence,
        "mode": "ML" if model is not None else "SIMULATED_FUSION",
        "identity": face_result.get("identity", "UNKNOWN")
    }