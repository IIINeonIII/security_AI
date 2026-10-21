# =========================
# MODEL SETTINGS
# =========================

FACE_MODEL = "ArcFace"
VOICE_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


# =========================
# THRESHOLDS
# =========================

FACE_THRESHOLD = 0.75
VOICE_THRESHOLD = 0.75

FUSION_DECISION_THRESHOLD = 0.60
FUSION_CONFIDENCE_FALLBACK = 0.60


# =========================
# FUSION WEIGHTS (heuristic fallback)
# =========================

FACE_WEIGHT = 0.6
VOICE_WEIGHT = 0.4


# =========================
# AUDIO SETTINGS
# =========================

SAMPLE_RATE = 16000
VOICE_MIN_DURATION = 1.0


# =========================
# FACE SETTINGS
# =========================

MIN_FACE_SIZE = 80
BLUR_THRESHOLD = 100


# =========================
# PATHS
# =========================

DATA_PATH = "data"
FACE_DATA_PATH = "data/faces"
VOICE_DATA_PATH = "data/voices"

EMBEDDING_PATH = "data/embeddings"

FACE_EMBEDDINGS_FILE = "data/embeddings/face_embeddings.npy"
FACE_LABELS_FILE = "data/embeddings/face_labels.npy"

VOICE_EMBEDDINGS_FILE = "data/embeddings/voice_embeddings.npy"
VOICE_LABELS_FILE = "data/embeddings/voice_labels.npy"

FUSION_MODEL_PATH = "data/embeddings/fusion_model.pkl"

LOG_PATH = "logs/access_log.txt"


# =========================
# SYSTEM SETTINGS
# =========================

CAMERA_INDEX = 0
AUTO_CAPTURE_FACE = True
AUTO_CAPTURE_VOICE = False  # later when mic is integrated


# =========================
# DEBUG SETTINGS
# =========================

DEBUG_MODE = True
PRINT_EMBEDDINGS_SHAPE = False