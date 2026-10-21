import numpy as np
from deepface import DeepFace

MODEL_NAME = "ArcFace"


def encode_face(image):
    """
    Convert face image → embedding vector.
    """

    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        enforce_detection=True
    )

    embedding = np.array(result[0]["embedding"])

    # normalize
    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm

    return embedding