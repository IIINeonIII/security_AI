import cv2
import numpy as np
import torch
import torchaudio
import librosa


# =========================
# 👁️ FACE PREPROCESSING
# =========================

def align_face(image, face_detector=None):
    """
    Basic face alignment using eye positions.
    If no detector is provided, returns resized image.
    """

    try:
        # If you don't pass a detector, fallback to simple resize
        if face_detector is None:
            return cv2.resize(image, (224, 224))

        # Detect facial landmarks (eyes)
        landmarks = face_detector(image)

        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]

        # Compute angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate image to align eyes horizontally
        eyes_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2
        )

        rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)
        aligned = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

        return cv2.resize(aligned, (224, 224))

    except Exception:
        # fallback if alignment fails
        return cv2.resize(image, (224, 224))


def normalize_image(image):
    """
    Normalize pixel values to [0,1].
    """
    return image / 255.0


# =========================
# 🎤 AUDIO PREPROCESSING
# =========================

def trim_silence(audio, sr, top_db=25):
    """
    Remove silence from audio using librosa.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def reduce_noise(audio, sr):
    """
    Lightweight noise reduction using spectral gating (simple version).
    """
    # Convert to float numpy
    audio = audio.astype(np.float32)

    # Estimate noise profile (first 0.5 sec assumption)
    noise_sample = audio[:int(0.5 * sr)]
    noise_power = np.mean(noise_sample ** 2)

    # Simple noise gate
    threshold = noise_power * 1.5
    cleaned = np.where(audio ** 2 < threshold, 0, audio)

    return cleaned


def preprocess_audio(audio_path, target_sr=16000):
    """
    Full audio preprocessing pipeline:
    - load
    - resample
    - trim silence
    - reduce noise
    """

    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)

    # Trim silence
    audio = trim_silence(audio, sr)

    # Reduce noise
    audio = reduce_noise(audio, sr)

    # Normalize amplitude
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return torch.tensor(audio).unsqueeze(0)