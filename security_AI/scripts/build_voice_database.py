import os
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

DATASET_PATH = "data/voices"
OUTPUT_PATH = "data/embeddings"

# =========================
# PRETRAINED MODEL
# =========================

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)


# =========================
# AUDIO PROCESSING
# =========================

def load_audio(file_path):
    signal, fs = torchaudio.load(file_path)

    # Convert stereo → mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    return signal


def extract_embedding(audio_path):
    signal = load_audio(audio_path)

    with torch.no_grad():
        embedding = speaker_model.encode_batch(signal)

    return embedding.squeeze().numpy()


# =========================
# BUILD DATABASE (TRAINING STEP)
# =========================

def build_voice_database():
    embeddings = []
    labels = []

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing voice for {person_name}...")

        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)

            try:
                embedding = extract_embedding(file_path)

                embeddings.append(embedding)
                labels.append(person_name)

            except Exception as e:
                print(f"[WARNING] Skipping {file_path}: {e}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print(f"[INFO] Generated voice embeddings shape: {embeddings.shape}")

    return embeddings, labels


# =========================
# SAVE DATABASE
# =========================

def save_voice_database(embeddings, labels):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    np.save(os.path.join(OUTPUT_PATH, "voice_embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_PATH, "voice_labels.npy"), labels)

    print(f"[INFO] Saved {len(embeddings)} voice embeddings")


# =========================
# LOAD DATABASE (FOR STREAMLIT)
# =========================

def load_voice_database():
    embeddings = np.load(os.path.join(OUTPUT_PATH, "voice_embeddings.npy"))
    labels = np.load(os.path.join(OUTPUT_PATH, "voice_labels.npy"))

    print(f"[INFO] Loaded {len(embeddings)} voice embeddings")

    return embeddings, labels


# =========================
# RUN AS SCRIPT
# =========================

if __name__ == "__main__":
    embeddings, labels = build_voice_database()

    if len(embeddings) == 0:
        print("[ERROR] No voice embeddings created.")
    else:
        save_voice_database(embeddings, labels)