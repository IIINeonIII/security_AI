import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

model = EncoderClassifier.from_hparams(source=MODEL_NAME)


def encode_voice(audio_signal):
    """
    Convert audio signal → embedding vector.
    """

    with torch.no_grad():
        embedding = model.encode_batch(audio_signal)

    embedding = np.array(embedding.squeeze().cpu())

    return embedding