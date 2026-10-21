import numpy as np


def normalize_embedding(embedding):
    """
    Normalize embedding to unit length.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def find_best_match(query_embedding, database_embeddings, labels):
    """
    Vectorized similarity search using cosine similarity (dot product).

    Args:
        query_embedding (np.array): shape (D,)
        database_embeddings (np.array): shape (N, D)
        labels (np.array): shape (N,)

    Returns:
        best_label (str)
        best_score (float)
    """

    # 🔥 Normalize query embedding (CRITICAL)
    query_embedding = normalize_embedding(query_embedding)

    # ⚡ Compute all similarities at once
    scores = np.dot(database_embeddings, query_embedding)

    # 🎯 Get best match
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_label = labels[best_idx]

    return best_label, float(best_score)


def is_match(score, threshold=0.75):
    """
    Decide if similarity score passes the threshold.
    """
    return score >= threshold