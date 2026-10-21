import random
import numpy as np
import cv2
import os
from datetime import datetime


# =========================
# USER POOL (REALISTIC DEMO IDS)
# =========================

USER_POOL = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve"
]


# =========================
# SIMULATED BIOMETRIC INPUT
# =========================

def simulate_inputs(user_type):

    if user_type == "recognized":
        name = random.choice(USER_POOL)

        return (
            {
                "score": round(random.uniform(0.80, 0.98), 2),
                "quality": round(random.uniform(0.80, 1.0), 2),
                "identity": name
            },
            {
                "score": round(random.uniform(0.80, 0.98), 2),
                "quality": round(random.uniform(0.80, 1.0), 2),
                "identity": name
            }
        )

    else:
        return (
            {
                "score": round(random.uniform(0.20, 0.50), 2),
                "quality": round(random.uniform(0.40, 0.70), 2),
                "identity": "UNKNOWN"
            },
            {
                "score": round(random.uniform(0.20, 0.50), 2),
                "quality": round(random.uniform(0.40, 0.70), 2),
                "identity": "UNKNOWN"
            }
        )


# =========================
# FAKE LIVE CAMERA FRAME
# =========================

def fake_frame():
    return np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)


# =========================
# SAVE EVIDENCE FRAME
# =========================

def save_frame(frame, identity):
    os.makedirs("captures", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captures/{identity}_{timestamp}.jpg"

    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return filename