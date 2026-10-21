import os
from datetime import datetime

from config import LOG_PATH


def ensure_log_file():
    """
    Ensure logs directory and file exist.
    """

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("=== ACCESS LOG START ===\n\n")


def log_access_attempt(user_label, face_result, voice_result, final_decision):
    """
    Log one authentication attempt.

    Parameters:
        user_label (str): predicted or claimed identity
        face_result (dict): output from face inference
        voice_result (dict): output from voice inference
        final_decision (str): GRANTED or DENIED
    """

    ensure_log_file()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
----------------------------------------
TIME: {timestamp}

USER: {user_label}

FACE:
    label: {face_result.get("label")}
    score: {face_result.get("score")}
    status: {face_result.get("status")}

VOICE:
    label: {voice_result.get("label")}
    score: {voice_result.get("score")}
    status: {voice_result.get("status")}

FINAL DECISION: {final_decision}
----------------------------------------
"""

    with open(LOG_PATH, "a") as f:
        f.write(log_entry)

    print("[LOG] Access attempt recorded.")