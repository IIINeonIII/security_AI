import streamlit as st
import pandas as pd
from datetime import datetime

from modules.auth.authentication import authenticate
from demo_mode import simulate_inputs, fake_frame, save_frame


# =========================
# SIMPLE LOGIN SYSTEM
# =========================

APP_PASSWORD = "admin123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.title("🔐 Secure Access Required")

    password = st.text_input("Enter password", type="password")

    if st.button("Login"):

        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()


# =========================
# CAMERA OFFLINE FRAME
# =========================

def camera_offline_frame():
    import numpy as np
    import cv2

    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        "CAMERA OFFLINE",
        (50, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="Security Control Room", layout="wide")

st.title("🛡️ SECURITY OPERATIONS CENTER")
st.markdown("---")


# =========================
# SESSION STATE
# =========================

if "logs" not in st.session_state:
    st.session_state.logs = []

if "system_active" not in st.session_state:
    st.session_state.system_active = False

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("⚙️ SYSTEM CONTROL")

    if st.button("🟢 Start Monitoring"):
        st.session_state.system_active = True

    if st.button("🔴 Stop Monitoring"):
        st.session_state.system_active = False

    st.markdown("---")
    st.header("🧪 SIMULATION")

    simulate_user = st.button("👤 Authorized User")
    simulate_intruder = st.button("🚨 Intruder")

    # =========================
    # CLEAR LOGS (WITH CONFIRMATION)
    # =========================

    st.markdown("---")
    st.header("🧹 Maintenance")

    if st.button("🗑️ Clear Logs & History"):
        st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("Are you sure you want to delete all logs?")

        colA, colB = st.columns(2)

        with colA:
            if st.button("✅ Yes, clear"):
                timestamp = datetime.now().strftime("%H:%M:%S")

                st.session_state.logs = []
                st.session_state.last_frame = None
                st.session_state.confirm_clear = False

                st.success(f"Logs cleared at {timestamp}")
                st.rerun()

        with colB:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear = False


# =========================
# SYSTEM STATUS
# =========================

if st.session_state.system_active:
    st.success("SYSTEM STATUS: ACTIVE")
else:
    st.error("SYSTEM STATUS: OFFLINE")


# =========================
# LAYOUT
# =========================

col1, col2, col3 = st.columns([2, 2, 1])


# =========================
# LIVE FEED
# =========================

with col1:

    st.markdown("### 🎥 LIVE FEED")

    frame_box = st.empty()

    if st.session_state.system_active:
        frame = fake_frame()
        st.session_state.last_frame = frame
    else:
        frame = camera_offline_frame()
        st.session_state.last_frame = None

    frame_box.image(frame, channels="RGB")


# =========================
# AUTH FUNCTION
# =========================

def run_auth(mode):

    face_result, voice_result = simulate_inputs(mode)
    auth_result = authenticate(face_result, voice_result)

    identity = auth_result.get("identity", "UNKNOWN")

    if st.session_state.last_frame is not None:
        frame = st.session_state.last_frame
    else:
        frame = camera_offline_frame()

    saved_path = save_frame(frame, identity)

    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "identity": identity,
        "decision": auth_result["final_decision"],
        "confidence": auth_result["confidence"],
        "mode": auth_result["mode"],
        "face_score": face_result["score"],
        "voice_score": voice_result["score"],
        "evidence": saved_path
    }

    st.session_state.logs.append(log_entry)

    return auth_result, face_result, voice_result, saved_path


# =========================
# CONTROL PANEL
# =========================

with col2:

    st.markdown("### 🎮 AUTHENTICATION")

    triggered = None

    if simulate_user:
        triggered = "recognized"
    elif simulate_intruder:
        triggered = "unrecognized"

    if triggered and st.session_state.system_active:

        auth_result, face_result, voice_result, saved_path = run_auth(triggered)

        st.markdown("## 🔐 RESULT")

        if auth_result["final_decision"] == "GRANTED":
            st.success("ACCESS GRANTED")
        else:
            st.error("ACCESS DENIED")

        st.markdown("## 👤 IDENTITY")

        if auth_result["identity"] == "UNKNOWN":
            st.error("Unrecognized Individual")
        else:
            st.success(f"Recognized: {auth_result['identity']}")

        st.markdown("## 📊 BIOMETRIC METRICS")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Face Match", f"{face_result['score']:.2f}")
            if face_result["score"] < 0.75:
                st.warning("Below threshold")

        with colB:
            st.metric("Voice Match", f"{voice_result['score']:.2f}")
            if voice_result["score"] < 0.70:
                st.warning("Below threshold")

        with colC:
            st.metric("Fusion Confidence", f"{auth_result['confidence']:.2f}")

        st.info(f"Mode: {auth_result['mode']}")
        st.success(f"Evidence saved: {saved_path}")

    elif triggered and not st.session_state.system_active:
        st.warning("System is OFFLINE. Start monitoring first.")


# =========================
# ALERT PANEL
# =========================

with col3:

    st.markdown("### 🚨 ALERTS")

    if st.session_state.logs:

        latest = st.session_state.logs[-1]

        last_intrusion = None
        for log in reversed(st.session_state.logs):
            if log["identity"] == "UNKNOWN":
                last_intrusion = log
                break

        if latest["identity"] == "UNKNOWN":
            st.error("🚨 ACTIVE ALERT")
            st.write("Unauthorized access detected")
            st.write(f"Time: {latest['time']}")
            st.write(f"Confidence: {latest['confidence']:.2f}")

        elif last_intrusion:
            st.warning("⚠️ PRIOR SECURITY EVENT")
            st.write(f"At {last_intrusion['time']}, a possible intrusion attempt was detected.")
            st.write("System currently stable.")

        else:
            st.success("✅ No security threats detected")

    else:
        st.info("No events yet")


# =========================
# EVENT LOG
# =========================

st.markdown("---")
st.markdown("### 📜 EVENT LOG")


def highlight_row(row):
    if row["identity"] == "UNKNOWN":
        return ["background-color: #ff4b4b"] * len(row)
    return [""] * len(row)


if st.session_state.logs:

    df = pd.DataFrame(st.session_state.logs)

    st.dataframe(
        df.sort_values(by="time", ascending=False).style.apply(highlight_row, axis=1),
        use_container_width=True
    )

else:
    st.info("No logs recorded")


# =========================
# RECENT ACTIVITY
# =========================

st.markdown("###  RECENT ACTIVITY")

for log in reversed(st.session_state.logs[-5:]):
    if log["identity"] == "UNKNOWN":
        st.error(f"[{log['time']}] Intrusion detected")
    else:
        st.success(f"[{log['time']}] Access granted: {log['identity']}")


# =========================
# REPLAY
# =========================

st.markdown("---")

if st.button("🔁 Replay Last Event"):
    if st.session_state.logs:
        last = st.session_state.logs[-1]
        st.image(last["evidence"], caption="Last captured evidence")


# =========================
# INCIDENT HISTORY
# =========================

st.markdown("---")
st.markdown("###  INCIDENT HISTORY")

intrusions = [log for log in st.session_state.logs if log["identity"] == "UNKNOWN"]
intrusion_count = len(intrusions)

if intrusion_count > 0:
    st.error(f"Total Intrusion Attempts: {intrusion_count}")
else:
    st.success("No intrusion attempts recorded")

if intrusions:

    for i, log in enumerate(reversed(intrusions), 1):

        with st.expander(f"⚠️ Attempt #{intrusion_count - i + 1} — {log['time']}"):

            st.write(f"**Time:** {log['time']}")
            st.write(f"**Confidence:** {log['confidence']:.2f}")
            st.write(f"**Mode:** {log['mode']}")

            st.image(log["evidence"], caption="Captured Evidence", width=250)

else:
    st.info("No incidents to display")
