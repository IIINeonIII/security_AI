import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000


def record_voice(duration=3, filename="temp.wav"):
    """
    Record audio from microphone and save as WAV file.
    """

    print("[INFO] Recording started...")

    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()

    write(filename, SAMPLE_RATE, recording)

    print("[INFO] Recording finished.")

    return filename