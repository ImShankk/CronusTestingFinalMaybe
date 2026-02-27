import subprocess
import os
import winsound
import speech_recognition as sr


def listen() -> str:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.3)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("STT error:", e)
        return ""


def speak(text: str):
    p = subprocess.Popen(
        [
            r"C:\piper_windows_amd64\piper\piper.exe",
            "-m",
            r"C:\piper_windows_amd64\piper\en_US-lessac-low.onnx",
            "-f",
            "out.wav",
        ],
        stdin=subprocess.PIPE,
        text=True,
    )

    # Send text via STDIN (this is REQUIRED)
    p.stdin.write(text)
    p.stdin.close()
    p.wait()

    # Play audio (blocking is OK here)
    winsound.PlaySound("out.wav", winsound.SND_FILENAME)
