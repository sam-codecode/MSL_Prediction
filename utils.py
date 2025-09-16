# utils.py
import pyttsx3

def speak_text(text):
    """
    Speak the full text using pyttsx3 (Jarvis-style wrapper).
    This is blocking (pyttsx3.runAndWait()) — that's intentional so the
    Read button reads the entire current_text at once.
    """
    try:
        if not text:
            return
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)     # reading speed
        engine.setProperty('volume', 1.0)   # max volume
        voices = engine.getProperty('voices')
        # pick voice index safely (fall back to 0)
        try:
            engine.setProperty('voice', voices[1].id)  # prefer female if available
        except Exception:
            engine.setProperty('voice', voices[0].id)

        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[MSL Voice Error] {e}")
