# =============================================================================
# tts.py — Cross-platform Text-to-Speech (Filipino + English)
#
# Primary : edge-tts  (online, high quality, supports fil-PH and en-US)
# Fallback: pyttsx3   (offline, English only)
#
# Language detection: simple Filipino word heuristic
# Audio playback    : pygame (cross-platform, macOS + Windows)
# All speech runs in a background thread so the UI stays responsive.
# =============================================================================

import asyncio
import os
import tempfile
import threading

# ── Filipino vocabulary for language detection ─────────────────────────────────
_FILIPINO_WORDS = {
    "ang", "mga", "ng", "sa", "na", "at", "ay", "ko", "mo", "siya",
    "kami", "tayo", "namin", "ninyo", "nila", "ako", "ikaw", "sila",
    "ito", "iyan", "iyon", "hindi", "oo", "po", "nga", "lang", "naman",
    "dito", "doon", "para", "kung", "pero", "kaya", "may", "mayroon",
    "wala", "yung", "yun", "kasi", "talaga", "nag", "mag", "pag",
    "din", "rin", "daw", "raw", "ba", "pa", "na", "ho", "ano", "sino",
    "bakit", "kailan", "saan", "paano", "alin", "kanino", "lahat",
    "bawat", "isa", "dalawa", "tatlo", "apat", "lima", "kamusta",
    "kumain", "uminom", "matulog", "maglaro", "pumunta", "bukas",
    "kinabukasan", "kahapon", "ngayon", "mamaya", "maganda", "mabuti",
    "masama", "malaki", "maliit", "bago", "luma", "puso", "bahay",
    "pamilya", "kaibigan", "gabi", "umaga", "hapon", "tanghali",
    "gutom", "pagkain", "tubig", "sakit", "gamot", "doktor", "ospital",
}

_EDGE_VOICES = {
    "fil": "fil-PH-BlessicaNeural",   # Filipino female
    "en":  "en-US-JennyNeural",       # English female
}

# ── Language detection ─────────────────────────────────────────────────────────
def _detect_language(text: str) -> str:
    words = text.lower().split()
    fil_count = sum(1 for w in words if w in _FILIPINO_WORDS)
    ratio = fil_count / max(len(words), 1)
    return "fil" if ratio >= 0.25 else "en"


# ── edge-tts (online) ──────────────────────────────────────────────────────────
async def _edge_speak_async(text: str, voice: str, tmp_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)


def _play_with_pygame(path: str):
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"⚠ pygame playback error: {e}")


def _speak_edge(text: str):
    lang  = _detect_language(text)
    voice = _EDGE_VOICES[lang]
    print(f"🔊 edge-tts [{lang}] voice={voice}: {text}")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        asyncio.run(_edge_speak_async(text, voice, tmp.name))
        _play_with_pygame(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ── pyttsx3 fallback (offline, English only) ──────────────────────────────────
def _speak_pyttsx3(text: str):
    print(f"🔊 pyttsx3 (offline fallback): {text}")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"⚠ pyttsx3 error: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────
def speak(text: str):
    """
    Speak `text` in a background thread.
    Tries edge-tts first; falls back to pyttsx3 if unavailable or offline.
    Returns immediately — UI stays responsive while audio plays.
    """
    text = text.strip()
    if not text:
        return

    def _worker():
        try:
            import edge_tts   # noqa: F401 — just check availability
            _speak_edge(text)
        except ImportError:
            print("⚠ edge-tts not installed — falling back to pyttsx3")
            _speak_pyttsx3(text)
        except Exception as e:
            print(f"⚠ edge-tts failed ({e}) — falling back to pyttsx3")
            _speak_pyttsx3(text)

    threading.Thread(target=_worker, daemon=True).start()
