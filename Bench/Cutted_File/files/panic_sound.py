# =============================================================================
# panic_sound.py — Emergency alarm sound (no extra dependencies)
#
# Generates a two-tone alternating alarm (whoop-whoop) using stdlib wave
# module, then plays it via pygame (already installed for TTS).
# Runs in a background thread so the UI stays responsive.
# The alarm loops until stop() is called.
# =============================================================================

import array
import math
import os
import tempfile
import threading
import wave

_alarm_thread   = None
_stop_event     = threading.Event()

# ── Waveform generation ────────────────────────────────────────────────────────
def _generate_tone(freq: float, duration_ms: int, sample_rate: int = 44100) -> array.array:
    n_samples = int(sample_rate * duration_ms / 1000)
    samples   = array.array("h")
    for i in range(n_samples):
        t       = i / sample_rate
        # Sine wave with slight fade-in/out to avoid clicks
        fade    = min(i, n_samples - i, 200) / 200.0
        value   = int(32767 * fade * math.sin(2 * math.pi * freq * t))
        samples.append(value)
    return samples


def _write_wav(samples: array.array, path: str, sample_rate: int = 44100):
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())


def _build_alarm_wav() -> str:
    """Build a whoop-whoop alarm wav and return the temp file path."""
    rate     = 44100
    hi, lo   = 1200, 800
    beep_ms  = 300
    hi_tone  = _generate_tone(hi, beep_ms, rate)
    lo_tone  = _generate_tone(lo, beep_ms, rate)
    pattern  = hi_tone + lo_tone + hi_tone + lo_tone  # one cycle ~1.2 s

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _write_wav(pattern, tmp.name, rate)
    return tmp.name


# ── Playback ───────────────────────────────────────────────────────────────────
def _alarm_loop(wav_path: str):
    try:
        import pygame
        pygame.mixer.init(frequency=44100)
        sound = pygame.mixer.Sound(wav_path)
        while not _stop_event.is_set():
            sound.play()
            # Wait for this playback to finish or stop signal
            ms = int(sound.get_length() * 1000)
            _stop_event.wait(timeout=ms / 1000.0)
        sound.stop()
        pygame.mixer.quit()
    except Exception as e:
        print(f"⚠ Alarm playback error: {e}")
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


# ── Public API ─────────────────────────────────────────────────────────────────
def start():
    """Start the alarm loop in a background thread."""
    global _alarm_thread
    if _alarm_thread and _alarm_thread.is_alive():
        return   # already running
    _stop_event.clear()
    wav_path      = _build_alarm_wav()
    _alarm_thread = threading.Thread(target=_alarm_loop, args=(wav_path,), daemon=True)
    _alarm_thread.start()


def stop():
    """Stop the alarm."""
    _stop_event.set()
