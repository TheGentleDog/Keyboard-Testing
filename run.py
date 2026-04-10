# =============================================================================
# run.py — Single entry point for Gaze-Based Filipino Keyboard
#
# Flow:
#   1. Gaze tracker starts (OpenCV, fullscreen calibration)
#   2. Once calibration is done, mouse control is enabled automatically
#   3. Tkinter keyboard launches on the main thread
#   4. Gaze tracker keeps running in background, moving the mouse cursor
#   5. Closing the keyboard also stops the gaze tracker
# =============================================================================

import sys
import os
import threading

# ── Make keyboard modules importable ─────────────────────────────────────────
KEYBOARD_DIR = os.path.join(os.path.dirname(__file__), "Bench", "Cutted_File", "files")
sys.path.insert(0, KEYBOARD_DIR)

# ── Import gaze tracker ───────────────────────────────────────────────────────
from gaze_tracker2 import GazeTrackerApp
import argparse

# ── Import keyboard bootstrap ─────────────────────────────────────────────────
from model import ngram_model, get_context_words
from generate_flores_rules import generate_if_missing as _ensure_flores
from config import (
    FILIPINO_DATASET_FILE,
    ENGLISH_DATASET_FILE,
    NGRAM_CACHE_FILE,
)


def _ensure_datasets():
    missing = []
    if not os.path.exists(FILIPINO_DATASET_FILE):
        missing.append(("Filipino", FILIPINO_DATASET_FILE, "generate_dataset"))
    if not os.path.exists(ENGLISH_DATASET_FILE):
        missing.append(("English", ENGLISH_DATASET_FILE, "generate_dataset_english"))
    if not missing:
        return
    try:
        import transformers  # noqa
    except ImportError:
        print("❌  'transformers' not installed. pip install transformers torch")
        sys.exit(1)
    for label, path, module_name in missing:
        print(f"🚀  Generating {label} dataset...")
        module = __import__(module_name)
        module.generate(output_file=path)
    if os.path.exists(NGRAM_CACHE_FILE):
        os.remove(NGRAM_CACHE_FILE)


def main():
    ap = argparse.ArgumentParser(description="Gaze-Based Filipino Keyboard")
    ap.add_argument("--camera",  type=int,   default=0)
    ap.add_argument("--points",  type=int,   default=9,  choices=[5, 9, 16, 25])
    ap.add_argument("--samples", type=int,   default=60)
    ap.add_argument("--ema",     type=float, default=0.15)
    ap.add_argument("--pnoise",  type=float, default=1e-3)
    ap.add_argument("--mnoise",  type=float, default=12.0)
    args = ap.parse_args()

    # ── Pre-load datasets & model ─────────────────────────────────────────────
    _ensure_datasets()
    _ensure_flores()

    if not ngram_model.load_cache():
        print("Building n-gram model from datasets...")
        ngram_model.train_from_builtin()
        ngram_model.save_cache()
    ngram_model.load_user_learning()

    # ── Events for coordination ───────────────────────────────────────────────
    calib_done  = threading.Event()
    stop_gaze   = threading.Event()

    # ── Build gaze tracker ────────────────────────────────────────────────────
    tracker = GazeTrackerApp(
        camera_id  = args.camera,
        num_points = args.points,
        spp        = args.samples,
        ema_alpha  = args.ema,
        pnoise     = args.pnoise,
        mnoise     = args.mnoise,
    )

    # ── Run gaze tracker in background thread ─────────────────────────────────
    gaze_thread = threading.Thread(
        target=tracker.run,
        kwargs={"calib_done_event": calib_done, "stop_event": stop_gaze},
        daemon=True,
    )
    gaze_thread.start()

    print("=" * 60)
    print("  GAZE-BASED DIGITAL KEYBOARD")
    print("  Waiting for calibration to complete...")
    print("=" * 60)

    # Block until calibration is done
    calib_done.wait()
    print("\n✓ Calibration complete — launching keyboard...\n")

    # ── Launch Tkinter keyboard on main thread ────────────────────────────────
    from ui import FilipinoKeyboard

    app = FilipinoKeyboard()

    def on_close():
        stop_gaze.set()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()

    # Cleanup
    stop_gaze.set()
    print("[Info] Application closed.")


if __name__ == "__main__":
    main()
