# =============================================================================
# main.py — Entry point for the Filipino Gaze-Based Keyboard
# =============================================================================

import os
import sys
from config import DATASET_FILE, NGRAM_CACHE_FILE


# ─────────────────────────────────────────────
# STEP 1: Generate dataset if missing
# ─────────────────────────────────────────────
def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return  # already exists, nothing to do

    print("=" * 60)
    print("⚠  No dataset found.")
    print(f"   Expected: {DATASET_FILE}")
    print("=" * 60)

    # Check if transformers is available
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("\n❌  'transformers' package not installed.")
        print("    Install it with:  pip install transformers torch")
        print("    Then re-run the keyboard.\n")
        sys.exit(1)

    print("\n🚀  Running RoBERTa-Tagalog dataset generator...")
    print("    This runs ONCE and may take 5–15 minutes on CPU.")
    print("    The model (~500 MB) will be downloaded on first run.\n")

    from generate_dataset import generate
    generate(output_file=DATASET_FILE)

    # Delete stale n-gram cache so it rebuilds from the new dataset
    if os.path.exists(NGRAM_CACHE_FILE):
        os.remove(NGRAM_CACHE_FILE)
        print(f"🗑  Removed stale cache: {NGRAM_CACHE_FILE}\n")


# ─────────────────────────────────────────────
# STEP 2: Boot model + launch UI
# ─────────────────────────────────────────────
def main():
    _ensure_dataset()

    # Import after dataset is guaranteed to exist
    from model import ngram_model
    from ui import FilipinoKeyboard

    print("=" * 60)
    print("FILIPINO KEYBOARD - LIVE AUTOCOMPLETE VERSION")
    print("Gaze-Based Digital Keyboard")
    print("=" * 60)

    if not ngram_model.load_cache():
        print("\nNo cache found. Building from built-in vocabulary...")
        ngram_model.train_from_builtin()
        ngram_model.save_cache()

    print("\nLoading user learning...")
    ngram_model.load_user_learning()

    app = FilipinoKeyboard()
    app.mainloop()


if __name__ == "__main__":
    main()
