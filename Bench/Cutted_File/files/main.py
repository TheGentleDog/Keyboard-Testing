# =============================================================================
# main.py — Entry point for the Gaze-Based Keyboard
# =============================================================================

import os
import sys
from config import (
    FILIPINO_DATASET_FILE,
    ENGLISH_DATASET_FILE,
    NGRAM_CACHE_FILE,
)


# ─────────────────────────────────────────────
# STEP 1: Generate datasets if missing
# ─────────────────────────────────────────────
def _ensure_datasets():
    missing = []
    if not os.path.exists(FILIPINO_DATASET_FILE):
        missing.append(("Filipino", FILIPINO_DATASET_FILE, "generate_dataset"))
    if not os.path.exists(ENGLISH_DATASET_FILE):
        missing.append(("English", ENGLISH_DATASET_FILE, "generate_dataset_english"))

    if not missing:
        return  # both datasets already exist

    print("=" * 60)
    print("⚠  One or more datasets not found:")
    for label, path, _ in missing:
        print(f"   {label}: {path}")
    print("=" * 60)

    # Check if transformers is available before attempting generation
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("\n❌  'transformers' package not installed.")
        print("    Install it with:  pip install transformers torch")
        print("    Then re-run the keyboard.\n")
        sys.exit(1)

    for label, path, module_name in missing:
        print(f"\n🚀  Generating {label} dataset via RoBERTa...")
        print("    This runs ONCE and may take 5–15 minutes on CPU.")
        print("    The model (~500 MB) will be downloaded on first run.\n")
        try:
            module = __import__(module_name)
            module.generate(output_file=path)
        except Exception as e:
            print(f"❌  Failed to generate {label} dataset: {e}")
            sys.exit(1)

    # Delete stale n-gram cache so it rebuilds cleanly from both datasets
    if os.path.exists(NGRAM_CACHE_FILE):
        os.remove(NGRAM_CACHE_FILE)
        print(f"🗑  Removed stale cache: {NGRAM_CACHE_FILE}\n")


# ─────────────────────────────────────────────
# STEP 2: Boot model + launch UI
# ─────────────────────────────────────────────
def main():
    _ensure_datasets()

    # Import after datasets are guaranteed to exist
    from model import ngram_model
    from ui import FilipinoKeyboard

    print("=" * 60)
    print("GAZE-BASED DIGITAL KEYBOARD")
    print("Filipino + English Autocomplete")
    print("=" * 60)

    if not ngram_model.load_cache():
        print("\nNo cache found — building from datasets...")
        ngram_model.train_from_builtin()
        ngram_model.save_cache()

    print("\nLoading user learning...")
    ngram_model.load_user_learning()

    app = FilipinoKeyboard()
    app.mainloop()


if __name__ == "__main__":
    main()
