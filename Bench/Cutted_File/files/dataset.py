# =============================================================================
# dataset.py — Load Filipino vocabulary, shortcuts, and corpus from JSON
# =============================================================================

import json
from config import DATASET_FILE


def load_dataset():
    """Load Filipino vocabulary and corpus from JSON file."""
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✓ Loaded dataset: {data['metadata']['description']}")
        print(f"  Version: {data['metadata']['version']}")
        print(f"  Words: {data['metadata']['total_words']}")
        print(f"  Phrases: {data['metadata']['total_phrases']}")

        words = []
        for category, word_list in data['vocabulary'].items():
            words.extend(word_list)

        shortcuts = data['shortcuts']
        corpus    = data['communication_corpus']

        return words, shortcuts, corpus

    except FileNotFoundError:
        print(f"⚠ Warning: {DATASET_FILE} not found!")
        print("⚠ Using minimal fallback vocabulary...")
        return _get_fallback_data()
    except Exception as e:
        print(f"⚠ Error loading dataset: {e}")
        print("⚠ Using minimal fallback vocabulary...")
        return _get_fallback_data()


def _get_fallback_data():
    """Minimal fallback if JSON file is missing."""
    words = [
        "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
        "kumain", "uminom", "matulog", "maganda", "masaya",
        "oo", "hindi", "salamat", "sorry", "nalang", "lang",
        "naman", "kasi", "kung", "dito", "doon", "pwede",
        "sana", "talaga", "para", "wala", "may", "yung"
    ]
    shortcuts = {
        "lng": "lang",   "nlng": "nalang", "nmn": "naman",
        "ksi": "kasi",   "kng": "kung",    "khit": "kahit",
        "d2":  "dito",   "dn":  "doon",    "pde": "pwede",   "pwd": "pwede",
        "sna": "sana",   "tlg": "talaga",  "tlga": "talaga",
        "sya": "siya",   "xa":  "siya",    "aq":  "ako",     "q":   "ako",
        "nde": "hindi",  "hnd": "hindi",   "pra": "para",
        "wla": "wala",   "my":  "may",     "ung": "yung",    "un":  "yun",
        "dba": "diba",   "db":  "diba",    "bat": "bakit",   "bkt": "bakit"
    }
    corpus = ["kumusta ka", "mabuti naman", "salamat", "ok lang"]
    return words, shortcuts, corpus


# Load on import so other modules can share the same data
FILIPINO_WORDS, FILIPINO_SHORTCUTS, COMMUNICATION_CORPUS = load_dataset()
