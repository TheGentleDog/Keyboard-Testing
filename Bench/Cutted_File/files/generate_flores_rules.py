"""
generate_flores_rules.py — Generates flores_rules.json from Flores et al. (2022)

Fetches the original training data directly from the published GitHub repository:
  ljyflores/efficient-spelling-normalization-filipino

This ensures the rules are derived from the original source dataset as cited
in the study, which can be defended to a panel as a primary reference.

Can be run standalone:
    python generate_flores_rules.py

Or called from main.py on first startup via generate_if_missing().
"""

import json
import os
import csv
import urllib.request
from collections import defaultdict
from datetime import date

OUTPUT_FILE = "flores_rules.json"

# ─────────────────────────────────────────────
# SOURCE URLS — Flores et al. (2022)
# ljyflores/efficient-spelling-normalization-filipino
# ─────────────────────────────────────────────
GITHUB_URLS = [
    "https://raw.githubusercontent.com/ljyflores/efficient-spelling-normalization-filipino/main/data/train_words.csv",
    "https://raw.githubusercontent.com/ljyflores/efficient-spelling-normalization-filipino/main/data/test_words.csv",
]


# ─────────────────────────────────────────────
# FETCH FROM GITHUB
# ─────────────────────────────────────────────
def _fetch_pairs():
    """
    Download word pairs directly from the Flores et al. GitHub repository.
    Returns a list of (abbreviated, correct) tuples.
    """
    pairs = []
    for url in GITHUB_URLS:
        print(f"  Fetching: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                lines = response.read().decode("utf-8").splitlines()
                reader = csv.reader(lines)
                for row in reader:
                    if len(row) >= 2:
                        short = row[0].strip().lower()
                        full  = row[1].strip().lower()
                        if short and full and short != full:
                            pairs.append((short, full))
            print(f"  ✓ Fetched {len(pairs)} pairs so far")
        except Exception as e:
            print(f"  ⚠  Could not fetch {url}: {e}")

    return pairs


# ─────────────────────────────────────────────
# RULE EXTRACTION
# ─────────────────────────────────────────────
def _extract_rules(pairs):
    """
    Derive substitution rules from the word pairs using character-level
    alignment — mirrors the rule extraction described in Flores et al. (2022).
    """
    word_rules = {}
    char_rules  = defaultdict(set)

    for short, full in pairs:
        # whole-word rule — direct abbreviation mapping
        word_rules[short] = full

        # character-level alignment
        i, j = 0, 0
        while i < len(short) and j < len(full):
            if short[i] == full[j]:
                i += 1; j += 1; continue
            for si in range(1, 4):
                for fi in range(1, 4):
                    s_sub = short[i:i+si]
                    f_sub = full[j:j+fi]
                    if (s_sub and f_sub and s_sub != f_sub
                            and all(c.isalnum() for c in s_sub)
                            and all(c.isalpha() for c in f_sub)):
                        char_rules[s_sub].add(f_sub)
            i += 1; j += 1

    # keep only compact, reliable char rules
    clean_char = {
        k: list(v)
        for k, v in char_rules.items()
        if len(k) <= 2 and len(v) <= 10
    }

    return word_rules, clean_char


# ─────────────────────────────────────────────
# MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────
def generate(output_file: str = OUTPUT_FILE):
    print("🌐 Fetching Flores et al. (2022) dataset from GitHub...")
    pairs = _fetch_pairs()

    if not pairs:
        print("❌  Failed to fetch any pairs from GitHub.")
        print("    Check your internet connection and try again.")
        return

    print(f"\n✓ Total pairs fetched : {len(pairs)}")
    print("🔧 Extracting substitution rules...")

    word_rules, char_rules = _extract_rules(pairs)

    print(f"✓ Word-level rules    : {len(word_rules)}")
    print(f"✓ Char-level rules    : {len(char_rules)}")

    dataset = {
        "metadata": {
            "source":       "Flores et al. (2022) — efficient-spelling-normalization-filipino",
            "github":       "https://github.com/ljyflores/efficient-spelling-normalization-filipino",
            "pairs_used":   len(pairs),
            "description":  "Substitution rules derived from the original Flores et al. dataset",
            "last_updated": str(date.today()),
        },
        "word_rules": word_rules,
        "char_rules":  char_rules,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved → {output_file}\n")


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────
def generate_if_missing(output_file: str = OUTPUT_FILE):
    """Called by main.py — only generates if the file is absent."""
    if not os.path.exists(output_file):
        print(f"⚠  {output_file} not found — fetching from GitHub...")
        generate(output_file)
    else:
        print(f"✓ Flores rules found: {output_file}")


if __name__ == "__main__":
    generate()
