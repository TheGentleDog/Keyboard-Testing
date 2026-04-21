#!/usr/bin/env python3
# =============================================================================
# test_completion_msp.py
# Evaluates the word-completion / autocomplete suggestion system using:
#   - MSP  (Mean Selection Point) — avg chars typed before target appears
#   - Appearance Rate             — how often target word shows up in top-k
#   - Hit@1                       — how often target is the #1 suggestion
#
# Mirrors the table format:
#   Word | Baseline | Avg Input until Target Shows | MSP
#
# Usage:
#   cd files/
#   python3 ../test_completion_msp.py
#   python3 ../test_completion_msp.py --words "kamusta,hello,gutom,salamat"
#   python3 ../test_completion_msp.py --lang tagalog
#   python3 ../test_completion_msp.py --lang both --top-k 5
#   python3 ../test_completion_msp.py --output msp_results.json
# =============================================================================

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR  = os.path.join(SCRIPT_DIR, "files")

if os.path.isdir(FILES_DIR):
    os.chdir(FILES_DIR)
    sys.path.insert(0, FILES_DIR)
elif os.path.isfile("model.py"):
    pass
else:
    print("❌  Run this script from the project root or from files/.")
    sys.exit(1)

warnings.filterwarnings("ignore")

from model import NgramModel

# ---------------------------------------------------------------------------
# Read PREDICTION_LANGUAGE from config.py (mirrors the live app setting)
# ---------------------------------------------------------------------------
try:
    from config import PREDICTION_LANGUAGE as _CFG_LANG
except ImportError:
    _CFG_LANG = "both"

_LANG_MAP = {"filipino": "filipino", "english": "english", "both": "both"}
CONFIG_PREDICTION_LANGUAGE = _LANG_MAP.get(_CFG_LANG.lower(), "both")

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_COLOUR = sys.platform != "win32"
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _COLOUR else t
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
RED    = lambda t: _c("31", t)
DIM    = lambda t: _c("2",  t)


# =============================================================================
# 1. MODEL LOADER
# =============================================================================

def load_model() -> NgramModel:
    model = NgramModel()
    if model.load_cache():
        model.load_user_learning()
        print(GREEN("✓ Model loaded from cache."))
    else:
        print(YELLOW("⚙  No cache — training from datasets…"))
        model.train_from_builtin()
        model.save_cache()
        model.load_user_learning()
        print(GREEN("✓ Model trained and cached."))
    return model


# =============================================================================
# 2. WORD LIST HELPERS
# =============================================================================

def load_vocab_words(lang: str) -> list[str]:
    """Pull all vocabulary words from the dataset for a given language."""
    path = "english_dataset.json" if lang == "english" else "filipino_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = set()
    for category, word_list in data.get("vocabulary", {}).items():
        for w in word_list:
            w = w.lower().strip()
            if w.isalpha() and len(w) >= 2:
                words.add(w)
    for seq in data.get("corpus_sequences", []):
        for token in seq:
            t = token.lower().strip()
            if t.isalpha() and len(t) >= 2:
                words.add(t)
    return sorted(words)


# =============================================================================
# 3. CORE EVALUATION — single word
# =============================================================================

def evaluate_word(
    model:    NgramModel,
    word:     str,
    top_k:    int = 5,
    context:  list[str] = None,
    language: str = "both",
) -> dict:
    """
    For a single target word, type it one character at a time and ask
    get_completion_suggestions() after each character.

    The `language` parameter mirrors config.PREDICTION_LANGUAGE so the
    test sees the same filtered suggestions as the live keyboard.

    Returns a dict with:
        word            — the target word
        baseline        — total characters in the word
        selection_point — chars typed when target first appeared (None if never)
        msp             — selection_point / baseline  (None if never appeared)
        appeared        — True/False
        hit_at_1        — True if target was the #1 suggestion when it appeared
        prefix_details  — list of per-prefix records (prefix, suggestions, hit)
    """
    word     = word.lower().strip()
    baseline = len(word)
    context  = context or []

    prefix_details  = []
    selection_point = None
    hit_at_1        = False

    for i in range(1, baseline + 1):
        prefix      = word[:i]
        suggestions = model.get_completion_suggestions(
            prefix, context=context, max_results=top_k, language=language
        )

        hit = word in suggestions

        if hit and selection_point is None:
            selection_point = i
            hit_at_1        = (suggestions[0] == word) if suggestions else False

        prefix_details.append({
            "prefix":      prefix,
            "prefix_len":  i,
            "suggestions": suggestions,
            "hit":         hit,
        })

    msp = round(selection_point / baseline, 4) if selection_point else None

    return {
        "word":            word,
        "baseline":        baseline,
        "selection_point": selection_point,
        "msp":             msp,
        "appeared":        selection_point is not None,
        "hit_at_1":        hit_at_1,
        "prefix_details":  prefix_details,
    }


# =============================================================================
# 4. BATCH EVALUATION
# =============================================================================

def evaluate_word_list(
    model:    NgramModel,
    words:    list[str],
    top_k:    int = 5,
    lang:     str = "both",
    language: str = "both",
) -> list[dict]:
    results = []
    for word in words:
        r = evaluate_word(model, word, top_k=top_k, language=language)
        r["lang"] = lang
        results.append(r)
    return results


def evaluate_full_vocab(
    model:     NgramModel,
    lang:      str,
    top_k:     int = 5,
    max_words: int = None,
    language:  str = "both",
) -> list[dict]:
    """Evaluate every word in the vocabulary for a given language."""
    words = load_vocab_words(lang)
    if max_words:
        import random; random.seed(42)
        words = random.sample(words, min(max_words, len(words)))
    print(f"  ✓ {len(words):,} words to evaluate for {lang}")
    results = []
    for word in words:
        r = evaluate_word(model, word, top_k=top_k, language=language)
        r["lang"] = lang
        results.append(r)
    return results


# =============================================================================
# 5. AGGREGATE STATS
# =============================================================================

def aggregate(results: list[dict]) -> dict:
    """Compute aggregate MSP stats from a list of word results."""
    n            = len(results)
    appeared     = [r for r in results if r["appeared"]]
    hit1_count   = sum(1 for r in results if r["hit_at_1"])
    msps         = [r["msp"] for r in appeared]

    return {
        "n_words":         n,
        "n_appeared":      len(appeared),
        "n_not_appeared":  n - len(appeared),
        "appearance_rate": round(len(appeared) / n, 4) if n else 0,
        "hit_at_1_rate":   round(hit1_count / n, 4)   if n else 0,
        "avg_msp":         round(sum(msps) / len(msps), 4) if msps else None,
        "min_msp":         round(min(msps), 4) if msps else None,
        "max_msp":         round(max(msps), 4) if msps else None,
    }


# =============================================================================
# 6. TERMINAL REPORTING
# =============================================================================

def print_header(title: str):
    w = 72
    print("\n" + "═" * w)
    print(BOLD(f"  {title}"))
    print("═" * w)


def print_word_table(results: list[dict], top_k: int):
    """Print the MSP table."""
    print(f"\n  {BOLD('Word Completion — MSP Table')}")
    print(f"  {'Word':<16} {'Baseline':>10}  {'Input until':>14}  {'MSP':>16}  {'Hit@1':>6}  {'In Top-'+str(top_k):>8}")
    print(f"  {'':16} {'(chars)':>10}  {'Target Shows':>14}  {'':>16}  {'':>6}  {'':>8}")
    print("  " + "─" * 78)

    for r in results:
        word  = r["word"]
        base  = r["baseline"]
        sp    = r["selection_point"]
        msp   = r["msp"]
        h1    = r["hit_at_1"]
        hit   = r["appeared"]

        sp_str  = str(sp) if sp else "—"
        msp_str = f"{sp}/{base} = {msp:.2f}" if msp else "never appeared"
        h1_str  = GREEN("✓") if h1  else DIM("—")
        hit_str = GREEN("✓") if hit else RED("✗")

        print(f"  {word.capitalize():<16} {base:>10}  {sp_str:>14}  {msp_str:>16}  {h1_str:>6}  {hit_str:>8}")


def print_aggregate(stats: dict, label: str):
    print(f"\n  {CYAN(BOLD(label))}")
    print(f"  Words evaluated   : {stats['n_words']:,}")
    print(f"  Appeared in top-k : {stats['n_appeared']:,}  ({stats['appearance_rate']*100:.1f}%)")
    print(f"  Hit@1 rate        : {stats['hit_at_1_rate']*100:.1f}%")
    if stats["avg_msp"] is not None:
        print(f"  Avg MSP           : {stats['avg_msp']:.4f}")
        print(f"  MSP range         : {stats['min_msp']:.4f} – {stats['max_msp']:.4f}")
    else:
        print(f"  Avg MSP           : n/a (target never appeared)")


def print_prefix_drill(result: dict):
    """Show per-prefix breakdown for a single word."""
    print(f"\n  {BOLD('Prefix drill-down:')}  '{result['word']}'")
    print(f"  {'Prefix':<12} {'Suggestions':<50}  {'Hit':>5}")
    print("  " + "─" * 70)
    for d in result["prefix_details"]:
        sugg_str = ", ".join(d["suggestions"][:5]) or "—"
        hit_str  = GREEN("✓") if d["hit"] else RED("✗")
        print(f"  {d['prefix']:<12} {sugg_str:<50}  {hit_str:>5}")


# =============================================================================
# 7. TKINTER SUMMARY TABLE
# =============================================================================

def show_msp_table(
    en_stats: dict | None,
    fil_stats: dict | None,
    top_k:    int,
):
    """
    Pop up a tkinter window showing only the aggregate summary table.
    If both languages were evaluated → side-by-side English vs Tagalog.
    If only one language → single-column summary.
    """
    try:
        import tkinter as tk
    except ImportError:
        print("  ⚠  tkinter not available — skipping summary window.")
        return

    # ── Colours ───────────────────────────────────────────────────────────────
    BG        = "#1e1e2e"
    HEADER_BG = "#313244"
    ROW_ODD   = "#262637"
    ROW_EVEN  = "#1e1e2e"
    TEXT_FG   = "#cdd6f4"
    HEADER_FG = "#89b4fa"
    METRIC_FG = "#a6adc8"
    TITLE_FG  = "#cba6f7"
    BETTER_BG = "#1e3a2f"
    BETTER_FG = "#a6e3a1"
    EN_COL    = "#89dceb"
    FIL_COL   = "#f38ba8"
    BORDER    = "#45475a"

    # ── Summary rows: (label, key, is_pct, low_is_better) ────────────────────
    summary_rows = [
        ("Words Evaluated",   "n_words",         False, False),
        ("Appeared in Top-K", "n_appeared",       False, False),
        ("Appearance Rate",   "appearance_rate",  True,  False),
        ("Hit@1 Rate",        "hit_at_1_rate",    True,  False),
        ("Avg MSP",           "avg_msp",          False, True ),
        ("Min MSP",           "min_msp",          False, True ),
        ("Max MSP",           "max_msp",          False, True ),
    ]

    def fmt(v, is_pct):
        if v is None:             return "n/a"
        if is_pct:                return f"{v * 100:.2f}%"
        if isinstance(v, float):  return f"{v:.4f}"
        return f"{int(v):,}"

    # ── Window ────────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Word Completion — MSP Summary")
    root.configure(bg=BG)
    root.resizable(False, False)

    both = en_stats and fil_stats
    w    = 600 if both else 400
    h    = 360
    x    = (root.winfo_screenwidth()  - w) // 2
    y    = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    # ── Title ─────────────────────────────────────────────────────────────────
    title_frame = tk.Frame(root, bg=HEADER_BG, pady=12)
    title_frame.pack(fill="x")
    tk.Label(title_frame,
             text="Word Completion  ·  MSP Summary",
             bg=HEADER_BG, fg=TITLE_FG,
             font=("Segoe UI", 13, "bold")).pack()
    tk.Label(title_frame,
             text=f"Mean Selection Point  ·  Top-{top_k} suggestions",
             bg=HEADER_BG, fg=METRIC_FG,
             font=("Segoe UI", 9)).pack()

    # ── Table ─────────────────────────────────────────────────────────────────
    table_outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
    table_outer.pack(fill="both", expand=True, padx=20, pady=(14, 6))

    table = tk.Frame(table_outer, bg=BG)
    table.pack(fill="both", expand=True)

    col_widths = [22, 14, 14] if both else [22, 18]

    def cell(text, row, col, bg, fg, bold=False, anchor="center"):
        frm = tk.Frame(table, bg=BORDER)
        frm.grid(row=row, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))
        tk.Label(frm, text=text, bg=bg, fg=fg,
                 font=("Segoe UI", 10, "bold") if bold else ("Segoe UI", 10),
                 anchor=anchor, padx=10, pady=7,
                 width=col_widths[col]).pack(fill="both", expand=True)

    # column headers
    cell("Metric", 0, 0, HEADER_BG, HEADER_FG, bold=True, anchor="w")
    if both:
        cell("🇺🇸  English", 0, 1, HEADER_BG, EN_COL,  bold=True)
        cell("🇵🇭  Tagalog", 0, 2, HEADER_BG, FIL_COL, bold=True)
    else:
        lang_label = "🇺🇸  English" if en_stats else "🇵🇭  Tagalog"
        lang_col   = EN_COL if en_stats else FIL_COL
        cell(lang_label, 0, 1, HEADER_BG, lang_col, bold=True)

    stats = en_stats or fil_stats
    for i, (label, key, is_pct, low_better) in enumerate(summary_rows):
        row_bg = ROW_ODD if i % 2 == 0 else ROW_EVEN

        if both:
            ev = en_stats.get(key)
            fv = fil_stats.get(key)

            if ev is not None and fv is not None and ev != fv:
                en_better  = (ev < fv) if low_better else (ev > fv)
                fil_better = not en_better
            else:
                en_better = fil_better = False

            cell(label,       i+1, 0, row_bg,                              METRIC_FG,                   anchor="w")
            cell(fmt(ev, is_pct), i+1, 1, BETTER_BG if en_better  else row_bg, BETTER_FG if en_better  else TEXT_FG, bold=en_better)
            cell(fmt(fv, is_pct), i+1, 2, BETTER_BG if fil_better else row_bg, BETTER_FG if fil_better else TEXT_FG, bold=fil_better)
        else:
            v = stats.get(key)
            cell(label,          i+1, 0, row_bg, METRIC_FG, anchor="w")
            cell(fmt(v, is_pct), i+1, 1, row_bg, TEXT_FG)

    for c in range(len(col_widths)):
        table.columnconfigure(c, weight=1)

    # ── Legend ────────────────────────────────────────────────────────────────
    if both:
        leg = tk.Frame(root, bg=BG)
        leg.pack(pady=(4, 0))
        box = tk.Frame(leg, bg=BETTER_BG, width=13, height=13)
        box.pack(side="left", padx=(0, 6))
        box.pack_propagate(False)
        tk.Label(leg, text="= better score  (for MSP, lower is better)",
                 bg=BG, fg=METRIC_FG,
                 font=("Segoe UI", 8, "italic")).pack(side="left")

    # ── Close button ──────────────────────────────────────────────────────────
    tk.Button(root, text="  Close  ",
              bg=HEADER_BG, fg=TEXT_FG,
              activebackground=BORDER, activeforeground=TEXT_FG,
              relief="flat", font=("Segoe UI", 10),
              cursor="hand2", command=root.destroy,
              padx=16, pady=6).pack(pady=(6, 14))

    root.mainloop()


# =============================================================================
# 8. JSON EXPORT
# =============================================================================

def save_results(
    output_path:  str,
    args,
    word_results: list[dict],
    en_stats:     dict | None,
    fil_stats:    dict | None,
):
    import datetime

    def summarise(r):
        return {k: v for k, v in r.items() if k != "prefix_details"}

    doc = {
        "meta": {
            "generated_at":        datetime.datetime.now().isoformat(timespec="seconds"),
            "top_k":               args.top_k,
            "languages":           args.lang,
            "prediction_language": CONFIG_PREDICTION_LANGUAGE,
            "model_cache":         "ngram_model_standalone.json",
            "description":         "MSP evaluation of word completion suggestions",
        },
        "word_results":      [summarise(r) for r in word_results],
        "word_results_full": word_results,
    }

    if en_stats:
        doc["english_aggregate"] = en_stats
    if fil_stats:
        doc["tagalog_aggregate"] = fil_stats

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 Results saved → {output_path}")
    print(f"     ({len(word_results):,} words documented)")


# =============================================================================
# 9. MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MSP evaluation of the word-completion suggestion system."
    )
    p.add_argument("--words",     type=str,  default=None,               metavar="W1,W2,...",
                   help="Comma-separated list of specific words to test (default: full vocab)")
    p.add_argument("--lang",      choices=["english", "tagalog", "both"], default="both",
                   help="Which language vocabulary to pull words from (default: both)")
    p.add_argument("--top-k",     type=int,  default=5,                  metavar="K",
                   help="Suggestion bar size (default: 5)")
    p.add_argument("--max-words", type=int,  default=100,                metavar="N",
                   help="Cap words per language when using full vocab (default: 100)")
    p.add_argument("--drill",     type=str,  default=None,               metavar="WORD",
                   help="Print per-prefix breakdown for one specific word")
    p.add_argument("--output",    type=str,  default="msp_results.json", metavar="FILE",
                   help="Output JSON file (default: msp_results.json)")
    return p.parse_args()


def main():
    args = parse_args()

    print(BOLD(CYAN("\n╔══════════════════════════════════════════════════════════════════╗")))
    print(BOLD(CYAN("║   WORD COMPLETION — MSP EVALUATION                              ║")))
    print(BOLD(CYAN("║   Mean Selection Point  ·  Appearance Rate  ·  Hit@1            ║")))
    print(BOLD(CYAN("╚══════════════════════════════════════════════════════════════════╝")))

    # ── Language filter ───────────────────────────────────────────────────────
    language = CONFIG_PREDICTION_LANGUAGE
    lang_display = {
        "both":     "Both (Filipino + English mixed)",
        "filipino": "Filipino only",
        "english":  "English only",
    }.get(language, language)
    print(f"\n  {BOLD('Language filter')} (from config.py): {CYAN(lang_display)}")
    if language != "both":
        print(f"  {DIM('Suggestions filtered the same way as the live keyboard.')}")

    print_header("LOADING MODEL")
    model = load_model()

    all_results = []
    en_stats    = None
    fil_stats   = None

    if args.words:
        word_list = [w.strip().lower() for w in args.words.split(",") if w.strip()]
        print_header(f"EVALUATING {len(word_list)} SPECIFIC WORDS")
        results = evaluate_word_list(
            model, word_list, top_k=args.top_k, lang=args.lang, language=language
        )
        all_results.extend(results)
        print_word_table(results, args.top_k)

    else:
        if args.lang in ("english", "both"):
            print_header("EVALUATING ENGLISH VOCABULARY")
            en_results = evaluate_full_vocab(
                model, "english", top_k=args.top_k,
                max_words=args.max_words, language=language
            )
            en_stats = aggregate(en_results)
            print_word_table(en_results[:20], args.top_k)
            if len(en_results) > 20:
                print(f"  … and {len(en_results)-20} more (see JSON output)")
            print_aggregate(en_stats, "English — Aggregate Stats")
            all_results.extend(en_results)

        if args.lang in ("tagalog", "both"):
            print_header("EVALUATING TAGALOG VOCABULARY")
            fil_results = evaluate_full_vocab(
                model, "tagalog", top_k=args.top_k,
                max_words=args.max_words, language=language
            )
            fil_stats = aggregate(fil_results)
            print_word_table(fil_results[:20], args.top_k)
            if len(fil_results) > 20:
                print(f"  … and {len(fil_results)-20} more (see JSON output)")
            print_aggregate(fil_stats, "Tagalog — Aggregate Stats")
            all_results.extend(fil_results)

    # ── Per-prefix drill ──────────────────────────────────────────────────────
    if args.drill:
        target = args.drill.lower().strip()
        match  = next((r for r in all_results if r["word"] == target), None)
        if match:
            print_prefix_drill(match)
        else:
            print_header(f"PREFIX DRILL: '{target}'")
            r = evaluate_word(model, target, top_k=args.top_k, language=language)
            print_prefix_drill(r)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    save_results(args.output, args, all_results, en_stats, fil_stats)

    # ── GUI summary table ─────────────────────────────────────────────────────
    print(f"\n  📊 Opening summary table window…")
    show_msp_table(en_stats, fil_stats, args.top_k)

    print(GREEN(BOLD("\n✓ MSP evaluation complete.\n")))


if __name__ == "__main__":
    main()
