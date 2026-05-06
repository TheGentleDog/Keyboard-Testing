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
#   python3 ../test_completion_msp.py --graph-output msp_graph.png
# =============================================================================

import os
import sys
import json
import argparse
import warnings
import random
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
# 8. JSON EXPORT + GRAPHING
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


def save_graph(
    output_path: str,
    word_results: list[dict],
    en_stats: dict | None,
    fil_stats: dict | None,
    top_k: int,
):
    """
    Save a PNG chart from the current MSP run:
      - scatterplot of baseline word length vs decimal MSP
      - horizontal baseline line where MSP equals 1.0
    """
    if not output_path:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("\n  ! matplotlib not installed - skipping graph output.")
        print("     Install it with: pip install matplotlib")
        return

    if not word_results:
        print("\n  ! No word results available - skipping graph output.")
        return

    groups = []
    if en_stats:
        groups.append(("English", "english", "#3266ad", "o", 40))
    if fil_stats:
        groups.append(("Tagalog", "tagalog", "#d85a30", "^", 50))
    if not groups:
        lang = word_results[0].get("lang", "words")
        groups.append((lang.capitalize(), lang, "#3266ad", "o", 40))

    fig, ax = plt.subplots(figsize=(9, 7))

    rng = random.Random(42)
    plotted_results = []
    for label, lang_key, color, marker, size in groups:
        lang_results = [
            r for r in word_results
            if r.get("lang", lang_key) == lang_key and r.get("msp") is not None
        ]
        if not lang_results:
            continue

        plotted_results.extend(lang_results)
        x_vals = [
            max(1.0, r["baseline"] + rng.uniform(-0.25, 0.25))
            for r in lang_results
        ]
        y_vals = [
            min(1.05, max(0.0, r["msp"] + rng.uniform(-0.01, 0.01)))
            for r in lang_results
        ]
        ax.scatter(
            x_vals,
            y_vals,
            marker=marker,
            color=color,
            alpha=0.25,
            edgecolors=color,
            linewidths=0.5,
            s=size,
            label=label,
            zorder=3,
        )

    if not plotted_results:
        print("\n  ! No words have MSP values - skipping graph output.")
        plt.close(fig)
        return

    max_baseline = max(r.get("baseline", 0) for r in plotted_results)
    max_msp = max(r.get("msp", 0) for r in plotted_results)
    y_max = max(1.05, max_msp + 0.05)

    ax.plot(
        [1, max_baseline],
        [1.0, 1.0],
        linestyle="--",
        color="#888780",
        linewidth=1.2,
        alpha=0.6,
        label="MSP = 1.0 baseline",
        zorder=2,
    )

    never_count = sum(1 for r in word_results if r.get("selection_point") is None)
    if never_count:
        ax.text(
            0.99,
            0.02,
            f"{never_count:,} word(s) never appeared",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#888780",
        )

    ax.set_xlabel("Baseline (word length)", fontsize=12, color="#444441")
    ax.set_ylabel("MSP", fontsize=12, color="#444441")
    ax.set_title(
        f"Baseline vs MSP (Top-{top_k})",
        fontsize=14,
        fontweight="medium",
        color="#2c2c2a",
        pad=14,
    )
    ax.set_xlim(1, max_baseline + 1)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(color="#d3d1c7", linestyle="-", linewidth=0.5, alpha=0.7, zorder=1)
    ax.set_facecolor("#fafaf9")
    fig.patch.set_facecolor("#ffffff")

    for spine in ax.spines.values():
        spine.set_edgecolor("#b4b2a9")
        spine.set_linewidth(0.8)

    ax.tick_params(colors="#5f5e5a", labelsize=10)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#d3d1c7", facecolor="#ffffff")

    n_english = sum(1 for r in word_results if r.get("lang") == "english")
    n_tagalog = sum(1 for r in word_results if r.get("lang") == "tagalog")
    fig.text(
        0.13,
        0.01,
        f"n = {n_english:,} English words, {n_tagalog:,} Tagalog words",
        fontsize=9,
        color="#888780",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Graph saved -> {output_path}")


PREDICTION_MODE_COLUMNS = [
    ("Tagalog mode", "filipino"),
    ("English mode", "english"),
    ("Bilingual mode", "both"),
]


def _fmt_mode_value(stats: dict | None, key: str, is_pct: bool, lower_is_better: bool = False) -> str:
    if not stats:
        return "n/a"
    value = stats.get(key)
    if value is None:
        return "n/a"
    if is_pct:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{int(value):,}"


def print_prediction_mode_table(matrix: dict, top_k: int):
    """Print separate MSP tables for Tagalog and English test cases."""
    print_header("MSP PREDICTION-MODE COMPARISON")

    metric_rows = [
        ("Evaluated Words", "n_words", False, False),
        ("Top-K Appearance", "n_appeared", False, False),
        ("Rate of Appearance in Suggestions", "appearance_rate", True, False),
        ("Avg MSP (Mean Selection Points)", "avg_msp", False, True),
    ]

    cell_w = 17
    for test_lang, title in (("tagalog", "Tagalog words"), ("english", "English words")):
        if test_lang not in matrix:
            continue

        print(f"\n  {title}")
        header = f"  {'Metric':<18}" + ''.join(f"{mode_label:>{cell_w}}" for mode_label, _ in PREDICTION_MODE_COLUMNS)
        print(header)
        print("  " + "-" * (18 + cell_w * len(PREDICTION_MODE_COLUMNS)))

        for label, key, is_pct, lower_is_better in metric_rows:
            row = f"  {label:<18}"
            for _, mode_key in PREDICTION_MODE_COLUMNS:
                row += f"{_fmt_mode_value(matrix[test_lang].get(mode_key), key, is_pct, lower_is_better):>{cell_w}}"
            print(row)

    print(f"\n  Lower Avg MSP is better. Rates and appearance counts are higher-is-better. Top-{top_k} suggestions.")


def show_prediction_mode_table(matrix: dict, top_k: int):
    """Show the prediction-mode comparison as one Tkinter table."""
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except ImportError:
        print("  ! tkinter not available - skipping comparison table window.")
        return

    if not matrix:
        return

    columns = []
    for lang_label, lang_key in (("Filipino/Tagalog", "tagalog"), ("English", "english")):
        if lang_key in matrix:
            for mode_label, mode_key in PREDICTION_MODE_COLUMNS:
                columns.append((lang_label, mode_label, lang_key, mode_key))

    rows = [
        ("Evaluated Words", "n_words", False),
        ("Top-K Appearance", "n_appeared", False),
        ("Rate of Appearance in Suggestions", "appearance_rate", True),
        ("Avg MSP (Mean Selection Points)", "avg_msp", False),
    ]

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"  ! tkinter could not open comparison table - skipping UI ({exc})")
        return
    root.title("MSP Prediction-Mode Comparison")
    root.configure(bg="#1e1e2e")
    root.resizable(False, False)

    title_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
    header_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
    cell_font = tkfont.Font(family="Segoe UI", size=9)

    BG = "#1e1e2e"
    HEADER_BG = "#313244"
    ROW_A = "#292938"
    ROW_B = "#1e1e2e"
    FG = "#cdd6f4"
    DIM_FG = "#a6adc8"
    ACCENT = "#89b4fa"

    tk.Label(
        root,
        text=f"MSP Prediction-Mode Comparison - Top-{top_k}",
        bg=BG,
        fg=ACCENT,
        font=title_font,
        pady=12,
    ).pack(fill="x")

    frame = tk.Frame(root, bg=BG, padx=14, pady=8)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Metric", width=18, bg=HEADER_BG, fg=ACCENT, font=header_font, padx=8, pady=7).grid(row=0, column=0, rowspan=2, sticky="nsew", padx=1, pady=1)

    col = 1
    grouped = []
    for lang_label, _, lang_key, _ in columns:
        if not grouped or grouped[-1][0] != lang_key:
            grouped.append((lang_key, lang_label, 1))
        else:
            grouped[-1] = (grouped[-1][0], grouped[-1][1], grouped[-1][2] + 1)

    for _, lang_label, span in grouped:
        tk.Label(frame, text=lang_label, bg=HEADER_BG, fg=ACCENT, font=header_font, padx=8, pady=7).grid(row=0, column=col, columnspan=span, sticky="nsew", padx=1, pady=1)
        col += span

    for idx, (_, mode_label, _, _) in enumerate(columns, start=1):
        tk.Label(frame, text=mode_label, width=17, bg=HEADER_BG, fg=DIM_FG, font=header_font, padx=8, pady=7).grid(row=1, column=idx, sticky="nsew", padx=1, pady=1)

    for row_idx, (label, key, is_pct) in enumerate(rows, start=2):
        bg = ROW_A if row_idx % 2 == 0 else ROW_B
        tk.Label(frame, text=label, bg=bg, fg=FG, font=cell_font, anchor="w", padx=8, pady=7).grid(row=row_idx, column=0, sticky="nsew", padx=1, pady=1)
        for col_idx, (_, _, lang_key, mode_key) in enumerate(columns, start=1):
            text = _fmt_mode_value(matrix[lang_key].get(mode_key), key, is_pct)
            tk.Label(frame, text=text, bg=bg, fg=FG, font=cell_font, padx=8, pady=7).grid(row=row_idx, column=col_idx, sticky="nsew", padx=1, pady=1)

    tk.Label(
        root,
        text="Lower Avg MSP is better. Other rates are higher-is-better.",
        bg=BG,
        fg=DIM_FG,
        font=("Segoe UI", 8),
        pady=8,
    ).pack(fill="x")

    tk.Button(root, text="Close", command=root.destroy, bg=HEADER_BG, fg=FG, relief="flat", padx=18, pady=6).pack(pady=(0, 12))
    root.mainloop()


def save_prediction_mode_comparison(output_path: str, args, matrix: dict):
    if not output_path:
        return

    import datetime

    doc = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "top_k": args.top_k,
            "languages": args.lang,
            "max_words": args.max_words,
            "comparison": "MSP by test language and prediction mode",
        },
        "results": matrix,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n  Comparison results saved -> {output_path}")


def save_prediction_mode_comparison_full(output_path: str, args, matrix: dict, records: dict):
    if not output_path:
        return

    import datetime

    def summarise_word(r):
        return {k: v for k, v in r.items() if k != "prefix_details"}

    doc = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "top_k": args.top_k,
            "languages": args.lang,
            "max_words": args.max_words,
            "comparison": "MSP by test language and prediction mode",
        },
        "summary": matrix,
        "test_cases": {},
    }

    for lang, modes in records.items():
        doc["test_cases"][lang] = {}
        for mode, results in modes.items():
            doc["test_cases"][lang][mode] = {
                "word_results": [summarise_word(r) for r in results],
                "word_results_full": results,
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n  Comparison results saved -> {output_path}")


def prediction_mode_graph_path(base_path: str, prediction_language: str) -> str:
    root, ext = os.path.splitext(base_path)
    ext = ext or ".png"
    return f"{root}_{prediction_language}_mode{ext}"


def run_prediction_mode_comparison(model: NgramModel, args):
    selected_langs = ["tagalog", "english"] if args.lang == "both" else [args.lang]
    matrix = {}
    records = {}
    graph_data = {
        prediction_language: {
            "word_results": [],
            "english": None,
            "tagalog": None,
        }
        for _, prediction_language in PREDICTION_MODE_COLUMNS
    }

    for lang in selected_langs:
        matrix[lang] = {}
        records[lang] = {}
        display = "Tagalog" if lang == "tagalog" else "English"
        if args.words:
            words = [w.strip().lower() for w in args.words.split(",") if w.strip()]
        else:
            words = load_vocab_words(lang)
            if args.max_words:
                import random
                random.seed(42)
                words = random.sample(words, min(args.max_words, len(words)))

        for mode_label, prediction_language in PREDICTION_MODE_COLUMNS:
            print_header(f"{display.upper()} WORDS - {mode_label.upper()}")
            results = evaluate_word_list(
                model,
                words,
                top_k=args.top_k,
                lang=lang,
                language=prediction_language,
            )
            stats = aggregate(results)
            matrix[lang][prediction_language] = stats
            records[lang][prediction_language] = results
            graph_data[prediction_language]["word_results"].extend(results)
            graph_data[prediction_language][lang] = stats
            print_aggregate(stats, f"{display} words - {mode_label}")

    print_prediction_mode_table(matrix, args.top_k)
    save_prediction_mode_comparison_full(args.output, args, matrix, records)

    if not args.no_graph:
        for _, prediction_language in PREDICTION_MODE_COLUMNS:
            data = graph_data[prediction_language]
            save_graph(
                prediction_mode_graph_path(args.graph_output, prediction_language),
                data["word_results"],
                data["english"],
                data["tagalog"],
                args.top_k,
            )

    show_prediction_mode_table(matrix, args.top_k)


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
    p.add_argument("--compare-prediction-modes", action="store_true",
                   help="Compare tagalog, english, and both prediction modes in one table")
    p.add_argument("--drill",     type=str,  default=None,               metavar="WORD",
                   help="Print per-prefix breakdown for one specific word")
    p.add_argument("--output",    type=str,  default=None, metavar="FILE",
                   help="Output JSON file (default: msp_results_<lang>.json)")
    p.add_argument("--graph-output", type=str, default=None, metavar="PNG",
                   help="Output PNG graph file (default: msp_graph_<lang>.png)")
    p.add_argument("--no-graph", action="store_true",
                   help="Skip PNG graph generation")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None and args.compare_prediction_modes:
        args.output = f"msp_mode_comparison_{args.lang}.json"
    elif args.output is None:
        args.output = f"msp_results_{args.lang}.json"
    if args.graph_output is None:
        args.graph_output = f"msp_graph_{args.lang}.png"

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

    if args.compare_prediction_modes:
        run_prediction_mode_comparison(model, args)
        print(GREEN(BOLD("\n✓ MSP prediction-mode comparison complete.\n")))
        return

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

    # ── Save graph ────────────────────────────────────────────────────────────
    if not args.no_graph:
        save_graph(
            args.graph_output,
            all_results,
            en_stats,
            fil_stats,
            args.top_k,
        )

    # ── GUI summary table ─────────────────────────────────────────────────────
    print(f"\n  📊 Opening summary table window…")
    show_msp_table(en_stats, fil_stats, args.top_k)

    print(GREEN(BOLD("\n✓ MSP evaluation complete.\n")))


if __name__ == "__main__":
    main()
