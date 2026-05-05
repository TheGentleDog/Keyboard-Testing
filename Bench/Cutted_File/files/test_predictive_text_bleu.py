#!/usr/bin/env python3
# =============================================================================
# test_predictive_text_bleu.py
# Evaluates the N-gram predictive text / next-word prediction system
# for both English and Tagalog using BLEU metrics.
#
# Usage:
#   cd files/
#   python3 ../test_predictive_text_bleu.py
#   python3 ../test_predictive_text_bleu.py --lang english
#   python3 ../test_predictive_text_bleu.py --lang tagalog
#   python3 ../test_predictive_text_bleu.py --top-k 3
#   python3 ../test_predictive_text_bleu.py --lang both --max-cases 500
#   python3 ../test_predictive_text_bleu.py --graph-output bleu_graph_both.png
# =============================================================================

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict, Counter
from math import log, exp

# ---------------------------------------------------------------------------
# Make sure we're running from the files/ directory (where the model lives)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR  = os.path.join(SCRIPT_DIR, "files")

if os.path.isdir(FILES_DIR):
    os.chdir(FILES_DIR)
    sys.path.insert(0, FILES_DIR)
elif os.path.isfile("model.py"):
    pass  # already in files/
else:
    print("❌  Run this script from the project root or from files/.")
    sys.exit(1)

# Suppress NLTK warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NLTK — download required data silently
# ---------------------------------------------------------------------------
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction,
)

# ---------------------------------------------------------------------------
# Import the project's model
# ---------------------------------------------------------------------------
from model import NgramModel

try:
    from config import PREDICTION_LANGUAGE as _CFG_PREDICTION_LANGUAGE
except Exception:
    _CFG_PREDICTION_LANGUAGE = "both"

_LANG_MAP = {
    "both": "both",
    "english": "english",
    "tagalog": "filipino",
    "filipino": "filipino",
}

CONFIG_PREDICTION_LANGUAGE = _LANG_MAP.get(
    str(_CFG_PREDICTION_LANGUAGE).lower(),
    "both",
)

# ---------------------------------------------------------------------------
# ANSI colours (disabled on Windows)
# ---------------------------------------------------------------------------
_COLOUR = sys.platform != "win32"

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
RED    = lambda t: _c("31", t)
DIM    = lambda t: _c("2",  t)


# =============================================================================
# 1. BUILD & LOAD MODEL
# =============================================================================

def load_model() -> NgramModel:
    """Train or load the shared n-gram model."""
    model = NgramModel()
    if model.load_cache():
        model.load_user_learning()
        print(GREEN("✓ Model loaded from cache."))
    else:
        print(YELLOW("⚙  No cache found — training from datasets…"))
        model.train_from_builtin()
        model.save_cache()
        model.load_user_learning()
        print(GREEN("✓ Model trained and cached."))
    return model


# =============================================================================
# 2. DATASET HELPERS
# =============================================================================

def load_sequences(lang: str) -> list[list[str]]:
    """
    Return a list of token-lists from corpus_sequences.
    Falls back to communication_corpus if corpus_sequences is absent.
    """
    path = "english_dataset.json" if lang == "english" else "filipino_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seqs = data.get("corpus_sequences", [])
    if seqs:
        return [[t.lower() for t in seq] for seq in seqs if len(seq) >= 2]

    # fallback
    corpus = data.get("communication_corpus", [])
    return [phrase.lower().split() for phrase in corpus if len(phrase.split()) >= 2]


def build_test_cases(sequences: list[list[str]], min_len: int = 3):
    """
    Convert token sequences into (context, reference_next_word) pairs.

    For each sequence we generate one test case per position ≥ 1,
    using all preceding tokens as context and the current token as reference.

    Returns list of (context_tokens, reference_word) tuples.
    """
    cases = []
    for seq in sequences:
        if len(seq) < min_len:
            continue
        for i in range(1, len(seq)):
            context = seq[:i]
            ref_word = seq[i]
            if not ref_word.isalpha():
                continue
            cases.append((context, ref_word))
    return cases


# =============================================================================
# 3. PREDICTION WRAPPER
# =============================================================================

def predict_next_words(
    model: NgramModel,
    context: list[str],
    top_k: int = 5,
    language: str = "both",
) -> list[str]:
    """
    Ask the model for the top-k next-word predictions given a context.
    Uses get_next_word_suggestions (trigram → bigram → unigram fallback).
    """
    return model.get_next_word_suggestions(
        context=context,
        max_results=top_k,
        language=language,
    )


# =============================================================================
# 4. BLEU CALCULATIONS
# =============================================================================

SMOOTHER = SmoothingFunction().method1   # add-one smoothing for short sentences


def sentence_bleu_score(
    reference: str,
    hypothesis: str,
    weights=(0.5, 0.5),
) -> float:
    """
    BLEU score between a single reference word and a single hypothesis word.

    Because both strings are single words (unigrams), we use 1-gram + 2-gram
    character-level n-grams to give partial credit (e.g. "gutom" vs "guto").
    """
    ref_chars  = list(reference)
    hyp_chars  = list(hypothesis)
    return sentence_bleu(
        [ref_chars],
        hyp_chars,
        weights=weights,
        smoothing_function=SMOOTHER,
    )


def top_k_bleu(
    reference: str,
    hypotheses: list[str],
    weights=(0.5, 0.5),
) -> float:
    """Best BLEU among all top-k hypotheses (oracle top-k)."""
    if not hypotheses:
        return 0.0
    return max(sentence_bleu_score(reference, h, weights) for h in hypotheses)


def corpus_level_bleu(
    references: list[str],
    hypotheses_list: list[list[str]],
    weights=(0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Corpus-level BLEU treating each reference word as a 1-sentence document.
    hypotheses_list[i] is the ranked list of predicted words for case i;
    we use the best-matching hypothesis (oracle selection).
    """
    ref_corpus  = []
    hyp_corpus  = []
    for ref, hyps in zip(references, hypotheses_list):
        if not hyps:
            continue
        # pick hypothesis with the highest sentence BLEU
        best_hyp = max(hyps, key=lambda h: sentence_bleu_score(ref, h))
        ref_corpus.append([list(ref)])      # list of reference lists
        hyp_corpus.append(list(best_hyp))   # hypothesis tokens (characters)
    if not ref_corpus:
        return 0.0
    return corpus_bleu(ref_corpus, hyp_corpus, weights=weights,
                       smoothing_function=SMOOTHER)


# =============================================================================
# 5. METRICS COLLECTION
# =============================================================================

def evaluate(
    model:   NgramModel,
    cases:   list[tuple[list[str], str]],
    top_k:   int = 5,
    language: str = "both",
    verbose: bool = False,
    sample_n: int = 10,
) -> dict:
    """
    Run all test cases and collect:
        - hit@1   : reference is the #1 prediction
        - hit@k   : reference appears anywhere in top-k predictions
        - MRR     : Mean Reciprocal Rank
        - avg sentence BLEU (top-1)
        - avg oracle BLEU  (best of top-k)
        - corpus BLEU
    """
    hit1_count  = 0
    hitk_count  = 0
    mrr_sum     = 0.0
    bleu1_sum   = 0.0
    bleu_oracle_sum = 0.0

    all_refs     = []
    all_hyps     = []
    all_records  = []   # every test case, saved to JSON
    sample_records = []

    for idx, (context, ref) in enumerate(cases):
        preds = predict_next_words(model, context, top_k=top_k, language=language)

        # Hit metrics
        rank = None
        for r, word in enumerate(preds, start=1):
            if word == ref:
                rank = r
                break

        hit1  = rank == 1
        hitk  = rank is not None
        rr    = (1.0 / rank) if rank else 0.0
        b1    = sentence_bleu_score(ref, preds[0]) if preds else 0.0
        bok   = top_k_bleu(ref, preds)

        hit1_count      += hit1
        hitk_count      += hitk
        mrr_sum         += rr
        bleu1_sum       += b1
        bleu_oracle_sum += bok

        all_refs.append(ref)
        all_hyps.append(preds)

        record = {
            "id":          idx,
            "context":     " ".join(context),
            "context_len": len(context),
            "reference":   ref,
            "predictions": preds,
            "hit_at_1":    hit1,
            f"hit_at_{top_k}": hitk,
            "rank":        rank,
            "reciprocal_rank": round(rr, 4),
            "bleu_top1":   round(b1, 4),
            "bleu_oracle": round(bok, 4),
        }
        all_records.append(record)

        if verbose and idx < sample_n:
            sample_records.append(record)

    n = len(cases)
    corp_bleu = corpus_level_bleu(all_refs, all_hyps)

    return {
        "n_cases":         n,
        "hit@1":           hit1_count / n,
        f"hit@{top_k}":    hitk_count / n,
        "MRR":             mrr_sum / n,
        "avg_bleu_top1":   bleu1_sum / n,
        "avg_bleu_oracle": bleu_oracle_sum / n,
        "corpus_bleu":     corp_bleu,
        "prediction_language": language,
        "samples":         sample_records,
        "all_records":     all_records,
    }


# =============================================================================
# 6. CONTEXT-WINDOW BREAKDOWN
# =============================================================================

def evaluate_by_context_len(
    model:  NgramModel,
    cases:  list[tuple[list[str], str]],
    top_k:  int = 5,
    language: str = "both",
) -> dict:
    """Break down hit@1 and avg_bleu_oracle by context length (1, 2, 3+)."""
    buckets = defaultdict(list)
    for context, ref in cases:
        cl = min(len(context), 3)
        buckets[cl].append((context, ref))

    result = {}
    for cl, bucket_cases in sorted(buckets.items()):
        label = f"ctx_len={cl}" if cl < 3 else "ctx_len=3+"
        hits = 0
        bleu = 0.0
        for context, ref in bucket_cases:
            preds = predict_next_words(model, context, top_k=top_k, language=language)
            hits  += any(p == ref for p in preds[:1])
            bleu  += top_k_bleu(ref, preds)
        n = len(bucket_cases)
        result[label] = {
            "n":              n,
            "hit@1":          hits / n,
            "avg_bleu_oracle": bleu / n,
        }
    return result


# =============================================================================
# 7. REPORTING
# =============================================================================

def _bar(value: float, width: int = 30, fill="█", empty="░") -> str:
    filled = round(value * width)
    return fill * filled + empty * (width - filled)


def print_header(title: str):
    w = 72
    print("\n" + "═" * w)
    print(BOLD(f"  {title}"))
    print("═" * w)


def print_metrics(label: str, metrics: dict, top_k: int):
    print(f"\n{CYAN(BOLD(label))}")
    print(f"  Test cases     : {metrics['n_cases']:,}")
    print()

    rows = [
        ("Hit@1  (exact match)",       metrics["hit@1"],           True),
        (f"Hit@{top_k}  (in top-{top_k})", metrics[f"hit@{top_k}"], True),
        ("MRR   (mean recip. rank)",   metrics["MRR"],              True),
        ("Avg BLEU – top-1 pred",      metrics["avg_bleu_top1"],    False),
        ("Avg BLEU – oracle top-k",    metrics["avg_bleu_oracle"],  False),
        ("Corpus BLEU",                metrics["corpus_bleu"],      False),
    ]

    for name, val, is_rate in rows:
        pct_str = f"{val*100:6.2f}%"
        bar     = _bar(min(val * (1 if is_rate else 4), 1.0))
        colour  = GREEN if val >= 0.5 else (YELLOW if val >= 0.25 else RED)
        print(f"  {name:<36} {colour(pct_str)}  {DIM(bar)}")


def print_context_breakdown(breakdown: dict):
    print(f"\n  {BOLD('Context-length breakdown:')}")
    print(f"  {'Context':<14} {'n':>6}  {'Hit@1':>8}  {'Oracle BLEU':>12}")
    print("  " + "-" * 46)
    for label, stats in breakdown.items():
        h1  = f"{stats['hit@1']*100:.1f}%"
        bok = f"{stats['avg_bleu_oracle']*100:.1f}%"
        print(f"  {label:<14} {stats['n']:>6}  {h1:>8}  {bok:>12}")


def print_samples(samples: list[dict], top_k: int):
    if not samples:
        return
    print(f"\n  {BOLD('Sample predictions (first 10 test cases):')}")
    print(f"  {'Context':<28} {'Reference':<14} {'Top-1':>10}  {'Hit':>5}  {'BLEU':>6}")
    print("  " + "-" * 70)
    for s in samples:
        ctx   = s["context"][-28:] if len(s["context"]) > 28 else s["context"]
        ref   = s["reference"]
        top1  = s["predictions"][0] if s["predictions"] else "—"
        hit   = GREEN("✓") if s["hit@1"] else RED("✗")
        bleu  = f"{s['bleu_oracle']*100:.1f}%"
        print(f"  {ctx:<28} {ref:<14} {top1:>10}  {hit:>5}  {bleu:>6}")


def print_comparison(en_metrics: dict, fil_metrics: dict, top_k: int):
    print_header("SIDE-BY-SIDE COMPARISON")
    keys = [
        ("hit@1",             f"Hit@1"),
        (f"hit@{top_k}",      f"Hit@{top_k}"),
        ("MRR",               "MRR"),
        ("avg_bleu_top1",     "Avg BLEU top-1"),
        ("avg_bleu_oracle",   "Avg BLEU oracle"),
        ("corpus_bleu",       "Corpus BLEU"),
    ]
    print(f"\n  {'Metric':<26} {'English':>10}  {'Tagalog':>10}  {'Winner':>10}")
    print("  " + "-" * 62)
    for key, label in keys:
        ev = en_metrics.get(key, 0.0)
        fv = fil_metrics.get(key, 0.0)
        winner = "English" if ev > fv else ("Tagalog" if fv > ev else "Tie")
        wc     = GREEN if winner == "English" else (CYAN if winner == "Tagalog" else YELLOW)
        print(f"  {label:<26} {ev*100:>9.2f}%  {fv*100:>9.2f}%  {wc(winner):>10}")


PREDICTION_MODE_COLUMNS = [
    ("tagalog mode", "filipino"),
    ("english mode", "english"),
    ("both mode", "both"),
]


def _fmt_mode_metric(metrics: dict | None, key: str, is_pct: bool = True) -> str:
    if not metrics:
        return "n/a"
    value = metrics.get(key)
    if value is None:
        return "n/a"
    if is_pct:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{int(value):,}"


def print_prediction_mode_table(matrix: dict, top_k: int):
    """Print one table comparing each test language across prediction modes."""
    print_header("BLEU PREDICTION-MODE COMPARISON")

    col_labels = []
    for lang_label, lang_key in (("Filipino/Tagalog", "tagalog"), ("English", "english")):
        if lang_key in matrix:
            for mode_label, mode_key in PREDICTION_MODE_COLUMNS:
                col_labels.append((lang_label, mode_label, lang_key, mode_key))

    metric_rows = [
        ("Hit@1", "hit@1", True),
        (f"Hit@{top_k}", f"hit@{top_k}", True),
        ("MRR", "MRR", True),
        ("Avg BLEU oracle", "avg_bleu_oracle", True),
        ("Corpus BLEU", "corpus_bleu", True),
        ("Cases", "n_cases", False),
    ]

    cell_w = 21
    print(f"\n  {'':<18}" + "".join(f"{lang:<{cell_w}}" for lang, _, _, _ in col_labels))
    print(f"  {'Metric':<18}" + "".join(f"{mode:<{cell_w}}" for _, mode, _, _ in col_labels))
    print("  " + "-" * (18 + cell_w * len(col_labels)))

    for label, key, is_pct in metric_rows:
        row = f"  {label:<18}"
        for _, _, lang_key, mode_key in col_labels:
            row += f"{_fmt_mode_metric(matrix[lang_key].get(mode_key), key, is_pct):>{cell_w}}"
        print(row)

    print(f"\n  Higher is better for BLEU, Hit@K, and MRR. Top-{top_k} predictions.")


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
        ("Hit@1", "hit@1", True),
        (f"Hit@{top_k}", f"hit@{top_k}", True),
        ("MRR", "MRR", True),
        ("Avg BLEU Oracle", "avg_bleu_oracle", True),
        ("Corpus BLEU", "corpus_bleu", True),
        ("Cases", "n_cases", False),
    ]

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"  ! tkinter could not open comparison table - skipping UI ({exc})")
        return
    root.title("BLEU Prediction-Mode Comparison")
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
        text=f"BLEU Prediction-Mode Comparison - Top-{top_k}",
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
            text = _fmt_mode_metric(matrix[lang_key].get(mode_key), key, is_pct)
            tk.Label(frame, text=text, bg=bg, fg=FG, font=cell_font, padx=8, pady=7).grid(row=row_idx, column=col_idx, sticky="nsew", padx=1, pady=1)

    tk.Label(
        root,
        text="Higher is better for BLEU, Hit@K, and MRR.",
        bg=BG,
        fg=DIM_FG,
        font=("Segoe UI", 8),
        pady=8,
    ).pack(fill="x")

    tk.Button(root, text="Close", command=root.destroy, bg=HEADER_BG, fg=FG, relief="flat", padx=18, pady=6).pack(pady=(0, 12))
    root.mainloop()


def save_prediction_mode_comparison(output_path: str, args, matrix: dict, top_k: int):
    if not output_path:
        return

    import datetime

    doc = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "top_k": top_k,
            "min_seq_len": args.min_seq_len,
            "max_cases": args.max_cases,
            "languages": args.lang,
            "comparison": "BLEU by test language and prediction mode",
        },
        "results": matrix,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n  Comparison results saved -> {output_path}")


def save_prediction_mode_comparison_full(output_path: str, args, matrix: dict, records: dict, top_k: int):
    if not output_path:
        return

    import datetime

    doc = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "top_k": top_k,
            "min_seq_len": args.min_seq_len,
            "max_cases": args.max_cases,
            "languages": args.lang,
            "comparison": "BLEU by test language and prediction mode",
        },
        "summary": matrix,
        "test_cases": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n  Comparison results saved -> {output_path}")


def prediction_mode_graph_path(base_path: str, prediction_language: str) -> str:
    root, ext = os.path.splitext(base_path)
    ext = ext or ".png"
    return f"{root}_{prediction_language}_mode{ext}"


def graph_variant_path(base_path: str, suffix: str) -> str:
    root, ext = os.path.splitext(base_path)
    ext = ext or ".png"
    return f"{root}_{suffix}{ext}"


def compact_metrics(metrics: dict) -> dict:
    return {
        k: v
        for k, v in metrics.items()
        if k not in ("samples", "all_records", "breakdown")
    }


def run_prediction_mode_comparison(model: NgramModel, args):
    selected_langs = ["tagalog", "english"] if args.lang == "both" else [args.lang]
    matrix = {}
    records = {}
    graph_data = {
        prediction_language: {
            "english": None,
            "tagalog": None,
        }
        for _, prediction_language in PREDICTION_MODE_COLUMNS
    }

    for lang in selected_langs:
        display = "Tagalog (Filipino)" if lang == "tagalog" else "English"
        print_header(f"LOADING {display.upper()} TEST CASES")
        sequences = load_sequences(lang)
        cases = build_test_cases(sequences, min_len=args.min_seq_len)
        print(f"  Generated {len(cases):,} test cases")

        if args.max_cases and len(cases) > args.max_cases:
            import random
            random.seed(42)
            cases = random.sample(cases, args.max_cases)
            print(f"  Capped to {args.max_cases:,} cases")

        matrix[lang] = {}
        records[lang] = {}
        for mode_label, prediction_language in PREDICTION_MODE_COLUMNS:
            print_header(f"{display.upper()} CASES - {mode_label.upper()}")
            metrics = evaluate(
                model,
                cases,
                top_k=args.top_k,
                language=prediction_language,
                verbose=False,
            )
            graph_data[prediction_language][lang] = metrics
            matrix[lang][prediction_language] = compact_metrics(metrics)
            records[lang][prediction_language] = metrics.get("all_records", [])
            print_metrics(f"{display} cases - {mode_label}", metrics, args.top_k)

    print_prediction_mode_table(matrix, args.top_k)
    save_prediction_mode_comparison_full(args.output, args, matrix, records, args.top_k)

    if not args.no_graph:
        for _, prediction_language in PREDICTION_MODE_COLUMNS:
            data = graph_data[prediction_language]
            save_graph(
                prediction_mode_graph_path(args.graph_output, prediction_language),
                data["english"],
                data["tagalog"],
                args.top_k,
                args.lang,
                max_line_cases=None,
            )

    show_prediction_mode_table(matrix, args.top_k)


# =============================================================================
# 8. GUI SUMMARY TABLE
# =============================================================================

def show_summary_table(
    en_metrics,
    fil_metrics,
    top_k: int,
):
    """
    Opens a tkinter window showing a side-by-side summary table of
    English vs Tagalog metrics. Highlights the better-performing language
    per row in green. No winner label — just the numbers.
    """
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except ImportError:
        print("Warning: tkinter not available — skipping GUI table.")
        return

    if not en_metrics and not fil_metrics:
        return

    rows = [
        ("Hit@1  (exact match)",            "hit@1"),
        (f"Hit@{top_k}  (in top-{top_k})",  f"hit@{top_k}"),
        ("MRR   (mean recip. rank)",         "MRR"),
        ("Avg BLEU - top-1 pred",            "avg_bleu_top1"),
        ("Avg BLEU - oracle top-k",          "avg_bleu_oracle"),
        ("Corpus BLEU",                      "corpus_bleu"),
    ]

    root = tk.Tk()
    root.title("BLEU Evaluation — Summary Results")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    BG        = "#1e1e2e"
    HEADER_BG = "#313244"
    ROW_A     = "#292938"
    ROW_B     = "#1e1e2e"
    FG        = "#cdd6f4"
    FG_DIM    = "#6c7086"
    BETTER    = "#a6e3a1"
    EQUAL     = "#f9e2af"
    ACCENT    = "#89b4fa"

    title_font  = tkfont.Font(family="Segoe UI", size=13, weight="bold")
    header_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    cell_font   = tkfont.Font(family="Segoe UI Mono", size=10)
    meta_font   = tkfont.Font(family="Segoe UI", size=9)

    # Title bar
    title_frame = tk.Frame(root, bg=BG, pady=14)
    title_frame.pack(fill="x", padx=20)
    tk.Label(
        title_frame,
        text="Predictive Text — BLEU Evaluation Results",
        font=title_font, bg=BG, fg=ACCENT,
    ).pack()
    tk.Label(
        title_frame,
        text=f"Next-word prediction  •  Top-{top_k} suggestions  •  N-gram model",
        font=meta_font, bg=BG, fg=FG_DIM,
    ).pack()

    # Table
    table_frame = tk.Frame(root, bg=BG, padx=20, pady=4)
    table_frame.pack(fill="both", expand=True)

    COL_METRIC  = 0
    COL_EN      = 1
    COL_FIL     = 2
    COL_HEADERS = ["Metric", "English", "Tagalog (Filipino)"]
    COL_WIDTHS  = [28, 16, 20]

    for col, (text, width) in enumerate(zip(COL_HEADERS, COL_WIDTHS)):
        anchor = "w" if col == 0 else "center"
        tk.Label(
            table_frame, text=text, width=width,
            font=header_font, bg=HEADER_BG, fg=ACCENT,
            anchor=anchor, padx=12, pady=8, relief="flat",
        ).grid(row=0, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

    for row_idx, (label, key) in enumerate(rows, start=1):
        bg = ROW_A if row_idx % 2 == 0 else ROW_B
        ev = en_metrics.get(key,  0.0) if en_metrics  else None
        fv = fil_metrics.get(key, 0.0) if fil_metrics else None

        if ev is not None and fv is not None:
            if ev > fv:
                en_fg, fil_fg = BETTER, FG
            elif fv > ev:
                en_fg, fil_fg = FG, BETTER
            else:
                en_fg, fil_fg = EQUAL, EQUAL
        else:
            en_fg = fil_fg = FG

        en_text  = f"{ev*100:6.2f}%" if ev  is not None else "—"
        fil_text = f"{fv*100:6.2f}%" if fv  is not None else "—"

        tk.Label(
            table_frame, text=label, font=cell_font,
            bg=bg, fg=FG, anchor="w", padx=12, pady=7,
        ).grid(row=row_idx, column=COL_METRIC, sticky="nsew", padx=(0, 1), pady=(0, 1))
        tk.Label(
            table_frame, text=en_text, font=cell_font,
            bg=bg, fg=en_fg, anchor="center", padx=12, pady=7,
        ).grid(row=row_idx, column=COL_EN, sticky="nsew", padx=(0, 1), pady=(0, 1))
        tk.Label(
            table_frame, text=fil_text, font=cell_font,
            bg=bg, fg=fil_fg, anchor="center", padx=12, pady=7,
        ).grid(row=row_idx, column=COL_FIL, sticky="nsew", padx=(0, 1), pady=(0, 1))

    # Test case count row
    sep_row = len(rows) + 1
    tk.Frame(table_frame, bg=HEADER_BG, height=1).grid(
        row=sep_row, column=0, columnspan=3, sticky="ew", pady=(6, 0)
    )
    en_n  = f"{en_metrics['n_cases']:,}"  if en_metrics  else "—"
    fil_n = f"{fil_metrics['n_cases']:,}" if fil_metrics else "—"
    tk.Label(
        table_frame, text="Test cases evaluated", font=cell_font,
        bg=ROW_A, fg=FG_DIM, anchor="w", padx=12, pady=7,
    ).grid(row=sep_row + 1, column=COL_METRIC, sticky="nsew", padx=(0, 1), pady=(0, 1))
    tk.Label(
        table_frame, text=en_n, font=cell_font,
        bg=ROW_A, fg=FG_DIM, anchor="center", padx=12, pady=7,
    ).grid(row=sep_row + 1, column=COL_EN, sticky="nsew", padx=(0, 1), pady=(0, 1))
    tk.Label(
        table_frame, text=fil_n, font=cell_font,
        bg=ROW_A, fg=FG_DIM, anchor="center", padx=12, pady=7,
    ).grid(row=sep_row + 1, column=COL_FIL, sticky="nsew", padx=(0, 1), pady=(0, 1))

    # Legend
    legend_frame = tk.Frame(root, bg=BG, pady=10)
    legend_frame.pack(fill="x", padx=20)
    tk.Label(legend_frame, text="●  Higher value per row",
             font=meta_font, bg=BG, fg=BETTER).pack(side="left", padx=(0, 16))
    tk.Label(legend_frame, text="●  Equal",
             font=meta_font, bg=BG, fg=EQUAL).pack(side="left", padx=(0, 16))
    tk.Label(legend_frame, text="●  Standard",
             font=meta_font, bg=BG, fg=FG).pack(side="left")

    # Close button
    btn_frame = tk.Frame(root, bg=BG, pady=10)
    btn_frame.pack()
    tk.Button(
        btn_frame, text="  Close  ", font=header_font,
        bg=HEADER_BG, fg=FG, activebackground=ACCENT, activeforeground=BG,
        relief="flat", padx=16, pady=6, cursor="hand2",
        command=root.destroy,
    ).pack()

    root.mainloop()


# =============================================================================
# 9. JSON EXPORT
# =============================================================================

# =============================================================================
# 8. TKINTER SUMMARY TABLE
# =============================================================================

def show_summary_table(
    en_metrics,
    fil_metrics,
    top_k: int,
):
    """
    Pop up a tkinter window showing a side-by-side summary table of
    English vs Tagalog results. The better score per row is highlighted
    in green — no 'winner' label, just visual emphasis.
    Only shown when both languages were evaluated.
    """
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except ImportError:
        print("  ⚠  tkinter not available — skipping summary window.")
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    rows = [
        ("Test Cases",       "n_cases",           False),
        ("Hit@1",            "hit@1",              True),
        (f"Hit@{top_k}",     f"hit@{top_k}",       True),
        ("MRR",              "MRR",                True),
        ("Avg BLEU Top-1",   "avg_bleu_top1",      True),
        ("Avg BLEU Oracle",  "avg_bleu_oracle",    True),
        ("Corpus BLEU",      "corpus_bleu",        True),
    ]

    # ── Colours ───────────────────────────────────────────────────────────────
    BG          = "#1e1e2e"   # dark background
    HEADER_BG   = "#313244"   # slightly lighter for headers
    ROW_ODD     = "#262637"
    ROW_EVEN    = "#1e1e2e"
    TEXT_FG     = "#cdd6f4"   # light lavender text
    HEADER_FG   = "#89b4fa"   # blue header text
    METRIC_FG   = "#a6adc8"   # dim metric label
    BETTER_BG   = "#1e3a2f"   # dark green highlight cell bg
    BETTER_FG   = "#a6e3a1"   # green text for better score
    NEUTRAL_FG  = "#cdd6f4"   # normal score text
    TITLE_FG    = "#cba6f7"   # purple title
    EN_COL      = "#89dceb"   # cyan for English column
    FIL_COL     = "#f38ba8"   # pink/red for Tagalog column
    BORDER      = "#45475a"

    # ── Window ────────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("BLEU Evaluation — Summary")
    root.configure(bg=BG)
    root.resizable(False, False)

    # centre on screen
    root.update_idletasks()
    w, h = 640, 420
    x = (root.winfo_screenwidth()  - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    # ── Title ─────────────────────────────────────────────────────────────────
    title_frame = tk.Frame(root, bg=HEADER_BG, pady=12)
    title_frame.pack(fill="x")

    tk.Label(
        title_frame,
        text="Predictive Text  ·  BLEU Evaluation Summary",
        bg=HEADER_BG, fg=TITLE_FG,
        font=("Segoe UI", 13, "bold"),
    ).pack()
    tk.Label(
        title_frame,
        text="English  vs  Tagalog (Filipino)  —  N-gram NLP Model",
        bg=HEADER_BG, fg=METRIC_FG,
        font=("Segoe UI", 9),
    ).pack()

    # ── Table frame ───────────────────────────────────────────────────────────
    table_outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
    table_outer.pack(fill="both", expand=True, padx=20, pady=(12, 8))

    table = tk.Frame(table_outer, bg=BG)
    table.pack(fill="both", expand=True)

    col_widths = [22, 14, 14]   # metric | english | tagalog  (chars)

    def cell(parent, text, row, col, bg, fg, bold=False, anchor="center"):
        font_spec = ("Segoe UI", 10, "bold") if bold else ("Segoe UI", 10)
        pad_x = 10
        frm = tk.Frame(parent, bg=BORDER)
        frm.grid(row=row, column=col, sticky="nsew", padx=(0,1), pady=(0,1))
        lbl = tk.Label(
            frm, text=text,
            bg=bg, fg=fg,
            font=font_spec,
            anchor=anchor,
            padx=pad_x, pady=7,
            width=col_widths[col],
        )
        lbl.pack(fill="both", expand=True)
        return lbl

    # column headers
    cell(table, "Metric",          0, 0, HEADER_BG, HEADER_FG, bold=True, anchor="w")
    cell(table, "🇺🇸  English",     0, 1, HEADER_BG, EN_COL,    bold=True)
    cell(table, "🇵🇭  Tagalog",     0, 2, HEADER_BG, FIL_COL,   bold=True)

    # data rows
    for i, (label, key, is_pct) in enumerate(rows):
        ev  = en_metrics.get(key,  0.0)
        fv  = fil_metrics.get(key, 0.0)
        row_bg = ROW_ODD if i % 2 == 0 else ROW_EVEN

        # format values
        if is_pct:
            ev_str = f"{ev * 100:.2f}%"
            fv_str = f"{fv * 100:.2f}%"
        else:
            ev_str = f"{int(ev):,}"
            fv_str = f"{int(fv):,}"

        # determine which is better (higher = better for all metrics)
        en_is_better  = is_pct and ev > fv
        fil_is_better = is_pct and fv > ev

        cell(table, label,  i+1, 0,
             row_bg, METRIC_FG, anchor="w")
        cell(table, ev_str, i+1, 1,
             BETTER_BG if en_is_better  else row_bg,
             BETTER_FG if en_is_better  else NEUTRAL_FG,
             bold=en_is_better)
        cell(table, fv_str, i+1, 2,
             BETTER_BG if fil_is_better else row_bg,
             BETTER_FG if fil_is_better else NEUTRAL_FG,
             bold=fil_is_better)

    for c in range(3):
        table.columnconfigure(c, weight=1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_frame = tk.Frame(root, bg=BG)
    legend_frame.pack(pady=(0, 6))

    legend_box = tk.Frame(legend_frame, bg=BETTER_BG, width=14, height=14)
    legend_box.pack(side="left", padx=(0, 6))
    legend_box.pack_propagate(False)

    tk.Label(
        legend_frame,
        text="= higher score (better performance)",
        bg=BG, fg=METRIC_FG,
        font=("Segoe UI", 9),
    ).pack(side="left")

    # ── Close button ──────────────────────────────────────────────────────────
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=(0, 14))

    tk.Button(
        btn_frame,
        text="  Close  ",
        bg=HEADER_BG, fg=TEXT_FG,
        activebackground=BORDER,
        activeforeground=TEXT_FG,
        relief="flat",
        font=("Segoe UI", 10),
        cursor="hand2",
        command=root.destroy,
        padx=16, pady=6,
    ).pack()

    root.mainloop()


def save_results(
    output_path: str,
    args,
    en_metrics,
    fil_metrics,
    top_k:       int,
):
    """
    Write a structured JSON file documenting every test case and the
    aggregate metrics for each language evaluated.

    Schema
    ------
    {
      "meta": { run parameters },
      "english": {
        "summary": { aggregate metrics },
        "context_breakdown": { ... },
        "test_cases": [ { per-case record }, ... ]
      },
      "tagalog": { same structure },
      "comparison": { side-by-side summary }   // only if both languages run
    }
    """
    import datetime

    def _strip(metrics):
        summary = {}
        for k, v in metrics.items():
            if k in ("samples", "all_records", "breakdown"):
                continue
            summary[k] = round(v, 6) if isinstance(v, (int, float)) else v
        breakdown  = metrics.get("breakdown", {})
        test_cases = metrics.get("all_records", [])
        return summary, breakdown, test_cases

    doc = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "top_k":        top_k,
            "min_seq_len":  args.min_seq_len,
            "max_cases":    args.max_cases,
            "languages":    args.lang,
            "prediction_language": resolve_prediction_language(args.prediction_language),
            "prediction_language_source": args.prediction_language,
            "model_cache":  "ngram_model_standalone.json",
        }
    }

    if en_metrics:
        summary, breakdown, test_cases = _strip(en_metrics)
        doc["english"] = {
            "summary":           summary,
            "context_breakdown": breakdown,
            "test_cases":        test_cases,
        }

    if fil_metrics:
        summary, breakdown, test_cases = _strip(fil_metrics)
        doc["tagalog"] = {
            "summary":           summary,
            "context_breakdown": breakdown,
            "test_cases":        test_cases,
        }

    if en_metrics and fil_metrics:
        metric_keys = [
            ("hit@1",           "hit_at_1"),
            (f"hit@{top_k}",    f"hit_at_{top_k}"),
            ("MRR",             "mrr"),
            ("avg_bleu_top1",   "avg_bleu_top1"),
            ("avg_bleu_oracle", "avg_bleu_oracle"),
            ("corpus_bleu",     "corpus_bleu"),
        ]
        comparison = {}
        for src_key, out_key in metric_keys:
            ev = en_metrics.get(src_key, 0.0)
            fv = fil_metrics.get(src_key, 0.0)
            winner = "english" if ev > fv else ("tagalog" if fv > ev else "tie")
            comparison[out_key] = {
                "english": round(ev, 6),
                "tagalog": round(fv, 6),
                "winner":  winner,
            }
        doc["comparison"] = comparison

    with open(output_path, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(doc, f, indent=2, ensure_ascii=False)

    total_cases = (
        (en_metrics["n_cases"]  if en_metrics  else 0) +
        (fil_metrics["n_cases"] if fil_metrics else 0)
    )
    print(f"\n  \U0001f4be Results saved \u2192 {output_path}")
    print(f"     ({total_cases:,} test cases documented)")


def save_graph(
    output_path: str,
    en_metrics,
    fil_metrics,
    top_k: int,
    lang: str,
    max_line_cases: int | None = 40,
):
    """
    Save a PNG chart from the current BLEU run:
      - one aggregate bar chart
      - one per-case BLEU line chart
    """
    if not output_path:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  ! matplotlib not installed - skipping graph output.")
        print("     Install it with: pip install matplotlib")
        return

    groups = []
    if en_metrics:
        groups.append(("English", en_metrics, "#3b82f6"))
    if fil_metrics:
        groups.append(("Tagalog", fil_metrics, "#ef4444"))

    if not groups:
        print("\n  ! No metrics available - skipping graph output.")
        return

    lang_title = {
        "english": "English",
        "tagalog": "Tagalog",
        "both": "English + Tagalog",
    }.get(lang, lang.capitalize())

    aggregate_path = graph_variant_path(output_path, "aggregate")
    per_case_path = graph_variant_path(output_path, "per_case")

    fig_bar, ax_bar = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    fig_bar.suptitle(
        f"Predictive Text BLEU Aggregate Scores - {lang_title} (Top-{top_k})",
        fontsize=15,
        fontweight="bold",
    )

    metric_keys = [
        ("Hit@1", "hit@1"),
        (f"Hit@{top_k}", f"hit@{top_k}"),
        ("MRR", "MRR"),
        ("BLEU Top-1", "avg_bleu_top1"),
        ("BLEU Oracle", "avg_bleu_oracle"),
        ("Corpus BLEU", "corpus_bleu"),
    ]
    x = list(range(len(metric_keys)))
    bar_width = 0.36 if len(groups) > 1 else 0.5
    offsets = [-bar_width / 2, bar_width / 2] if len(groups) > 1 else [0]

    for offset, (label, metrics, color) in zip(offsets, groups):
        values = [metrics.get(key, 0.0) * 100.0 for _, key in metric_keys]
        bars = ax_bar.bar(
            [i + offset for i in x],
            values,
            width=bar_width,
            label=label,
            color=color,
            alpha=0.88,
        )
        for bar in bars:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_bar.set_title("Aggregate Metrics")
    ax_bar.set_ylabel("Score")
    ax_bar.set_ylim(0, max(1.0, ax_bar.get_ylim()[1] * 1.15))
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([label for label, _ in metric_keys], rotation=20, ha="right")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bar.legend()
    fig_bar.savefig(aggregate_path, dpi=160)
    plt.close(fig_bar)

    fig_line, ax_line = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    plotted = False
    for label, metrics, color in groups:
        records = metrics.get("all_records", [])
        if max_line_cases:
            records = records[:max_line_cases]
        if not records:
            continue
        plotted = True
        xs = list(range(1, len(records) + 1))
        oracle = [r.get("bleu_oracle", 0.0) * 100.0 for r in records]
        top1 = [r.get("bleu_top1", 0.0) * 100.0 for r in records]
        ax_line.plot(xs, oracle, label=f"{label} BLEU oracle", color=color, linewidth=1.8)
        ax_line.plot(xs, top1, label=f"{label} BLEU top-1", color=color, linewidth=1.1, linestyle="--", alpha=0.75)

    if plotted:
        case_label = "all test cases" if not max_line_cases else f"first {max_line_cases} cases per language"
        ax_line.set_title(f"Per-Case BLEU Line ({case_label})")
        ax_line.set_xlabel("Test case")
        ax_line.set_ylabel("BLEU")
        ax_line.set_ylim(0, 105)
        ax_line.grid(axis="y", linestyle="--", alpha=0.25)
        ax_line.legend()
    else:
        ax_line.text(
            0.5, 0.5,
            "No test-case records available; no BLEU line to plot.",
            ha="center",
            va="center",
        )
        ax_line.set_axis_off()

    fig_line.savefig(per_case_path, dpi=160)
    plt.close(fig_line)

    print(f"\n  Aggregate graph saved -> {aggregate_path}")
    print(f"  Per-case graph saved -> {per_case_path}")


# =============================================================================
# 8. MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BLEU-based evaluation of the predictive text / next-word "
                    "prediction system for English and Tagalog."
    )
    p.add_argument(
        "--lang",
        choices=["english", "tagalog", "both"],
        default="both",
        help="Which language(s) to evaluate (default: both)",
    )
    p.add_argument(
        "--prediction-language",
        choices=["config", "both", "english", "tagalog", "filipino"],
        default="config",
        help="Language filter used by the model while predicting (default: config.py)",
    )
    p.add_argument(
        "--compare-prediction-modes",
        action="store_true",
        help="Compare tagalog, english, and both prediction modes in one table",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of top predictions to consider (default: 5)",
    )
    p.add_argument(
        "--min-seq-len",
        type=int,
        default=3,
        help="Minimum sequence length to include in test set (default: 3)",
    )
    p.add_argument(
        "--max-cases",
        "--max-words",
        type=int,
        default=None,
        dest="max_cases",
        metavar="N",
        help="Cap total test cases per language (alias: --max-words; default: unlimited)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample predictions",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Path for the JSON results file (default: bleu_test_results_<lang>.json)",
    )
    p.add_argument(
        "--graph-output",
        type=str,
        default=None,
        metavar="PNG",
        help="Path for the PNG graph file (default: bleu_graph_<lang>.png)",
    )
    p.add_argument(
        "--graph-cases",
        type=int,
        default=40,
        metavar="N",
        help="How many test cases per language to draw in the line graph (default: 40)",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip PNG graph generation",
    )
    return p.parse_args()


def resolve_prediction_language(value: str) -> str:
    """Resolve CLI/config language names to model.py's language filter values."""
    if value == "config":
        return CONFIG_PREDICTION_LANGUAGE
    return _LANG_MAP.get(value, "both")


def run_evaluation(
    model:      NgramModel,
    lang:       str,
    top_k:      int,
    min_len:    int,
    max_cases:  int | None,
    prediction_language: str,
    verbose:    bool,
) -> dict:
    display = "English" if lang == "english" else "Tagalog (Filipino)"
    print_header(f"EVALUATING: {display.upper()}")

    print(f"\n  Loading {display} corpus sequences…")
    sequences = load_sequences(lang)
    print(f"  ✓ {len(sequences):,} sequences loaded")

    cases = build_test_cases(sequences, min_len=min_len)
    print(f"  ✓ {len(cases):,} test cases generated (min_len={min_len})")

    if max_cases and len(cases) > max_cases:
        import random; random.seed(42)
        cases = random.sample(cases, max_cases)
        print(f"  ⚡ Capped to {max_cases:,} cases")

    print(f"\n  Running predictions (top-k={top_k}, prediction_language={prediction_language})…")
    metrics = evaluate(
        model,
        cases,
        top_k=top_k,
        language=prediction_language,
        verbose=verbose,
    )
    print(f"  ✓ Done")

    print_metrics(f"Results — {display}", metrics, top_k)

    print(f"\n  Computing context-length breakdown…")
    breakdown = evaluate_by_context_len(
        model,
        cases,
        top_k=top_k,
        language=prediction_language,
    )
    print_context_breakdown(breakdown)
    metrics["breakdown"] = breakdown

    if verbose:
        print_samples(metrics.get("samples", []), top_k)

    return metrics


def main():
    args = parse_args()
    prediction_language = resolve_prediction_language(args.prediction_language)

    if args.output is None and args.compare_prediction_modes:
        args.output = f"bleu_mode_comparison_{args.lang}.json"
    elif args.output is None:
        args.output = f"bleu_test_results_{args.lang}.json"
    if args.graph_output is None:
        args.graph_output = f"bleu_graph_{args.lang}.png"

    print(BOLD(CYAN("\n╔══════════════════════════════════════════════════════════════════╗")))
    print(BOLD(CYAN("║   PREDICTIVE TEXT / NEXT-WORD PREDICTION — BLEU EVALUATION      ║")))
    print(BOLD(CYAN("║   Languages: English & Tagalog (Filipino)                        ║")))
    print(BOLD(CYAN("╚══════════════════════════════════════════════════════════════════╝")))

    print(f"\n  {BOLD('Prediction language')}: {CYAN(prediction_language)}")

    print_header("LOADING MODEL")
    model = load_model()

    if args.compare_prediction_modes:
        run_prediction_mode_comparison(model, args)
        print(GREEN(BOLD("\n✓ BLEU prediction-mode comparison complete.\n")))
        return

    en_metrics  = None
    fil_metrics = None

    if args.lang in ("english", "both"):
        en_metrics = run_evaluation(
            model,
            lang="english",
            top_k=args.top_k,
            min_len=args.min_seq_len,
            max_cases=args.max_cases,
            prediction_language=prediction_language,
            verbose=args.verbose,
        )

    if args.lang in ("tagalog", "both"):
        fil_metrics = run_evaluation(
            model,
            lang="tagalog",
            top_k=args.top_k,
            min_len=args.min_seq_len,
            max_cases=args.max_cases,
            prediction_language=prediction_language,
            verbose=args.verbose,
        )

    if en_metrics and fil_metrics:
        print_comparison(en_metrics, fil_metrics, args.top_k)

    # ── Save JSON results ─────────────────────────────────────────────────────
    save_results(args.output, args, en_metrics, fil_metrics, args.top_k)

    # ── Save graph ────────────────────────────────────────────────────────────
    if not args.no_graph:
        save_graph(
            args.graph_output,
            en_metrics,
            fil_metrics,
            args.top_k,
            args.lang,
            max_line_cases=args.graph_cases,
        )

    # ── GUI summary table ─────────────────────────────────────────────────────
    if en_metrics or fil_metrics:
        print(f"\n  📊 Opening summary table window…")
        show_summary_table(en_metrics, fil_metrics, args.top_k)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_header("INTERPRETATION GUIDE")
    print("""
  Metric           Interpretation
  ────────────────────────────────────────────────────────────────────
  Hit@1            Fraction of cases where the correct word is the
                   model's #1 prediction. Higher is better.

  Hit@K            Fraction of cases where the correct word appears
                   anywhere in the top-K predictions. Measures recall
                   of the suggestion bar.

  MRR              Mean Reciprocal Rank — average of 1/rank. Rewards
                   models that put the right word near the top.

  Avg BLEU top-1   Character-level BLEU between the reference word and
                   the model's #1 prediction. Gives partial credit for
                   near-misses (same stem / prefix).

  Avg BLEU oracle  Best character-level BLEU across all top-K preds.
                   Upper bound on quality if the user scrolls through
                   suggestions.

  Corpus BLEU      Corpus-level BLEU (standard MT metric). Aggregates
                   precision across all test sentences.
  ────────────────────────────────────────────────────────────────────
  BLEU is measured on character n-grams (unigram + bigram) because
  each prediction is a single word. Values above 40% indicate strong
  prefix-level accuracy; above 70% indicates near-perfect prediction.
""")

    print(GREEN(BOLD("✓ Evaluation complete.\n")))


if __name__ == "__main__":
    main()
