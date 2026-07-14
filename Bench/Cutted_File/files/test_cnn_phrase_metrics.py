#!/usr/bin/env python3
# =============================================================================
# test_cnn_phrase_metrics.py
# Evaluates the CNN phrase suggestion model using:
#   - Hit@1
#   - Hit@3
#   - MRR
#
# Usage:
#   cd Bench/Cutted_File/files
#   python test_cnn_phrase_metrics.py
#   python test_cnn_phrase_metrics.py --lang english
#   python test_cnn_phrase_metrics.py --lang tagalog --max-cases 300
#   python test_cnn_phrase_metrics.py --output cnn_phrase_metrics.json
# =============================================================================

import argparse
import json
import os
import random
import sys
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(SCRIPT_DIR, "cnn_phrase_model.py")):
    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, SCRIPT_DIR)
else:
    print("Run this script from the project root or Bench/Cutted_File/files/.")
    sys.exit(1)

import config
from cnn_phrase_model import cnn_phrase_model


LANG_DATASETS = {
    "english": ("english", config.ENGLISH_DATASET_FILE),
    "tagalog": ("filipino", config.FILIPINO_DATASET_FILE),
}

PREDICTION_LANGUAGE_MAP = {
    "config": getattr(config, "PREDICTION_LANGUAGE", "both"),
    "both": "both",
    "english": "english",
    "filipino": "filipino",
    "tagalog": "filipino",
}


def clean_token(token):
    return str(token).strip().lower()


def clean_sequence(seq):
    tokens = []
    for token in seq:
        token = clean_token(token)
        if token and (token.isalpha() or token.replace("'", "").isalpha()):
            tokens.append(token)
    return tokens


def phrase_text(tokens):
    return " ".join(tokens)


def load_dataset_sequences(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sequences = []
    for seq in data.get("corpus_sequences", []):
        tokens = clean_sequence(seq)
        if len(tokens) >= 2:
            sequences.append(tokens)
    for phrase in data.get("communication_corpus", []):
        tokens = clean_sequence(str(phrase).split())
        if len(tokens) >= 2:
            sequences.append(tokens)
    return sequences


def build_cases(langs, prefix_mode="all", max_context=5):
    cases = []
    seen = set()
    for display_lang in langs:
        model_lang, path = LANG_DATASETS[display_lang]
        for tokens in load_dataset_sequences(path):
            target = phrase_text(tokens)
            if prefix_mode == "longest":
                prefix_lengths = [min(len(tokens) - 1, max_context)]
            else:
                prefix_lengths = range(1, min(len(tokens), max_context + 1))

            for prefix_len in prefix_lengths:
                context = tokens[:prefix_len]
                if len(context) >= len(tokens):
                    continue
                key = (model_lang, phrase_text(context), target)
                if key in seen:
                    continue
                seen.add(key)
                cases.append({
                    "language": model_lang,
                    "context": context,
                    "target": target,
                })
    return cases


def evaluate_case(model, case, top_k, prediction_language):
    suggestions = model.get_phrase_suggestions(
        case["context"],
        max_results=top_k,
        language=prediction_language,
    )
    predicted = [item.get("phrase", "").lower() for item in suggestions]
    target = case["target"].lower()

    rank = None
    for idx, phrase in enumerate(predicted, start=1):
        if phrase == target:
            rank = idx
            break

    return {
        "language": case["language"],
        "context": phrase_text(case["context"]),
        "target": case["target"],
        "predictions": predicted,
        "rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_3": rank is not None and rank <= 3,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }


def evaluate_records(model, cases, top_k, prediction_language):
    return [
        evaluate_case(model, case, top_k=top_k, prediction_language=prediction_language)
        for case in cases
    ]


def summarize(records):
    total = len(records)
    if total == 0:
        return {
            "cases": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "mrr": 0.0,
        }
    return {
        "cases": total,
        "hit_at_1": sum(1 for r in records if r["hit_at_1"]) / total,
        "hit_at_3": sum(1 for r in records if r["hit_at_3"]) / total,
        "mrr": sum(r["reciprocal_rank"] for r in records) / total,
    }


def summarize_by_context_length(records):
    grouped = defaultdict(list)
    for record in records:
        context_len = len(str(record["context"]).split())
        grouped[context_len].append(record)
    return {
        context_len: summarize(context_records)
        for context_len, context_records in sorted(grouped.items())
    }


def print_summary(title, stats):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Cases : {stats['cases']}")
    print(f"Hit@1 : {stats['hit_at_1'] * 100:.2f}%")
    print(f"Hit@3 : {stats['hit_at_3'] * 100:.2f}%")
    print(f"MRR   : {stats['mrr']:.4f}")


def print_metrics_table(overall, by_language):
    rows = [("Overall", overall)]
    for language in sorted(by_language):
        rows.append((language.title(), by_language[language]))

    print("\nCNN Phrase Suggestion Metrics")
    print("-" * 66)
    print(f"{'Dataset':<14} {'Cases':>8} {'Hit@1':>12} {'Hit@3':>12} {'MRR':>12}")
    print("-" * 66)
    for label, stats in rows:
        print(
            f"{label:<14} "
            f"{stats['cases']:>8} "
            f"{stats['hit_at_1'] * 100:>11.2f}% "
            f"{stats['hit_at_3'] * 100:>11.2f}% "
            f"{stats['mrr']:>12.4f}"
        )
    print("-" * 66)


def print_context_table(by_context_length):
    print("\nCNN Phrase Metrics by Context Length")
    print("-" * 76)
    print(
        f"{'Context Words':<16} "
        f"{'Cases':>8} "
        f"{'Hit@1':>12} "
        f"{'Hit@3':>12} "
        f"{'MRR':>12}"
    )
    print("-" * 76)
    for context_len, stats in by_context_length.items():
        print(
            f"{context_len:<16} "
            f"{stats['cases']:>8} "
            f"{stats['hit_at_1'] * 100:>11.2f}% "
            f"{stats['hit_at_3'] * 100:>11.2f}% "
            f"{stats['mrr'] * 100:>11.2f}%"
        )
    print("-" * 76)


def save_metrics_graph(overall, by_language, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\nCould not save graph: matplotlib unavailable ({exc})")
        return

    rows = [("Overall", overall)]
    for language in sorted(by_language):
        rows.append((language.title(), by_language[language]))

    labels = [label for label, _ in rows]
    hit1 = [stats["hit_at_1"] * 100 for _, stats in rows]
    hit3 = [stats["hit_at_3"] * 100 for _, stats in rows]
    mrr = [stats["mrr"] * 100 for _, stats in rows]

    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    bars = [
        ax.bar([i - width for i in x], hit1, width, label="Hit@1", color="#3b82f6"),
        ax.bar(x, hit3, width, label="Hit@3", color="#10b981"),
        ax.bar([i + width for i in x], mrr, width, label="MRR", color="#f59e0b"),
    ]

    ax.set_title("CNN Phrase Suggestion Performance")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved graph to {output_path}")


def save_context_table_image(by_context_length, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\nCould not save context table image: matplotlib unavailable ({exc})")
        return

    columns = ["Context Words", "Cases", "Hit@1", "Hit@3", "MRR"]
    rows = []
    for context_len, stats in by_context_length.items():
        rows.append([
            str(context_len),
            f"{stats['cases']:,}",
            f"{stats['hit_at_1'] * 100:.1f}%",
            f"{stats['hit_at_3'] * 100:.1f}%",
            f"{stats['mrr'] * 100:.1f}%",
        ])

    fig_height = max(2.4, 0.48 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=180)
    ax.axis("off")
    ax.set_title(
        "CNN Phrase Metrics by Context Length",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.16, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f4f6")
        else:
            cell.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved context table image to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN phrase suggestions with Hit@1, Hit@3, and MRR."
    )
    parser.add_argument(
        "--lang",
        choices=["english", "tagalog", "both"],
        default="both",
        help="Dataset language to evaluate.",
    )
    parser.add_argument(
        "--prediction-language",
        choices=["config", "english", "filipino", "tagalog", "both"],
        default="config",
        help="Language filter used by the CNN model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of phrase suggestions to request. Use 3 for Hit@3.",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=["all", "longest"],
        default="all",
        help="Use all phrase prefixes or only the longest available prefix.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=500,
        help="Maximum number of test cases to sample. Use 0 for all cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling cases.",
    )
    parser.add_argument(
        "--output",
        default="cnn_phrase_metrics.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--graph-output",
        default="cnn_phrase_metrics_graph.png",
        help="PNG graph output path.",
    )
    parser.add_argument(
        "--context-table-output",
        default="cnn_phrase_metrics_context_table.png",
        help="PNG table output path for metrics grouped by context length.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip PNG graph and table image generation.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print the first 10 evaluated cases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    top_k = max(3, args.top_k)
    prediction_language = PREDICTION_LANGUAGE_MAP[args.prediction_language]

    langs = ["english", "tagalog"] if args.lang == "both" else [args.lang]
    cases = build_cases(
        langs,
        prefix_mode=args.prefix_mode,
        max_context=getattr(config, "CNN_PHRASE_MAX_CONTEXT", 5),
    )
    if args.max_cases and len(cases) > args.max_cases:
        cases = random.sample(cases, args.max_cases)

    print(f"Evaluating {len(cases)} cases with prediction_language={prediction_language}...")
    model = cnn_phrase_model
    print("Loading CNN phrase model...")
    if not cnn_phrase_model.load_or_train():
        print(f"CNN phrase model unavailable: {cnn_phrase_model.disabled_reason}")
        sys.exit(1)

    records = evaluate_records(
        model,
        cases,
        top_k=top_k,
        prediction_language=prediction_language,
    )

    overall = summarize(records)
    by_context_length = summarize_by_context_length(records)
    by_language = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["language"]].append(record)
    for language, language_records in grouped.items():
        by_language[language] = summarize(language_records)

    print_metrics_table(overall, by_language)
    print_context_table(by_context_length)

    if args.show_samples:
        print("\nSample cases")
        print("------------")
        for record in records[:10]:
            print(f"Context: {record['context']}")
            print(f"Target : {record['target']}")
            print(f"Top {top_k}: {', '.join(record['predictions']) or '-'}")
            print(f"Rank   : {record['rank'] or '-'}\n")

    payload = {
        "settings": {
            "lang": args.lang,
            "prediction_language": prediction_language,
            "top_k": top_k,
            "prefix_mode": args.prefix_mode,
            "max_cases": args.max_cases,
            "seed": args.seed,
        },
        "overall": overall,
        "by_language": by_language,
        "by_context_length": by_context_length,
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {args.output}")
    if not args.no_graph:
        save_metrics_graph(overall, by_language, args.graph_output)
        save_context_table_image(by_context_length, args.context_table_output)


if __name__ == "__main__":
    main()
