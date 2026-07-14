#!/usr/bin/env python3
# =============================================================================
# test_phrase_learning.py
# Tests phrase adaptation in two parts:
#   1. Live phrase memory threshold behavior
#   2. CNN phrase learning after retraining on saved phrase history
#
# Usage:
#   cd Bench/Cutted_File/files
#   python test_phrase_learning.py
#   python test_phrase_learning.py --thresholds 1,2,3 --context-words 1
#   python test_phrase_learning.py --thresholds 1,2,3 --compare-contexts 1,2
#   python test_phrase_learning.py --phrases "please adjust my pillow|i feel dizzy today"
# =============================================================================

import argparse
import json
import os
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(SCRIPT_DIR, "cnn_phrase_model.py")):
    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, SCRIPT_DIR)
else:
    print("Run this script from the project root or Bench/Cutted_File/files/.")
    sys.exit(1)

import config
import cnn_phrase_model as cnn_module
from cnn_phrase_model import CnnPhraseSuggester
from ui import FilipinoKeyboard


DEFAULT_PHRASES = [
    "please adjust my pillow",
    "i feel dizzy today",
    "call my brother please",
    "i need my glasses",
    "please open the window",
]


def phrase_tokens(phrase):
    return [token.strip().lower() for token in str(phrase).split() if token.strip()]


def phrase_context(phrase, context_words):
    tokens = phrase_tokens(phrase)
    if len(tokens) <= 1:
        return tokens
    usable = max(1, min(context_words, len(tokens) - 1))
    return tokens[:usable]


def parse_csv_numbers(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_phrases(args):
    if args.phrase_file:
        with open(args.phrase_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.phrases:
        return [part.strip() for part in args.phrases.split("|") if part.strip()]
    return DEFAULT_PHRASES


def live_memory_contains(phrase, count, threshold, context, top_k):
    old_threshold = getattr(config, "LIVE_PHRASE_MEMORY_MIN_COUNT", 1)
    try:
        config.LIVE_PHRASE_MEMORY_MIN_COUNT = threshold
        kb = FilipinoKeyboard.__new__(FilipinoKeyboard)
        kb.sentence_counts = {phrase: count}
        suggestions = kb._get_live_phrase_suggestions(context, max_results=top_k)
        predicted = [item.get("phrase", "").lower() for item in suggestions]
        return phrase.lower() in predicted, predicted
    finally:
        config.LIVE_PHRASE_MEMORY_MIN_COUNT = old_threshold


def test_live_memory(phrases, thresholds, context_words, top_k):
    records = []
    for threshold in thresholds:
        for phrase in phrases:
            context = phrase_context(phrase, context_words)
            before_count = max(0, threshold - 1)
            before_hit, before_predictions = live_memory_contains(
                phrase,
                before_count,
                threshold,
                context,
                top_k,
            )
            at_hit, at_predictions = live_memory_contains(
                phrase,
                threshold,
                threshold,
                context,
                top_k,
            )
            records.append({
                "phrase": phrase,
                "context": " ".join(context),
                "threshold": threshold,
                "count_before_threshold": before_count,
                "appeared_before_threshold": before_hit,
                "appeared_at_threshold": at_hit,
                "predictions_at_threshold": at_predictions,
            })
    return records


def test_cnn_retrain(phrases, count, context_words, top_k, epochs, learning_rate):
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        delete=False,
    )
    try:
        json.dump({phrase: count for phrase in phrases}, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()

        old_predefined_file = cnn_module._PREDEFINED_FILE
        cnn_module._PREDEFINED_FILE = temp_file.name
        try:
            model = CnnPhraseSuggester()
            ok = model.train_from_builtin(
                learning_rate=learning_rate,
                epochs=epochs,
                save_cache=False,
            )
            if not ok:
                return [], model.disabled_reason

            records = []
            for phrase in phrases:
                context = phrase_context(phrase, context_words)
                suggestions = model.get_phrase_suggestions(
                    context,
                    max_results=top_k,
                    language="both",
                )
                predicted = [item.get("phrase", "").lower() for item in suggestions]
                rank = None
                for idx, predicted_phrase in enumerate(predicted, start=1):
                    if predicted_phrase == phrase.lower():
                        rank = idx
                        break
                records.append({
                    "phrase": phrase,
                    "context": " ".join(context),
                    "appeared_after_cnn_retrain": rank is not None,
                    "rank": rank,
                    "predictions_after_retrain": predicted,
                })
            return records, ""
        finally:
            cnn_module._PREDEFINED_FILE = old_predefined_file
    finally:
        try:
            os.remove(temp_file.name)
        except Exception:
            pass


def summarize(records, key):
    if not records:
        return 0.0
    return sum(1 for record in records if record.get(key)) / len(records)


def summarize_learning_metrics(live_records, cnn_records):
    successful_live_records = [
        record for record in live_records
        if record.get("appeared_at_threshold")
    ]
    uses_to_learn = [
        record["threshold"]
        for record in successful_live_records
    ]
    return {
        "learning_success_rate": summarize(live_records, "appeared_at_threshold"),
        "retention_success_rate": summarize(cnn_records, "appeared_after_cnn_retrain"),
        "average_uses_to_learn": (
            sum(uses_to_learn) / len(uses_to_learn)
            if uses_to_learn else 0.0
        ),
        "minimum_uses_to_learn": min(uses_to_learn) if uses_to_learn else None,
        "maximum_uses_to_learn": max(uses_to_learn) if uses_to_learn else None,
    }


def print_learning_metrics(title, metrics):
    min_uses = metrics["minimum_uses_to_learn"]
    max_uses = metrics["maximum_uses_to_learn"]
    if min_uses is None:
        uses_text = "-"
    elif min_uses == max_uses:
        uses_text = str(min_uses)
    else:
        uses_text = f"{min_uses}-{max_uses}"

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Learning Success Rate  : {metrics['learning_success_rate'] * 100:.2f}%")
    print(f"Retention Success Rate : {metrics['retention_success_rate'] * 100:.2f}%")
    print(f"Average Uses-to-Learn  : {metrics['average_uses_to_learn']:.2f}")
    print(f"Uses-to-Learn Range    : {uses_text}")


def print_live_table(records):
    print("\nLive Phrase Memory Threshold Test")
    print("-" * 106)
    print(
        f"{'Threshold':>9} {'Phrase':<30} {'Context':<16} "
        f"{'Before':>8} {'At Threshold':>13}"
    )
    print("-" * 106)
    for record in records:
        print(
            f"{record['threshold']:>9} "
            f"{record['phrase'][:30]:<30} "
            f"{record['context'][:16]:<16} "
            f"{'Yes' if record['appeared_before_threshold'] else 'No':>8} "
            f"{'Yes' if record['appeared_at_threshold'] else 'No':>13}"
        )
    print("-" * 106)
    print(f"Live success at threshold: {summarize(records, 'appeared_at_threshold') * 100:.2f}%")


def print_cnn_table(records):
    print("\nCNN Retraining Learned Phrase Test")
    print("-" * 106)
    print(f"{'Phrase':<30} {'Context':<16} {'Appeared':>10} {'Rank':>6} {'Top Predictions':<35}")
    print("-" * 106)
    for record in records:
        predictions = ", ".join(record["predictions_after_retrain"][:3]) or "-"
        print(
            f"{record['phrase'][:30]:<30} "
            f"{record['context'][:16]:<16} "
            f"{'Yes' if record['appeared_after_cnn_retrain'] else 'No':>10} "
            f"{record['rank'] or '-':>6} "
            f"{predictions[:35]:<35}"
        )
    print("-" * 106)
    print(
        "CNN retrain success rate: "
        f"{summarize(records, 'appeared_after_cnn_retrain') * 100:.2f}%"
    )


def run_learning_test(phrases, thresholds, context_words, top_k, cnn_count, epochs, learning_rate):
    live_records = test_live_memory(
        phrases,
        thresholds,
        context_words,
        top_k,
    )
    cnn_records, cnn_error = test_cnn_retrain(
        phrases,
        count=cnn_count,
        context_words=context_words,
        top_k=top_k,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    metrics = summarize_learning_metrics(live_records, cnn_records)
    return {
        "context_words": context_words,
        "learning_metrics": metrics,
        "live_memory": {
            "success_rate": summarize(live_records, "appeared_at_threshold"),
            "records": live_records,
        },
        "cnn_retraining": {
            "success_rate": summarize(cnn_records, "appeared_after_cnn_retrain"),
            "error": cnn_error,
            "records": cnn_records,
        },
    }


def print_context_comparison(results):
    print("\nPhrase Learning Metrics by Context Length")
    print("-" * 105)
    print(
        f"{'Context Words':>13} {'Learning Success':>18} "
        f"{'Retention Success':>19} {'Avg Uses':>10} {'Uses Range':>12} {'CNN Error':<18}"
    )
    print("-" * 105)
    for result in results:
        error = result["cnn_retraining"].get("error") or "-"
        metrics = result["learning_metrics"]
        min_uses = metrics["minimum_uses_to_learn"]
        max_uses = metrics["maximum_uses_to_learn"]
        if min_uses is None:
            uses_range = "-"
        elif min_uses == max_uses:
            uses_range = str(min_uses)
        else:
            uses_range = f"{min_uses}-{max_uses}"
        print(
            f"{result['context_words']:>13} "
            f"{metrics['learning_success_rate'] * 100:>17.2f}% "
            f"{metrics['retention_success_rate'] * 100:>18.2f}% "
            f"{metrics['average_uses_to_learn']:>10.2f} "
            f"{uses_range:>12} "
            f"{error[:18]:<18}"
        )
    print("-" * 105)


def save_learning_graph(results, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\nCould not save graph: matplotlib unavailable ({exc})")
        return

    labels = [f"{result['context_words']} context word" if result["context_words"] == 1
              else f"{result['context_words']} context words"
              for result in results]
    live = [result["live_memory"]["success_rate"] * 100 for result in results]
    cnn = [result["cnn_retraining"]["success_rate"] * 100 for result in results]

    x = list(range(len(labels)))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    live_bars = ax.bar([i - width / 2 for i in x], live, width, label="Live Memory", color="#10b981")
    cnn_bars = ax.bar([i + width / 2 for i in x], cnn, width, label="CNN Retrain", color="#3b82f6")

    ax.set_title("Learned Phrase Adaptation Performance")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for group in (live_bars, cnn_bars):
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


def save_learning_metrics_table(results, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\nCould not save metrics table image: matplotlib unavailable ({exc})")
        return

    columns = [
        "Context Words",
        "Learning Success Rate",
        "Retention Success Rate",
        "Average Uses-to-Learn",
    ]
    rows = []
    for result in results:
        metrics = result["learning_metrics"]
        rows.append([
            str(result["context_words"]),
            f"{metrics['learning_success_rate'] * 100:.1f}%",
            f"{metrics['retention_success_rate'] * 100:.1f}%",
            f"{metrics['average_uses_to_learn']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(9.5, 2.8), dpi=180)
    ax.axis("off")
    ax.set_title(
        "Phrase Learning Metrics Summary",
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
        colWidths=[0.18, 0.28, 0.28, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

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
    print(f"Saved metrics table to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test live phrase memory and CNN retraining for newly learned phrases."
    )
    parser.add_argument(
        "--phrases",
        default=None,
        help='Pipe-separated phrases, e.g. "please adjust my pillow|i feel dizzy today".',
    )
    parser.add_argument(
        "--phrase-file",
        default=None,
        help="Optional text file with one test phrase per line.",
    )
    parser.add_argument(
        "--thresholds",
        default="1,2,3",
        help="Comma-separated live memory thresholds to test.",
    )
    parser.add_argument(
        "--context-words",
        type=int,
        default=1,
        help="Number of starting words used as context.",
    )
    parser.add_argument(
        "--compare-contexts",
        default=None,
        help="Comma-separated context lengths to compare, e.g. 1,2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of suggestions to check.",
    )
    parser.add_argument(
        "--cnn-count",
        type=int,
        default=None,
        help="Saved phrase count used for temporary CNN retraining. Defaults to max threshold.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epoch count for temporary CNN retraining.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for temporary CNN retraining.",
    )
    parser.add_argument(
        "--output",
        default="phrase_learning_results.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--graph-output",
        default="phrase_learning_graph.png",
        help="PNG graph output path.",
    )
    parser.add_argument(
        "--metrics-table-output",
        default="phrase_learning_metrics_table.png",
        help="PNG table output path for the three phrase-learning metrics.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip PNG graph generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phrases = parse_phrases(args)
    thresholds = parse_csv_numbers(args.thresholds)
    top_k = max(1, args.top_k)
    cnn_count = args.cnn_count if args.cnn_count is not None else max(thresholds)
    epochs = args.epochs or getattr(config, "CNN_PHRASE_EPOCHS", 12)
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else getattr(config, "CNN_PHRASE_LEARNING_RATE", 0.003)
    )

    if args.compare_contexts:
        context_values = parse_csv_numbers(args.compare_contexts)
        results = []
        for context_words in context_values:
            print(f"\n=== Context Words: {context_words} ===")
            result = run_learning_test(
                phrases,
                thresholds,
                context_words,
                top_k,
                cnn_count,
                epochs,
                learning_rate,
            )
            print_live_table(result["live_memory"]["records"])
            if result["cnn_retraining"]["error"]:
                print(f"\nCNN retraining test skipped: {result['cnn_retraining']['error']}")
            else:
                print_cnn_table(result["cnn_retraining"]["records"])
            print_learning_metrics(
                f"Learning Metrics ({context_words} Context Words)",
                result["learning_metrics"],
            )
            results.append(result)
        print_context_comparison(results)

        payload = {
            "settings": {
                "phrases": phrases,
                "thresholds": thresholds,
                "compare_contexts": context_values,
                "top_k": top_k,
                "cnn_count": cnn_count,
                "epochs": epochs,
                "learning_rate": learning_rate,
            },
            "context_comparison": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved context comparison to {args.output}")
        if not args.no_graph:
            save_learning_graph(results, args.graph_output)
            save_learning_metrics_table(results, args.metrics_table_output)
        return

    result = run_learning_test(
        phrases,
        thresholds,
        args.context_words,
        top_k,
        cnn_count,
        epochs,
        learning_rate,
    )
    live_records = result["live_memory"]["records"]
    cnn_records = result["cnn_retraining"]["records"]
    cnn_error = result["cnn_retraining"]["error"]

    print_live_table(live_records)
    if cnn_error:
        print(f"\nCNN retraining test skipped: {cnn_error}")
    else:
        print_cnn_table(cnn_records)
    print_learning_metrics(
        f"Learning Metrics ({args.context_words} Context Words)",
        result["learning_metrics"],
    )

    payload = {
        "settings": {
            "phrases": phrases,
            "thresholds": thresholds,
            "context_words": args.context_words,
            "top_k": top_k,
            "cnn_count": cnn_count,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "learning_metrics": result["learning_metrics"],
        "live_memory": {
            "success_rate": summarize(live_records, "appeared_at_threshold"),
            "records": live_records,
        },
        "cnn_retraining": {
            "success_rate": result["cnn_retraining"]["success_rate"],
            "error": cnn_error,
            "records": cnn_records,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {args.output}")
    if not args.no_graph:
        save_learning_graph([result], args.graph_output)
        save_learning_metrics_table([result], args.metrics_table_output)


if __name__ == "__main__":
    main()
