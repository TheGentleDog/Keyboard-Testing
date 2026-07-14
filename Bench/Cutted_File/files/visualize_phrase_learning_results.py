#!/usr/bin/env python3

import argparse
import json
import os


def load_context_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contexts = []
    live_rates = []
    cnn_rates = []
    for row in data.get("context_comparison", []):
        contexts.append(int(row["context_words"]))
        live_rates.append(float(row["live_memory"]["success_rate"]) * 100)
        cnn_rates.append(float(row["cnn_retraining"]["success_rate"]) * 100)
    return contexts, live_rates, cnn_rates


def load_per_case_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phrases = data.get("settings", {}).get("phrases", [])
    rows = []
    for context_row in data.get("context_comparison", []):
        live_records = context_row["live_memory"].get("records", [])
        cnn_records = context_row["cnn_retraining"].get("records", [])

        live_by_phrase = {}
        for phrase in phrases:
            phrase_records = [r for r in live_records if r.get("phrase") == phrase]
            live_by_phrase[phrase] = 100 if all(
                r.get("appeared_at_threshold", False)
                for r in phrase_records
            ) else 0

        cnn_by_phrase = {
            r.get("phrase"): 100 if r.get("appeared_after_cnn_retrain", False) else 0
            for r in cnn_records
        }

        rows.append({
            "context_words": int(context_row["context_words"]),
            "live": [live_by_phrase.get(phrase, 0) for phrase in phrases],
            "cnn": [cnn_by_phrase.get(phrase, 0) for phrase in phrases],
        })
    return phrases, rows


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_line_graph(contexts, live_rates, cnn_rates, output_path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)

    ax.plot(contexts, live_rates, marker="o", linewidth=2.5, color="#10b981", label="Live Memory")
    ax.plot(contexts, cnn_rates, marker="o", linewidth=2.5, color="#3b82f6", label="CNN Retraining")

    for x, y in zip(contexts, live_rates):
        ax.text(x, y + 2, f"{y:.1f}%", ha="center", fontsize=9)
    for x, y in zip(contexts, cnn_rates):
        ax.text(x, y + 2, f"{y:.1f}%", ha="center", fontsize=9)

    ax.set_title("Learned Phrase Success Rate by Context Length")
    ax.set_xlabel("Context Words")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(contexts)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_heatmap(contexts, live_rates, cnn_rates, output_path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=180)
    values = [live_rates, cnn_rates]
    labels = ["Live Memory", "CNN Retraining"]

    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title("Learned Phrase Adaptation Heatmap")
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([f"{context} Word" if context == 1 else f"{context} Words" for context in contexts])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for y, row in enumerate(values):
        for x, value in enumerate(row):
            color = "white" if value >= 70 else "#111827"
            ax.text(x, y, f"{value:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    fig.colorbar(image, ax=ax, label="Success Rate (%)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_summary_table(contexts, live_rates, cnn_rates, output_path):
    plt = setup_matplotlib()
    columns = ["Method"] + [
        f"{context} Word Context" if context == 1 else f"{context} Words Context"
        for context in contexts
    ]
    rows = [
        ["Live Memory"] + [f"{value:.1f}%" for value in live_rates],
        ["CNN Retraining"] + [f"{value:.1f}%" for value in cnn_rates],
    ]

    fig, ax = plt.subplots(figsize=(8, 2.7), dpi=180)
    ax.axis("off")
    ax.set_title("Learned Phrase Adaptation Summary", fontsize=14, fontweight="bold", pad=14)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.34, 0.33, 0.33],
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


def save_per_case_line_graph(phrases, rows, output_path):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(10, 4.2 * len(rows)),
        dpi=160,
        sharex=True,
    )
    if len(rows) == 1:
        axes = [axes]

    x = list(range(1, len(phrases) + 1))
    labels = [
        "\n".join(phrase.split()[:2]) + ("\n..." if len(phrase.split()) > 2 else "")
        for phrase in phrases
    ]

    for ax, row in zip(axes, rows):
        context = row["context_words"]
        context_label = "1 Word Context" if context == 1 else f"{context} Words Context"
        ax.plot(x, row["live"], marker="o", linewidth=2.5, color="#10b981", label="Live Memory")
        ax.plot(x, row["cnn"], marker="o", linewidth=2.5, color="#3b82f6", label="CNN Retraining")

        for idx, value in zip(x, row["live"]):
            ax.text(idx, min(value + 4, 104), f"{value:.0f}%", ha="center", fontsize=8)
        for idx, value in zip(x, row["cnn"]):
            offset = -9 if value == 100 else 4
            ax.text(idx, value + offset, f"{value:.0f}%", ha="center", fontsize=8)

        ax.set_title(f"Per-Test-Case Learning Success ({context_label})")
        ax.set_ylabel("Success (%)")
        ax.set_ylim(-5, 110)
        ax.set_yticks([0, 50, 100])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="lower right")

    axes[-1].set_xlabel("Phrase Test Case")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_slope_chart(contexts, live_rates, cnn_rates, output_path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)

    series = [
        ("Live Memory", live_rates, "#10b981"),
        ("CNN Retraining", cnn_rates, "#3b82f6"),
    ]
    x = [0, 1]
    x_labels = [
        f"{contexts[0]} Word Context",
        f"{contexts[-1]} Words Context",
    ]

    for label, rates, color in series:
        y = [rates[0], rates[-1]]
        ax.plot(x, y, marker="o", linewidth=2.5, color=color)
        ax.text(x[0] + 0.03, y[0], f"{label} {y[0]:.1f}%", ha="left", va="center", fontsize=9)
        ax.text(x[1] + 0.03, y[1], f"{label} {y[1]:.1f}%", ha="left", va="center", fontsize=9)

    ax.set_title("Change in Learned Phrase Success Rate")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Create phrase learning result visualizations.")
    parser.add_argument("--input", default="phrase_learning_results.json")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    contexts, live_rates, cnn_rates = load_context_results(args.input)
    phrases, per_case_rows = load_per_case_results(args.input)

    outputs = {
        "line": os.path.join(args.output_dir, "phrase_learning_line_graph.png"),
        "heatmap": os.path.join(args.output_dir, "phrase_learning_heatmap.png"),
        "table": os.path.join(args.output_dir, "phrase_learning_summary_table.png"),
        "slope": os.path.join(args.output_dir, "phrase_learning_slope_chart.png"),
        "per_case_line": os.path.join(args.output_dir, "phrase_learning_per_case_line_graph.png"),
    }
    save_line_graph(contexts, live_rates, cnn_rates, outputs["line"])
    save_heatmap(contexts, live_rates, cnn_rates, outputs["heatmap"])
    save_summary_table(contexts, live_rates, cnn_rates, outputs["table"])
    save_slope_chart(contexts, live_rates, cnn_rates, outputs["slope"])
    save_per_case_line_graph(phrases, per_case_rows, outputs["per_case_line"])

    for label, path in outputs.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
