"""
pixel_error.py - reusable pixel error metrics for gaze/pointer accuracy tests.

Pass target coordinates and measured gaze/pointer landing coordinates as lists
of (x, y) pairs. The calculator validates each pair, skips invalid samples, and
returns a structured report that can be printed or consumed by other code.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from itertools import zip_longest
from typing import Iterable, Optional, Sequence


Coordinate = Sequence[float]


def _parse_coordinate(value) -> Optional[tuple[float, float]]:
    """Return (x, y) as floats, or None when the coordinate is missing/invalid."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def calculate_pixel_error_report(
    targets: Iterable[Coordinate],
    gaze_points: Iterable[Coordinate],
) -> dict:
    """
    Calculate per-sample pixel errors and summary statistics.

    Args:
        targets: Iterable of target coordinates in (x, y) format.
        gaze_points: Iterable of measured gaze/pointer coordinates in (x, y) format.

    Returns:
        Dictionary with valid sample errors, invalid sample details, and summary
        metrics. Empty or fully invalid input returns zero valid samples and
        None for aggregate statistics.
    """
    pixel_errors = []
    sample_errors = []
    invalid_samples = []

    missing = object()
    for sample_index, (target_raw, gaze_raw) in enumerate(
        zip_longest(targets, gaze_points, fillvalue=missing),
        start=1,
    ):
        if target_raw is missing:
            invalid_samples.append({
                "sample_index": sample_index,
                "reason": "missing target coordinate",
                "target": None,
                "gaze": gaze_raw,
            })
            continue
        if gaze_raw is missing:
            invalid_samples.append({
                "sample_index": sample_index,
                "reason": "missing gaze coordinate",
                "target": target_raw,
                "gaze": None,
            })
            continue

        target = _parse_coordinate(target_raw)
        gaze = _parse_coordinate(gaze_raw)
        if target is None or gaze is None:
            invalid_samples.append({
                "sample_index": sample_index,
                "reason": "coordinate missing, malformed, or not numeric",
                "target": target_raw,
                "gaze": gaze_raw,
            })
            continue

        error_px = math.hypot(gaze[0] - target[0], gaze[1] - target[1])
        pixel_errors.append(error_px)
        sample_errors.append({
            "sample_index": sample_index,
            "target": target,
            "gaze": gaze,
            "error_px": error_px,
        })

    valid_sample_count = len(pixel_errors)
    if valid_sample_count == 0:
        mean_error = None
        median_error = None
        std_dev_error = None
        min_error = None
        max_error = None
    else:
        mean_error = statistics.mean(pixel_errors)
        median_error = statistics.median(pixel_errors)
        std_dev_error = statistics.pstdev(pixel_errors)
        min_error = min(pixel_errors)
        max_error = max(pixel_errors)

    return {
        "sample_errors": sample_errors,
        "pixel_errors": pixel_errors,
        "mean_error": mean_error,
        "median_error": median_error,
        "std_dev_error": std_dev_error,
        "min_error": min_error,
        "max_error": max_error,
        "valid_sample_count": valid_sample_count,
        "invalid_sample_count": len(invalid_samples),
        "invalid_samples": invalid_samples,
    }


def print_pixel_error_summary(report: dict) -> None:
    """Print a readable summary from calculate_pixel_error_report()."""
    print("Pixel Error Summary")
    print(f"  Valid samples : {report['valid_sample_count']}")
    print(f"  Invalid samples: {report['invalid_sample_count']}")

    if report["valid_sample_count"] == 0:
        print("  No valid samples to summarize.")
        return

    for item in report["sample_errors"]:
        print(
            f"  Sample {item['sample_index']}: "
            f"target={item['target']} gaze={item['gaze']} "
            f"error={item['error_px']:.2f}px"
        )

    print(f"  Mean error    : {report['mean_error']:.2f}px")
    print(f"  Median error  : {report['median_error']:.2f}px")
    print(f"  Std dev error : {report['std_dev_error']:.2f}px")
    print(f"  Min error     : {report['min_error']:.2f}px")
    print(f"  Max error     : {report['max_error']:.2f}px")


def save_pixel_error_graph(report: dict, output_path: str = "pixel_error_graph.png") -> Optional[str]:
    """
    Save a PNG graph of pixel error results.

    The graph shows each valid target/sample as a bar, with horizontal lines for
    mean and median error. Returns the saved path, or None if graph generation is
    skipped because there are no valid samples or matplotlib is unavailable.
    """
    if report.get("valid_sample_count", 0) == 0:
        print("[Info] No valid pixel error samples; graph output skipped.")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn] matplotlib is not installed; graph output skipped.")
        print("       Install it with: pip install matplotlib")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_errors = report["sample_errors"]
    sample_labels = [str(item["sample_index"]) for item in sample_errors]
    pixel_errors = [item["error_px"] for item in sample_errors]
    mean_error = report["mean_error"]
    median_error = report["median_error"]

    fig_width = max(8.0, min(16.0, 0.55 * len(pixel_errors) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6), constrained_layout=True)

    bars = ax.bar(sample_labels, pixel_errors, color="#2f80ed", edgecolor="#1b4f9c", linewidth=0.8)
    ax.axhline(mean_error, color="#d64545", linestyle="--", linewidth=1.6, label=f"Mean: {mean_error:.2f}px")
    ax.axhline(median_error, color="#2f9e44", linestyle=":", linewidth=1.8, label=f"Median: {median_error:.2f}px")

    ax.set_title("Gaze Pixel Error per Target")
    ax.set_xlabel("Target / sample number")
    ax.set_ylabel("Pixel error (px)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    max_error = max(pixel_errors)
    label_threshold = max_error * 0.08
    for bar, error in zip(bars, pixel_errors):
        if len(pixel_errors) <= 30 or error >= label_threshold:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{error:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if len(pixel_errors) > 14 else 0,
            )

    summary = (
        f"Valid: {report['valid_sample_count']} | "
        f"Std dev: {report['std_dev_error']:.2f}px | "
        f"Min: {report['min_error']:.2f}px | Max: {report['max_error']:.2f}px"
    )
    ax.text(0.01, -0.18, summary, transform=ax.transAxes, fontsize=9, color="#444444")

    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"[Info] Pixel error graph saved: {output}")
    return str(output)


def _json_safe(value):
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def save_pixel_error_data(
    report: dict,
    json_output_path: str = "pixel_error_report.json",
    csv_output_path: str = "pixel_error_samples.csv",
) -> dict:
    """
    Save the full report as JSON and per-sample results as CSV.

    Returns a dictionary with the paths that were written.
    """
    json_path = Path(json_output_path)
    csv_path = Path(csv_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "target_x",
                "target_y",
                "gaze_x",
                "gaze_y",
                "error_px",
            ],
        )
        writer.writeheader()
        for item in report.get("sample_errors", []):
            target = _parse_coordinate(item.get("target"))
            gaze = _parse_coordinate(item.get("gaze"))
            writer.writerow({
                "sample_index": item.get("sample_index"),
                "target_x": target[0] if target is not None else "",
                "target_y": target[1] if target is not None else "",
                "gaze_x": gaze[0] if gaze is not None else "",
                "gaze_y": gaze[1] if gaze is not None else "",
                "error_px": item.get("error_px"),
            })

    print(f"[Info] Pixel error JSON saved: {json_path}")
    print(f"[Info] Pixel error CSV saved: {csv_path}")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def load_coordinate_pairs_from_csv(
    csv_path: str,
    target_x_col: str = "target_x",
    target_y_col: str = "target_y",
    gaze_x_col: str = "gaze_x",
    gaze_y_col: str = "gaze_y",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Load target and gaze coordinate lists from a CSV file.

    Expected default headers:
        target_x,target_y,gaze_x,gaze_y

    Values are returned as strings and validated by calculate_pixel_error_report().
    """
    targets = []
    gaze_points = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append((row.get(target_x_col), row.get(target_y_col)))
            gaze_points.append((row.get(gaze_x_col), row.get(gaze_y_col)))

    return targets, gaze_points


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_project_imports() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _prompt_int(prompt: str, default: int, allowed: Optional[set[int]] = None) -> int:
    while True:
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
        except EOFError:
            print(f"\nNo input available; using default: {default}")
            return default
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if allowed is not None and value not in allowed:
            print(f"Please choose one of: {', '.join(str(v) for v in sorted(allowed))}")
            continue
        return value


def _limit_targets(targets: list[Coordinate], target_count: Optional[int]) -> list[Coordinate]:
    if target_count is None or target_count <= 0 or target_count >= len(targets):
        return targets
    return targets[:target_count]


def _targets_for_accuracy_count(calibration_points: int, target_count: Optional[int]) -> list[tuple[int, int]]:
    """
    Return accuracy-test target positions.

    Counts that match a calibration layout (5, 9, 16, 25) use that exact layout.
    Other counts fall back to the selected calibration layout and take the first N
    targets, keeping custom counts possible without defining a new pattern.
    """
    if target_count in {5, 9, 16, 25}:
        layout_points = target_count
    else:
        layout_points = calibration_points

    targets = _to_pixel_targets(_normalized_targets_for_calibration(layout_points))
    return _limit_targets(targets, target_count)


def _normalized_targets_for_calibration(num_points: int) -> list[tuple[float, float]]:
    _ensure_project_imports()
    from gaze_tracker2 import CalibrationManager

    return list(CalibrationManager(num_points=num_points, spp=1).pts)


def _to_pixel_targets(normalized_targets) -> list[tuple[int, int]]:
    _ensure_project_imports()
    from gaze_tracker2 import SCREEN_H, SCREEN_W

    return [
        (int(x * SCREEN_W), int(y * SCREEN_H))
        for x, y in normalized_targets
    ]


def run_gaze_pixel_accuracy_test(
    camera_id: int = 0,
    calibration_points: int = 9,
    calibration_samples: int = 60,
    test_targets: Optional[Iterable[Coordinate]] = None,
    target_count: Optional[int] = None,
    samples_per_target: int = 30,
    hold_frames: int = 20,
    target_radius_px: int = 64,
    ema_alpha: float = 0.3,
    pnoise: float = 0.0100,
    mnoise: float = 5.5,
    graph_output_path: Optional[str] = None,
    data_output_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    Run calibration, then show target dots and calculate gaze pixel error.

    Args:
        camera_id: Camera index for OpenCV.
        calibration_points: Calibration pattern size: 5, 9, 16, or 25.
        calibration_samples: Samples collected per calibration point.
        test_targets: Optional target coordinates in pixels. When None, the
            selected calibration pattern is reused as the accuracy target layout.
        target_count: Optional maximum number of accuracy targets to include.
        samples_per_target: Valid gaze samples averaged for each test target.
        hold_frames: Frames to wait at each target before collecting samples.
        target_radius_px: Radius of the visible target ring during testing.

    Returns:
        Pixel error report, or None if calibration/test is cancelled.
    """
    _ensure_project_imports()
    import cv2
    import numpy as np
    from gaze_tracker2 import (
        C_ACCENT,
        C_BG,
        C_DOT_INNER,
        C_DOT_OUTER,
        C_GRID,
        C_TEXT,
        C_WARN,
        GazeTrackerApp,
        SCREEN_H,
        SCREEN_W,
        draw_bar,
        draw_grid,
        txt,
    )

    tracker = GazeTrackerApp(
        camera_id=camera_id,
        num_points=calibration_points,
        spp=calibration_samples,
        ema_alpha=ema_alpha,
        pnoise=pnoise,
        mnoise=mnoise,
        show_camera_window=True,
        debug_landmarks=True,
        show_distance=True,
        tutorial_enabled=True,
    )

    print("[Info] Starting calibration. The keyboard will not open after this.")
    if not tracker.calibrate():
        print("[Info] Calibration cancelled.")
        return None

    if test_targets is not None:
        targets = _limit_targets(list(test_targets), target_count)
    else:
        targets = _targets_for_accuracy_count(calibration_points, target_count)
    if not targets:
        print("[Warn] No accuracy test targets were provided.")
        if getattr(tracker, "_cap", None) is not None:
            tracker._cap.release()
        return calculate_pixel_error_report([], [])

    target_points = []
    gaze_points = []
    completed_target_points = []

    cv2.namedWindow(tracker.WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(tracker.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    tracker.smoother.reset()
    tracker.hist.clear()

    try:
        for target_index, target in enumerate(targets, start=1):
            parsed_target = _parse_coordinate(target)
            if parsed_target is None:
                target_points.append(target)
                gaze_points.append(None)
                continue

            target_x, target_y = parsed_target
            target_px = (int(target_x), int(target_y))
            target_radius = max(24, int(target_radius_px))
            target_inner_radius = max(8, int(target_radius * 0.34))
            target_marker_size = max(28, int(target_radius * 0.9))
            collected = []
            frame_count = 0
            tracker.smoother.reset()

            while len(collected) < samples_per_target:
                ret, cam = tracker._cap.read()
                if not ret:
                    print("[Warn] Camera frame read failed during accuracy test.")
                    break

                frame_count += 1
                cam = cv2.flip(cam, 1)
                res = tracker.mesh.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

                feat = lms = None
                is_blinking = False
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    feat = tracker.extractor.extract(lms, cam.shape[1], cam.shape[0])
                    is_blinking = tracker.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])

                gaze_px = None
                if feat is not None and not is_blinking:
                    raw = tracker.calib.model.predict(feat)
                    smoothed = tracker.smoother.update(raw)
                    gaze_px = (
                        int(np.clip(smoothed[0] * SCREEN_W, 0, SCREEN_W - 1)),
                        int(np.clip(smoothed[1] * SCREEN_H, 0, SCREEN_H - 1)),
                    )
                    if frame_count > hold_frames:
                        collected.append(gaze_px)

                canvas = tracker.canvas
                canvas[:] = C_BG
                draw_grid(canvas)
                for previous_x, previous_y in completed_target_points:
                    cv2.circle(canvas, (previous_x, previous_y), 8, C_GRID, 1, cv2.LINE_AA)

                cv2.circle(canvas, target_px, target_radius, C_DOT_OUTER, 3, cv2.LINE_AA)
                cv2.circle(canvas, target_px, target_inner_radius, C_DOT_INNER, -1, cv2.LINE_AA)
                cv2.drawMarker(
                    canvas,
                    target_px,
                    C_ACCENT,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=target_marker_size,
                    thickness=3,
                )

                if gaze_px is not None:
                    cv2.circle(canvas, gaze_px, 12, C_WARN, 2, cv2.LINE_AA)

                progress = len(collected) / max(samples_per_target, 1)
                draw_bar(canvas, progress, SCREEN_W // 2 - 300, SCREEN_H - 70, 600, 16)
                label = (
                    f"Pixel Error Test  |  Target {target_index}/{len(targets)}  |  "
                    f"Samples {len(collected)}/{samples_per_target}"
                )
                txt(canvas, label, (SCREEN_W // 2 - 330, SCREEN_H - 92), scale=0.62, color=C_TEXT)
                txt(canvas, "Look at the target dot. Press Q to cancel.", (SCREEN_W // 2 - 245, SCREEN_H - 24),
                    scale=0.50, color=(120, 120, 120))
                if is_blinking:
                    txt(canvas, "Blink detected - not collecting", (32, 46), scale=0.62, color=C_WARN, thick=2)
                elif feat is None:
                    txt(canvas, "No face detected", (32, 46), scale=0.62, color=C_WARN, thick=2)

                cv2.imshow(tracker.WIN, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[Info] Accuracy test cancelled.")
                    return None

            target_points.append(target_px)
            completed_target_points.append(target_px)
            if collected:
                avg_x = statistics.mean(point[0] for point in collected)
                avg_y = statistics.mean(point[1] for point in collected)
                gaze_points.append((avg_x, avg_y))
            else:
                gaze_points.append(None)
    finally:
        if getattr(tracker, "_cap", None) is not None:
            tracker._cap.release()
        cv2.destroyAllWindows()

    report = calculate_pixel_error_report(target_points, gaze_points)
    print_pixel_error_summary(report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if data_output_dir is None:
        output_dir = _project_root() / "pixel_error_results"
    else:
        output_dir = Path(data_output_dir)

    if graph_output_path is None:
        graph_output_path = str(output_dir / f"pixel_error_graph_{stamp}.png")
    graph_path = save_pixel_error_graph(report, graph_output_path)
    report["graph_path"] = graph_path
    data_paths = save_pixel_error_data(
        report,
        json_output_path=str(output_dir / f"pixel_error_report_{stamp}.json"),
        csv_output_path=str(output_dir / f"pixel_error_samples_{stamp}.csv"),
    )
    report.update(data_paths)
    return report


def interactive_gaze_pixel_accuracy_test() -> Optional[dict]:
    """Prompt for calibration settings, then run the standalone accuracy test."""
    print("Gaze Pixel Error Test")
    print("Choose the calibration pattern to use before the target test.")
    camera_id = _prompt_int("Camera index", default=0)
    calibration_points = _prompt_int("Calibration points (5, 9, 16, 25)", default=9, allowed={5, 9, 16, 25})
    calibration_samples = _prompt_int("Calibration samples per point", default=60)
    max_target_count = 25
    target_count = _prompt_int(
        "Number of accuracy targets to include",
        default=calibration_points,
    )
    if target_count > max_target_count:
        print(f"Only {max_target_count} targets are available for this layout; using {max_target_count}.")
        target_count = max_target_count
    samples_per_target = _prompt_int("Accuracy samples per target", default=30)
    hold_frames = _prompt_int("Frames to wait before collecting each target", default=20)
    target_radius_px = _prompt_int("Target radius in pixels", default=64)

    return run_gaze_pixel_accuracy_test(
        camera_id=camera_id,
        calibration_points=calibration_points,
        calibration_samples=calibration_samples,
        target_count=target_count,
        samples_per_target=samples_per_target,
        hold_frames=hold_frames,
        target_radius_px=target_radius_px,
    )


if __name__ == "__main__":
    interactive_gaze_pixel_accuracy_test()
