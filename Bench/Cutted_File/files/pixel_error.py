"""
Pixel error tools for gaze or pointer accuracy testing.

This module is intentionally data-driven: no final target coordinates are
hardcoded here. Supply targets and gaze points from variables, CSV files, mouse
logs, eye-tracker output, or any other project-specific source.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _detect_screen_resolution() -> tuple[int, int]:
    """Return (width, height) in physical pixels for the primary monitor."""
    # On Windows, make the process DPI-aware before asking for screen metrics.
    # Otherwise high-DPI scaling can return logical pixels instead of real pixels.
    if sys.platform == "win32":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            w = ctypes.windll.user32.GetSystemMetrics(0)
            h = ctypes.windll.user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            pass

    # Try tkinter next; available in the standard library on most systems.
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    # Fall back to screeninfo (third-party, pip install screeninfo)
    try:
        from screeninfo import get_monitors
        monitor = get_monitors()[0]
        if monitor.width and monitor.height:
            return monitor.width, monitor.height
    except Exception:
        pass

    return 0, 0


Coordinate = tuple[float, float]
RawCoordinate = Optional[Sequence[object]]


@dataclass(frozen=True)
class ScreenInfo:
    width_px: int
    height_px: int
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    viewing_distance_cm: Optional[float] = None
    display_width_px: Optional[int] = None
    display_height_px: Optional[int] = None

    @classmethod
    def detect(cls) -> "ScreenInfo":
        """Detect the primary monitor's resolution at runtime."""
        width_px, height_px = _detect_screen_resolution()
        return cls(width_px=width_px, height_px=height_px)

    def as_label(self) -> str:
        if self.width_px and self.height_px:
            return f"{self.width_px}x{self.height_px}"
        return "unknown"


@dataclass(frozen=True)
class PixelErrorConfig:
    enable_ema: bool = False
    enable_kalman_filter: bool = False
    calibration_points: int = 9
    samples_per_calibration: int = 90
    test_points: int = 9
    accuracy_samples_per_test_point: int = 30


@dataclass(frozen=True)
class PixelErrorSample:
    sample_index: int
    target: Coordinate
    gaze_point: Coordinate
    error_px: float
    error_visual_angle_deg: Optional[float] = None
    display_error_px: Optional[float] = None


@dataclass(frozen=True)
class InvalidSample:
    sample_index: int
    target: RawCoordinate
    gaze_point: RawCoordinate
    reason: str


def _parse_coordinate(value: RawCoordinate) -> tuple[Optional[Coordinate], Optional[str]]:
    if value is None:
        return None, "coordinate is missing"

    if len(value) != 2:
        return None, "coordinate must contain exactly two values: x and y"

    x_raw, y_raw = value
    if x_raw is None or y_raw is None:
        return None, "coordinate has a missing x or y value"

    try:
        x = float(x_raw)
        y = float(y_raw)
    except (TypeError, ValueError):
        return None, "coordinate x and y values must be numeric"

    if not math.isfinite(x) or not math.isfinite(y):
        return None, "coordinate x and y values must be finite numbers"

    return (x, y), None


def calculate_pixel_error(target: Coordinate, gaze_point: Coordinate) -> float:
    """Return Euclidean distance in pixels between a target and gaze point."""
    target_x, target_y = target
    gaze_x, gaze_y = gaze_point
    return math.hypot(gaze_x - target_x, gaze_y - target_y)


def calculate_display_pixel_error(
    target: Coordinate,
    gaze_point: Coordinate,
    screen_info: ScreenInfo,
) -> Optional[float]:
    """Return error scaled from canvas pixels into detected display pixels."""
    if (
        not screen_info.width_px
        or not screen_info.height_px
        or not screen_info.display_width_px
        or not screen_info.display_height_px
    ):
        return None

    target_x, target_y = target
    gaze_x, gaze_y = gaze_point
    dx_px = (gaze_x - target_x) * screen_info.display_width_px / screen_info.width_px
    dy_px = (gaze_y - target_y) * screen_info.display_height_px / screen_info.height_px
    return math.hypot(dx_px, dy_px)


def calculate_visual_angle_error(
    target: Coordinate,
    gaze_point: Coordinate,
    screen_info: ScreenInfo,
) -> Optional[float]:
    """Return gaze error as degrees of visual angle when physical scale is known."""
    if (
        not screen_info.width_px
        or not screen_info.height_px
        or not screen_info.width_cm
        or not screen_info.height_cm
        or not screen_info.viewing_distance_cm
        or screen_info.width_cm <= 0
        or screen_info.height_cm <= 0
        or screen_info.viewing_distance_cm <= 0
    ):
        return None

    target_x, target_y = target
    gaze_x, gaze_y = gaze_point
    dx_cm = (gaze_x - target_x) * screen_info.width_cm / screen_info.width_px
    dy_cm = (gaze_y - target_y) * screen_info.height_cm / screen_info.height_px
    error_cm = math.hypot(dx_cm, dy_cm)
    return math.degrees(2.0 * math.atan2(error_cm, 2.0 * screen_info.viewing_distance_cm))


def calculate_pixel_error_report(
    targets: Iterable[RawCoordinate],
    gaze_points: Iterable[RawCoordinate],
    *,
    config: Optional[PixelErrorConfig] = None,
    screen_info: Optional[ScreenInfo] = None,
    report_invalid_samples: bool = True,
) -> dict:
    """
    Calculate per-sample and summary pixel error statistics.

    Args:
        targets: Iterable of target coordinates in (x, y) format.
        gaze_points: Iterable of measured gaze coordinates in (x, y) format.
        config: Optional run settings from the menu or caller.
        screen_info: Optional monitor resolution captured at test time.
            If None, resolution is detected automatically from the primary monitor.
        report_invalid_samples: Include invalid sample details in the result.

    Returns:
        A structured dictionary containing per-sample errors, aggregate
        statistics, invalid sample information, and optional config values.
    """
    if screen_info is None:
        screen_info = ScreenInfo.detect()
    target_list = list(targets)
    gaze_list = list(gaze_points)

    valid_samples: list[PixelErrorSample] = []
    invalid_samples: list[InvalidSample] = []

    paired_count = min(len(target_list), len(gaze_list))

    for index in range(paired_count):
        raw_target = target_list[index]
        raw_gaze = gaze_list[index]

        target, target_error = _parse_coordinate(raw_target)
        gaze_point, gaze_error = _parse_coordinate(raw_gaze)

        if target_error or gaze_error:
            reasons = []
            if target_error:
                reasons.append(f"target {target_error}")
            if gaze_error:
                reasons.append(f"gaze point {gaze_error}")
            invalid_samples.append(
                InvalidSample(index, raw_target, raw_gaze, "; ".join(reasons))
            )
            continue

        error_px = calculate_pixel_error(target, gaze_point)
        error_visual_angle_deg = calculate_visual_angle_error(target, gaze_point, screen_info)
        display_error_px = calculate_display_pixel_error(target, gaze_point, screen_info)
        valid_samples.append(
            PixelErrorSample(
                index,
                target,
                gaze_point,
                error_px,
                error_visual_angle_deg,
                display_error_px,
            )
        )

    if len(target_list) != len(gaze_list):
        longer = target_list if len(target_list) > len(gaze_list) else gaze_list
        missing_side = "gaze point" if len(target_list) > len(gaze_list) else "target"
        for index in range(paired_count, len(longer)):
            raw_target = target_list[index] if index < len(target_list) else None
            raw_gaze = gaze_list[index] if index < len(gaze_list) else None
            invalid_samples.append(
                InvalidSample(
                    index,
                    raw_target,
                    raw_gaze,
                    f"missing corresponding {missing_side}",
                )
            )

    pixel_errors = [sample.error_px for sample in valid_samples]
    visual_angle_errors = [
        sample.error_visual_angle_deg
        for sample in valid_samples
        if sample.error_visual_angle_deg is not None
    ]
    display_pixel_errors = [
        sample.display_error_px
        for sample in valid_samples
        if sample.display_error_px is not None
    ]
    per_target_errors = _calculate_per_target_errors(valid_samples)
    target_pixel_errors = [target["mean_error"] for target in per_target_errors]
    target_visual_angle_errors = [
        target["mean_visual_angle_error_deg"]
        for target in per_target_errors
        if target.get("mean_visual_angle_error_deg") is not None
    ]
    target_display_pixel_errors = [
        target["mean_display_error_px"]
        for target in per_target_errors
        if target.get("mean_display_error_px") is not None
    ]
    valid_sample_count = len(pixel_errors)

    if valid_sample_count == 0:
        mean_error = None
        median_error = None
        std_dev_error = None
        min_error = None
        max_error = None
    else:
        mean_error = statistics.fmean(pixel_errors)
        median_error = statistics.median(pixel_errors)
        std_dev_error = statistics.stdev(pixel_errors) if valid_sample_count > 1 else 0.0
        min_error = min(pixel_errors)
        max_error = max(pixel_errors)

    if not visual_angle_errors:
        mean_visual_angle_error_deg = None
        median_visual_angle_error_deg = None
        std_dev_visual_angle_error_deg = None
        min_visual_angle_error_deg = None
        max_visual_angle_error_deg = None
    else:
        mean_visual_angle_error_deg = statistics.fmean(visual_angle_errors)
        median_visual_angle_error_deg = statistics.median(visual_angle_errors)
        std_dev_visual_angle_error_deg = (
            statistics.stdev(visual_angle_errors) if len(visual_angle_errors) > 1 else 0.0
        )
        min_visual_angle_error_deg = min(visual_angle_errors)
        max_visual_angle_error_deg = max(visual_angle_errors)

    if not display_pixel_errors:
        mean_display_error_px = None
        median_display_error_px = None
        std_dev_display_error_px = None
        min_display_error_px = None
        max_display_error_px = None
    else:
        mean_display_error_px = statistics.fmean(display_pixel_errors)
        median_display_error_px = statistics.median(display_pixel_errors)
        std_dev_display_error_px = (
            statistics.stdev(display_pixel_errors) if len(display_pixel_errors) > 1 else 0.0
        )
        min_display_error_px = min(display_pixel_errors)
        max_display_error_px = max(display_pixel_errors)

    result = {
        "config": asdict(config) if config else None,
        "screen_info": asdict(screen_info),
        "valid_sample_count": valid_sample_count,
        "invalid_sample_count": len(invalid_samples),
        "pixel_errors": pixel_errors,
        "display_pixel_errors": display_pixel_errors,
        "visual_angle_errors_deg": visual_angle_errors,
        "samples": [asdict(sample) for sample in valid_samples],
        "per_target_errors": per_target_errors,
        "valid_target_count": len(per_target_errors),
        "mean_error": mean_error,
        "median_error": median_error,
        "std_dev_error": std_dev_error,
        "min_error": min_error,
        "max_error": max_error,
        "mean_display_error_px": mean_display_error_px,
        "median_display_error_px": median_display_error_px,
        "std_dev_display_error_px": std_dev_display_error_px,
        "min_display_error_px": min_display_error_px,
        "max_display_error_px": max_display_error_px,
        "mean_visual_angle_error_deg": mean_visual_angle_error_deg,
        "median_visual_angle_error_deg": median_visual_angle_error_deg,
        "std_dev_visual_angle_error_deg": std_dev_visual_angle_error_deg,
        "min_visual_angle_error_deg": min_visual_angle_error_deg,
        "max_visual_angle_error_deg": max_visual_angle_error_deg,
        "target_mean_error": statistics.fmean(target_pixel_errors) if target_pixel_errors else None,
        "target_median_error": statistics.median(target_pixel_errors) if target_pixel_errors else None,
        "target_std_dev_error": (
            statistics.stdev(target_pixel_errors) if len(target_pixel_errors) > 1 else 0.0
        ) if target_pixel_errors else None,
        "target_min_error": min(target_pixel_errors) if target_pixel_errors else None,
        "target_max_error": max(target_pixel_errors) if target_pixel_errors else None,
        "target_mean_display_error_px": (
            statistics.fmean(target_display_pixel_errors) if target_display_pixel_errors else None
        ),
        "target_median_display_error_px": (
            statistics.median(target_display_pixel_errors) if target_display_pixel_errors else None
        ),
        "target_std_dev_display_error_px": (
            statistics.stdev(target_display_pixel_errors)
            if len(target_display_pixel_errors) > 1
            else 0.0
        ) if target_display_pixel_errors else None,
        "target_min_display_error_px": (
            min(target_display_pixel_errors) if target_display_pixel_errors else None
        ),
        "target_max_display_error_px": (
            max(target_display_pixel_errors) if target_display_pixel_errors else None
        ),
        "target_mean_visual_angle_error_deg": (
            statistics.fmean(target_visual_angle_errors) if target_visual_angle_errors else None
        ),
        "target_median_visual_angle_error_deg": (
            statistics.median(target_visual_angle_errors) if target_visual_angle_errors else None
        ),
        "target_std_dev_visual_angle_error_deg": (
            statistics.stdev(target_visual_angle_errors)
            if len(target_visual_angle_errors) > 1
            else 0.0
        ) if target_visual_angle_errors else None,
        "target_min_visual_angle_error_deg": (
            min(target_visual_angle_errors) if target_visual_angle_errors else None
        ),
        "target_max_visual_angle_error_deg": (
            max(target_visual_angle_errors) if target_visual_angle_errors else None
        ),
    }

    if report_invalid_samples:
        result["invalid_samples"] = [asdict(sample) for sample in invalid_samples]

    return result


def _calculate_per_target_errors(valid_samples: Sequence[PixelErrorSample]) -> list[dict]:
    grouped: dict[Coordinate, list[PixelErrorSample]] = {}
    ordered_targets: list[Coordinate] = []

    for sample in valid_samples:
        if sample.target not in grouped:
            grouped[sample.target] = []
            ordered_targets.append(sample.target)
        grouped[sample.target].append(sample)

    per_target_errors = []
    for target_index, target in enumerate(ordered_targets, start=1):
        samples = grouped[target]
        errors = [sample.error_px for sample in samples]
        visual_angle_errors = [
            sample.error_visual_angle_deg
            for sample in samples
            if sample.error_visual_angle_deg is not None
        ]
        display_pixel_errors = [
            sample.display_error_px
            for sample in samples
            if sample.display_error_px is not None
        ]
        gaze_x_values = [sample.gaze_point[0] for sample in samples]
        gaze_y_values = [sample.gaze_point[1] for sample in samples]
        mean_gaze = (statistics.fmean(gaze_x_values), statistics.fmean(gaze_y_values))
        per_target_errors.append({
            "target_number": target_index,
            "target": target,
            "sample_count": len(samples),
            "mean_gaze_point": mean_gaze,
            "mean_error": statistics.fmean(errors),
            "median_error": statistics.median(errors),
            "std_dev_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
            "min_error": min(errors),
            "max_error": max(errors),
            "mean_display_error_px": (
                statistics.fmean(display_pixel_errors) if display_pixel_errors else None
            ),
            "median_display_error_px": (
                statistics.median(display_pixel_errors) if display_pixel_errors else None
            ),
            "std_dev_display_error_px": (
                statistics.stdev(display_pixel_errors) if len(display_pixel_errors) > 1 else 0.0
            ) if display_pixel_errors else None,
            "min_display_error_px": min(display_pixel_errors) if display_pixel_errors else None,
            "max_display_error_px": max(display_pixel_errors) if display_pixel_errors else None,
            "mean_visual_angle_error_deg": (
                statistics.fmean(visual_angle_errors) if visual_angle_errors else None
            ),
            "median_visual_angle_error_deg": (
                statistics.median(visual_angle_errors) if visual_angle_errors else None
            ),
            "std_dev_visual_angle_error_deg": (
                statistics.stdev(visual_angle_errors) if len(visual_angle_errors) > 1 else 0.0
            ) if visual_angle_errors else None,
            "min_visual_angle_error_deg": min(visual_angle_errors) if visual_angle_errors else None,
            "max_visual_angle_error_deg": max(visual_angle_errors) if visual_angle_errors else None,
        })

    return per_target_errors


def print_pixel_error_summary(report: dict) -> None:
    """Print a readable summary for terminal/manual testing."""
    print("\nPixel Error Summary")
    print("-------------------")

    screen_info = report.get("screen_info")
    if screen_info:
        w, h = screen_info.get("width_px", 0), screen_info.get("height_px", 0)
        resolution_label = f"{w}x{h}" if w and h else "unknown"
        print(f"Canvas resolution: {resolution_label}")
        display_w = screen_info.get("display_width_px")
        display_h = screen_info.get("display_height_px")
        if display_w and display_h:
            print(f"Detected display resolution: {display_w}x{display_h}")
        width_cm = screen_info.get("width_cm")
        height_cm = screen_info.get("height_cm")
        viewing_distance_cm = screen_info.get("viewing_distance_cm")
        if width_cm and height_cm and viewing_distance_cm:
            print(
                "Visual-angle setup: "
                f"{width_cm:.2f}cm x {height_cm:.2f}cm at {viewing_distance_cm:.2f}cm"
            )
        print()

    config = report.get("config")
    if config:
        print(f"EMA enabled: {config['enable_ema']}")
        print(f"Kalman filter enabled: {config['enable_kalman_filter']}")
        print(f"Calibration points: {config['calibration_points']}")
        print(f"Samples per calibration point: {config['samples_per_calibration']}")
        print(f"Test points: {config['test_points']}")
        print(f"Accuracy samples per test point: {config['accuracy_samples_per_test_point']}")
        print()

    valid_sample_count = report["valid_sample_count"]
    print(f"Valid samples tested: {valid_sample_count}")
    print(f"Valid target points: {report['valid_target_count']}")
    print(f"Invalid samples: {report['invalid_sample_count']}")

    if valid_sample_count == 0:
        print("No valid samples were available for pixel error statistics.")
        return

    print(f"Mean pixel error: {report['mean_error']:.2f}px")
    print(f"Median pixel error: {report['median_error']:.2f}px")
    print(f"Standard deviation: {report['std_dev_error']:.2f}px")
    print(f"Minimum pixel error: {report['min_error']:.2f}px")
    print(f"Maximum pixel error: {report['max_error']:.2f}px")
    if report.get("mean_display_error_px") is not None:
        print(f"Mean display-resolution pixel error: {report['mean_display_error_px']:.2f}px")
        print(f"Median display-resolution pixel error: {report['median_display_error_px']:.2f}px")
        print(f"Display-resolution standard deviation: {report['std_dev_display_error_px']:.2f}px")
        print(f"Minimum display-resolution pixel error: {report['min_display_error_px']:.2f}px")
        print(f"Maximum display-resolution pixel error: {report['max_display_error_px']:.2f}px")
    if report.get("mean_visual_angle_error_deg") is not None:
        print(f"Mean visual angle error: {report['mean_visual_angle_error_deg']:.3f}deg")
        print(f"Median visual angle error: {report['median_visual_angle_error_deg']:.3f}deg")
        print(f"Visual angle standard deviation: {report['std_dev_visual_angle_error_deg']:.3f}deg")
        print(f"Minimum visual angle error: {report['min_visual_angle_error_deg']:.3f}deg")
        print(f"Maximum visual angle error: {report['max_visual_angle_error_deg']:.3f}deg")
    else:
        print("Visual angle error: not calculated (screen size and viewing distance are required)")

    if report["per_target_errors"]:
        print("\nPer-target estimated pixel errors")
        for target in report["per_target_errors"]:
            print(
                f"Target {target['target_number']}: "
                f"samples={target['sample_count']} "
                f"mean_error={target['mean_error']:.2f}px "
                f"median_error={target['median_error']:.2f}px"
                + (
                    f" mean_display_error={target['mean_display_error_px']:.2f}px"
                    if target.get("mean_display_error_px") is not None
                    else ""
                )
                + (
                    f" mean_visual_angle={target['mean_visual_angle_error_deg']:.3f}deg"
                    if target.get("mean_visual_angle_error_deg") is not None
                    else ""
                )
            )

    print("\nPer-sample pixel errors")
    displayed_samples = report["samples"][:50]
    for sample in displayed_samples:
        print(
            f"Sample {sample['sample_index']}: "
            f"target={sample['target']} gaze={sample['gaze_point']} "
            f"error={sample['error_px']:.2f}px"
            + (
                f" display_error={sample['display_error_px']:.2f}px"
                if sample.get("display_error_px") is not None
                else ""
            )
            + (
                f" angle={sample['error_visual_angle_deg']:.3f}deg"
                if sample.get("error_visual_angle_deg") is not None
                else ""
            )
        )
    if len(report["samples"]) > len(displayed_samples):
        print(f"... {len(report['samples']) - len(displayed_samples)} more raw samples saved in CSV.")

    invalid_samples = report.get("invalid_samples") or []
    if invalid_samples:
        print("\nInvalid samples")
        for sample in invalid_samples:
            print(f"Sample {sample['sample_index']}: {sample['reason']}")


def generate_test_targets(test_points: int, *, screen_w: int, screen_h: int) -> list[Coordinate]:
    """
    Generate configurable test targets from a requested point count.

    These are not final hardcoded study targets. They are runtime-generated from
    the menu value so the real target strategy can be swapped later.
    """
    if test_points <= 0:
        return []

    if test_points == 1:
        normalized_points = [(0.5, 0.5)]
    else:
        columns = math.ceil(math.sqrt(test_points))
        rows = math.ceil(test_points / columns)
        x_values = [0.5] if columns == 1 else [
            0.1 + (0.8 * col / (columns - 1))
            for col in range(columns)
        ]
        y_values = [0.5] if rows == 1 else [
            0.1 + (0.8 * row / (rows - 1))
            for row in range(rows)
        ]
        normalized_points = [
            (x_value, y_value)
            for y_value in y_values
            for x_value in x_values
        ][:test_points]

    return [
        (round(x_value * screen_w), round(y_value * screen_h))
        for x_value, y_value in normalized_points
    ]


def save_pixel_error_report(report: dict, output_dir: str | Path = "pixel_error_results") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"pixel_error_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report_path


def save_display_pixel_error_report(
    report: dict, output_dir: str | Path = "pixel_error_results"
) -> Optional[Path]:
    """Save a JSON file containing only display-resolution pixel error data.

    Extracts the display pixel error fields from the full report and writes them
    to a separate, focused JSON file. Returns ``None`` when no display pixel
    error data is present (i.e. ``display_width_px`` / ``display_height_px``
    were not set in ``ScreenInfo``).
    """
    # Check whether any display pixel error data was actually collected.
    has_summary = report.get("mean_display_error_px") is not None
    has_samples = any(
        s.get("display_error_px") is not None for s in report.get("samples", [])
    )
    if not has_summary and not has_samples:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"display_pixel_error_report_{timestamp}.json"

    screen_info = report.get("screen_info") or {}
    per_target = [
        {
            "target_number": t["target_number"],
            "target": t["target"],
            "sample_count": t["sample_count"],
            "mean_display_error_px": t.get("mean_display_error_px"),
            "median_display_error_px": t.get("median_display_error_px"),
            "std_dev_display_error_px": t.get("std_dev_display_error_px"),
            "min_display_error_px": t.get("min_display_error_px"),
            "max_display_error_px": t.get("max_display_error_px"),
        }
        for t in report.get("per_target_errors", [])
    ]
    samples = [
        {
            "sample_index": s["sample_index"],
            "target": s["target"],
            "gaze_point": s["gaze_point"],
            "display_error_px": s.get("display_error_px"),
        }
        for s in report.get("samples", [])
        if s.get("display_error_px") is not None
    ]

    display_report = {
        "screen_info": {
            "canvas_width_px": screen_info.get("width_px"),
            "canvas_height_px": screen_info.get("height_px"),
            "display_width_px": screen_info.get("display_width_px"),
            "display_height_px": screen_info.get("display_height_px"),
        },
        "valid_sample_count": report["valid_sample_count"],
        "valid_target_count": report["valid_target_count"],
        "mean_display_error_px": report.get("mean_display_error_px"),
        "median_display_error_px": report.get("median_display_error_px"),
        "std_dev_display_error_px": report.get("std_dev_display_error_px"),
        "min_display_error_px": report.get("min_display_error_px"),
        "max_display_error_px": report.get("max_display_error_px"),
        "target_mean_display_error_px": report.get("target_mean_display_error_px"),
        "target_median_display_error_px": report.get("target_median_display_error_px"),
        "target_std_dev_display_error_px": report.get("target_std_dev_display_error_px"),
        "target_min_display_error_px": report.get("target_min_display_error_px"),
        "target_max_display_error_px": report.get("target_max_display_error_px"),
        "per_target_errors": per_target,
        "samples": samples,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(display_report, file, indent=2)
    return report_path


def save_pixel_error_samples(report: dict, output_dir: str | Path = "pixel_error_results") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_path = output_path / f"pixel_error_samples_{timestamp}.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample_index",
            "target_x",
            "target_y",
            "gaze_x",
            "gaze_y",
            "error_px",
            "error_visual_angle_deg",
        ])
        for sample in report["samples"]:
            target_x, target_y = sample["target"]
            gaze_x, gaze_y = sample["gaze_point"]
            writer.writerow([
                sample["sample_index"],
                target_x,
                target_y,
                gaze_x,
                gaze_y,
                sample["error_px"],
                sample.get("error_visual_angle_deg"),
            ])
    return samples_path


def save_pixel_error_graph(report: dict, output_dir: str | Path = "pixel_error_results") -> Optional[Path]:
    """Save a bar graph of estimated pixel error per target point."""
    if report["valid_sample_count"] == 0:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping pixel error graph.")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = output_path / f"pixel_error_graph_{timestamp}.png"

    per_target_errors = report.get("per_target_errors") or []
    if per_target_errors:
        pixel_errors = [target["mean_error"] for target in per_target_errors]
        sample_numbers = [target["target_number"] for target in per_target_errors]
        mean_error = report["target_mean_error"]
        median_error = report["target_median_error"]
        std_dev_error = report["target_std_dev_error"]
        min_error = report["target_min_error"]
        max_error = report["target_max_error"]
        valid_count = report["valid_target_count"]
    else:
        pixel_errors = report["pixel_errors"]
        sample_numbers = list(range(1, len(pixel_errors) + 1))
        mean_error = report["mean_error"]
        median_error = report["median_error"]
        std_dev_error = report["std_dev_error"]
        min_error = report["min_error"]
        max_error = report["max_error"]
        valid_count = report["valid_sample_count"]

    fig, ax = plt.subplots(figsize=(11.5, 7.25))
    bars = ax.bar(
        sample_numbers,
        pixel_errors,
        color="#2f80ed",
        edgecolor="#1d5fb8",
        linewidth=1.2,
    )

    for bar, error in zip(bars, pixel_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max_error * 0.01, 1.0),
            f"{error:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(mean_error, color="#e53935", linestyle="--", linewidth=1.7,
               label=f"Mean: {mean_error:.2f}px")
    ax.axhline(median_error, color="#2eaa4f", linestyle=":", linewidth=1.9,
               label=f"Median: {median_error:.2f}px")

    ax.set_title("Gaze Pixel Error per Target", fontsize=14)
    ax.set_xlabel("Target number", fontsize=11)
    ax.set_ylabel("Pixel error (px)", fontsize=11)
    ax.set_xticks(sample_numbers)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    top_padding = max(max_error * 0.16, 10.0)
    ax.set_ylim(0, max_error + top_padding)

    footer = (
        f"Valid: {valid_count} | "
        f"Std dev: {std_dev_error:.2f}px | "
        f"Min: {min_error:.2f}px | "
        f"Max: {max_error:.2f}px"
    )
    fig.text(0.08, 0.045, footer, ha="left", va="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(graph_path, dpi=120)
    plt.close(fig)

    return graph_path


def save_display_pixel_error_graph(
    report: dict, output_dir: str | Path = "pixel_error_results"
) -> Optional[Path]:
    """Save a bar graph of display-resolution pixel error per target point.

    This is distinct from the canvas-based pixel error graph: errors are scaled
    from the canvas coordinate space into actual display (monitor) pixels using
    the ratio stored in ``screen_info.display_width_px / screen_info.width_px``.
    Returns ``None`` when display-resolution data is unavailable or matplotlib
    is not installed.
    """
    if report["valid_sample_count"] == 0:
        return None

    # Require that display pixel error data was actually collected.
    per_target_errors = report.get("per_target_errors") or []
    has_per_target = any(
        t.get("mean_display_error_px") is not None for t in per_target_errors
    )
    has_sample_level = bool(report.get("display_pixel_errors"))
    if not has_per_target and not has_sample_level:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping display pixel error graph.")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = output_path / f"display_pixel_error_graph_{timestamp}.png"

    # Prefer per-target aggregated values; fall back to raw sample-level values.
    if has_per_target:
        filtered = [t for t in per_target_errors if t.get("mean_display_error_px") is not None]
        display_errors = [t["mean_display_error_px"] for t in filtered]
        sample_numbers = [t["target_number"] for t in filtered]
        mean_error = report.get("target_mean_display_error_px")
        median_error = report.get("target_median_display_error_px")
        std_dev_error = report.get("target_std_dev_display_error_px")
        min_error = report.get("target_min_display_error_px")
        max_error = report.get("target_max_display_error_px")
        valid_count = len(filtered)
        x_label = "Target number"
    else:
        display_errors = report["display_pixel_errors"]
        sample_numbers = list(range(1, len(display_errors) + 1))
        mean_error = report.get("mean_display_error_px")
        median_error = report.get("median_display_error_px")
        std_dev_error = report.get("std_dev_display_error_px")
        min_error = report.get("min_display_error_px")
        max_error = report.get("max_display_error_px")
        valid_count = len(display_errors)
        x_label = "Sample number"

    if not display_errors or mean_error is None or max_error is None:
        return None

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(11.5, 7.25))
    bars = ax.bar(
        sample_numbers,
        display_errors,
        color="#f57c00",
        edgecolor="#bf360c",
        linewidth=1.2,
    )

    for bar, error in zip(bars, display_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max_error * 0.01, 1.0),
            f"{error:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(mean_error, color="#e53935", linestyle="--", linewidth=1.7,
               label=f"Mean: {mean_error:.2f}px")
    ax.axhline(median_error, color="#2eaa4f", linestyle=":", linewidth=1.9,
               label=f"Median: {median_error:.2f}px")

    # Annotate screen_info so readers know which display resolution was used.
    screen_info = report.get("screen_info") or {}
    display_w = screen_info.get("display_width_px")
    display_h = screen_info.get("display_height_px")
    resolution_suffix = (
        f"  [display {display_w}×{display_h}px]" if display_w and display_h else ""
    )

    ax.set_title(f"Gaze Display Pixel Error per Target{resolution_suffix}", fontsize=14)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Display pixel error (px)", fontsize=11)
    ax.set_xticks(sample_numbers)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    top_padding = max(max_error * 0.16, 10.0)
    ax.set_ylim(0, max_error + top_padding)

    footer = (
        f"Valid: {valid_count} | "
        f"Std dev: {std_dev_error:.2f}px | "
        f"Min: {min_error:.2f}px | "
        f"Max: {max_error:.2f}px"
    )
    fig.text(0.08, 0.045, footer, ha="left", va="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(graph_path, dpi=120)
    plt.close(fig)

    return graph_path


def _import_gaze_tracker():
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import gaze_tracker2

    return gaze_tracker2


def _apply_screen_resolution(
    gaze_tracker2,
    canvas_size: Optional[tuple[int, int]] = None,
) -> ScreenInfo:
    display_info = ScreenInfo.detect()
    if canvas_size:
        canvas_width_px, canvas_height_px = canvas_size
        gaze_tracker2.SCREEN_W = canvas_width_px
        gaze_tracker2.SCREEN_H = canvas_height_px
        return ScreenInfo(
            width_px=canvas_width_px,
            height_px=canvas_height_px,
            display_width_px=display_info.width_px or None,
            display_height_px=display_info.height_px or None,
        )

    if display_info.width_px and display_info.height_px:
        gaze_tracker2.SCREEN_W = display_info.width_px
        gaze_tracker2.SCREEN_H = display_info.height_px
        screen_info = display_info
    else:
        screen_info = ScreenInfo(
            width_px=gaze_tracker2.SCREEN_W,
            height_px=gaze_tracker2.SCREEN_H,
        )
    return screen_info


def _build_accuracy_smoother(gaze_tracker2, config: PixelErrorConfig):
    if config.enable_kalman_filter and config.enable_ema:
        return gaze_tracker2.HybridSmoother(alpha=0.3)
    if config.enable_kalman_filter:
        return gaze_tracker2.KalmanFilter2D()
    if config.enable_ema:
        return gaze_tracker2.EMASmoother(alpha=0.3)
    return None


def _smooth_gaze_point(smoother, normalized_point):
    if smoother is None:
        return normalized_point
    return smoother.update(normalized_point)


def _draw_accuracy_target(gaze_tracker2, canvas, target, target_index, target_count, progress):
    cv2 = gaze_tracker2.cv2
    screen_w = gaze_tracker2.SCREEN_W
    screen_h = gaze_tracker2.SCREEN_H
    canvas[:] = gaze_tracker2.C_BG
    gaze_tracker2.draw_grid(canvas)

    target_x, target_y = int(target[0]), int(target[1])
    cv2.circle(canvas, (target_x, target_y), 42, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (target_x, target_y), 18, (0, 220, 80), -1, cv2.LINE_AA)
    cv2.circle(canvas, (target_x, target_y), 4, (255, 255, 255), -1, cv2.LINE_AA)

    label = f"Accuracy Test  Point {target_index + 1} / {target_count}"
    label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0][0]
    gaze_tracker2.txt(canvas, label, (screen_w // 2 - label_w // 2, 52),
                      scale=0.82, color=(220, 220, 220), thick=2)

    hint = "Keep eyes on the green dot  |  Q=quit"
    hint_w = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    gaze_tracker2.txt(canvas, hint, (screen_w // 2 - hint_w // 2, screen_h - 24),
                      scale=0.55, color=(130, 130, 130), thick=1)

    gaze_tracker2.draw_bar(canvas, progress, screen_w // 2 - 260, screen_h - 64, 520, 14)


class GazePixelErrorRunner:
    """Run calibration, collect gaze samples on test targets, and report pixel error."""

    def __init__(
        self,
        config: PixelErrorConfig,
        *,
        camera_id: int = 0,
        visual_angle_settings: Optional[tuple[float, float, float]] = None,
        canvas_size: Optional[tuple[int, int]] = None,
    ):
        self.config = config
        self.camera_id = camera_id
        self.visual_angle_settings = visual_angle_settings
        self.canvas_size = canvas_size

    def run(self) -> Optional[dict]:
        gaze_tracker2 = _import_gaze_tracker()
        detected_screen_info = _apply_screen_resolution(gaze_tracker2, self.canvas_size)
        print(f"Using canvas resolution: {detected_screen_info.as_label()}")
        if detected_screen_info.display_width_px and detected_screen_info.display_height_px:
            print(
                "Detected display resolution: "
                f"{detected_screen_info.display_width_px}x{detected_screen_info.display_height_px}"
            )
        cv2 = gaze_tracker2.cv2
        np = gaze_tracker2.np

        tracker = gaze_tracker2.GazeTrackerApp(
            camera_id=self.camera_id,
            num_points=self.config.calibration_points,
            spp=self.config.samples_per_calibration,
            show_camera_window=True,
            debug_landmarks=True,
            tutorial_enabled=True,
        )

        print("\nStarting gaze calibration...")
        if not tracker.calibrate():
            print("Calibration was cancelled or failed.")
            return None

        targets = generate_test_targets(
            self.config.test_points,
            screen_w=gaze_tracker2.SCREEN_W,
            screen_h=gaze_tracker2.SCREEN_H,
        )
        if not targets:
            print("No test targets were configured. Set Test Points to at least 1.")
            return calculate_pixel_error_report([], [], config=self.config)

        smoother = _build_accuracy_smoother(gaze_tracker2, self.config)
        gaze_points: list[RawCoordinate] = []
        target_samples: list[RawCoordinate] = []
        hold_frames = 20
        samples_per_target = max(1, self.config.accuracy_samples_per_test_point)

        cv2.namedWindow(tracker.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(tracker.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("\nStarting accuracy test. Look at each green dot until it advances.")

        try:
            for target_index, target in enumerate(targets):
                collected_for_target = 0
                frame_count = 0
                if smoother is not None and hasattr(smoother, "reset"):
                    smoother.reset()

                while collected_for_target < samples_per_target:
                    ret, cam = tracker._cap.read()
                    if not ret:
                        print("[Error] Camera frame read failed during accuracy test.")
                        break

                    cam = cv2.flip(cam, 1)
                    res = tracker.mesh.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

                    feat = lms = None
                    is_blinking = False
                    if res.multi_face_landmarks:
                        lms = res.multi_face_landmarks[0].landmark
                        feat = tracker.extractor.extract(lms, cam.shape[1], cam.shape[0])
                        is_blinking = tracker.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])

                    frame_count += 1
                    progress = collected_for_target / samples_per_target
                    _draw_accuracy_target(
                        gaze_tracker2,
                        tracker.canvas,
                        target,
                        target_index,
                        len(targets),
                        progress,
                    )

                    if feat is not None and not is_blinking and frame_count > hold_frames:
                        raw = tracker.calib.model.predict(feat)
                        smoothed = _smooth_gaze_point(smoother, raw)
                        gaze_x = float(np.clip(smoothed[0] * gaze_tracker2.SCREEN_W, 0, gaze_tracker2.SCREEN_W - 1))
                        gaze_y = float(np.clip(smoothed[1] * gaze_tracker2.SCREEN_H, 0, gaze_tracker2.SCREEN_H - 1))
                        gaze_points.append((gaze_x, gaze_y))
                        target_samples.append(target)
                        collected_for_target += 1

                        cv2.circle(tracker.canvas, (int(gaze_x), int(gaze_y)), 8,
                                   gaze_tracker2.C_CURSOR, -1, cv2.LINE_AA)

                    if tracker._pip:
                        distance_info = tracker._estimate_distance(
                            lms,
                            cam.shape[1],
                            cam.shape[0],
                        ) if tracker._show_distance else None
                        gaze_tracker2.overlay_pip(tracker.canvas, cam, distance_info=distance_info)

                    cv2.imshow(tracker.WIN, tracker.canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and chr(key).lower() == "q":
                        print("Accuracy test cancelled.")
                        break

                if collected_for_target < samples_per_target:
                    break
        finally:
            cv2.destroyWindow(tracker.WIN)
            if getattr(tracker, "_cap", None) is not None:
                tracker._cap.release()

        screen_info = detected_screen_info
        if self.visual_angle_settings:
            screen_width_cm, screen_height_cm, viewing_distance_cm = self.visual_angle_settings
            screen_info = ScreenInfo(
                width_px=detected_screen_info.width_px,
                height_px=detected_screen_info.height_px,
                width_cm=screen_width_cm,
                height_cm=screen_height_cm,
                viewing_distance_cm=viewing_distance_cm,
                display_width_px=detected_screen_info.display_width_px,
                display_height_px=detected_screen_info.display_height_px,
            )
        report = calculate_pixel_error_report(
            target_samples, gaze_points, config=self.config, screen_info=screen_info
        )
        print_pixel_error_summary(report)
        report_path = save_pixel_error_report(report)
        samples_path = save_pixel_error_samples(report)
        display_report_path = save_display_pixel_error_report(report)
        graph_path = save_pixel_error_graph(report)
        display_graph_path = save_display_pixel_error_graph(report)
        print(f"\nSaved report: {report_path}")
        print(f"Saved samples: {samples_path}")
        if display_report_path:
            print(f"Saved display px report: {display_report_path}")
        if graph_path:
            print(f"Saved graph (canvas px): {graph_path}")
        if display_graph_path:
            print(f"Saved graph (display px): {display_graph_path}")
        return report


def load_points_from_csv(
    csv_path: str | Path,
    *,
    target_x_column: str = "target_x",
    target_y_column: str = "target_y",
    gaze_x_column: str = "gaze_x",
    gaze_y_column: str = "gaze_y",
) -> tuple[list[RawCoordinate], list[RawCoordinate]]:
    """
    Load target and gaze point pairs from a CSV file.

    Expected default columns:
        target_x,target_y,gaze_x,gaze_y
    """
    targets: list[RawCoordinate] = []
    gaze_points: list[RawCoordinate] = []

    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            targets.append((row.get(target_x_column), row.get(target_y_column)))
            gaze_points.append((row.get(gaze_x_column), row.get(gaze_y_column)))

    return targets, gaze_points


def prompt_yes_no(label: str, *, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    try:
        answer = input(f"{label} [{suffix}]: ").strip().lower()
    except EOFError:
        print()
        return default
    if not answer:
        return default
    return answer in {"y", "yes", "true", "1", "on"}


def prompt_int(label: str, *, default: int, minimum: int = 0) -> int:
    while True:
        try:
            answer = input(f"{label} [{default}]: ").strip()
        except EOFError:
            print()
            return default
        if not answer:
            return default

        try:
            value = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value < minimum:
            print(f"Please enter a value greater than or equal to {minimum}.")
            continue

        return value


def prompt_optional_float(label: str, *, minimum: float = 0.0) -> Optional[float]:
    while True:
        try:
            answer = input(f"{label} [skip]: ").strip()
        except EOFError:
            print()
            return None
        if not answer:
            return None

        try:
            value = float(answer)
        except ValueError:
            print("Please enter a number.")
            continue

        if value <= minimum:
            print(f"Please enter a value greater than {minimum}.")
            continue

        return value


def prompt_visual_angle_settings() -> Optional[tuple[float, float, float]]:
    """Collect physical setup needed to convert pixel error to visual angle."""
    if not prompt_yes_no("Calculate visual angle error", default=False):
        return None

    width_cm = prompt_optional_float("Monitor visible width in cm")
    height_cm = prompt_optional_float("Monitor visible height in cm")
    viewing_distance_cm = prompt_optional_float("Eye-to-screen distance in cm")
    if width_cm is None or height_cm is None or viewing_distance_cm is None:
        print("Skipping visual angle error because one or more measurements were not provided.")
        return None

    return width_cm, height_cm, viewing_distance_cm


def prompt_canvas_size() -> Optional[tuple[int, int]]:
    """Optionally override the accuracy-test canvas coordinate space."""
    if not prompt_yes_no("Use custom accuracy canvas size", default=False):
        return None

    width_px = prompt_int("Canvas width px", default=1920, minimum=1)
    height_px = prompt_int("Canvas height px", default=1080, minimum=1)
    return width_px, height_px


def prompt_choice_int(label: str, choices: Sequence[int], *, default: int) -> int:
    choice_text = "/".join(str(choice) for choice in choices)
    while True:
        try:
            answer = input(f"{label} ({choice_text}) [{default}]: ").strip()
        except EOFError:
            print()
            return default
        if not answer:
            return default

        try:
            value = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value not in choices:
            print(f"Please choose one of: {choice_text}.")
            continue

        return value


def prompt_pixel_error_config() -> PixelErrorConfig:
    """Collect menu settings for a pixel-error accuracy test run."""
    print("\nPixel Error Accuracy Test Menu")
    print("------------------------------")
    enable_ema = prompt_yes_no("Enable EMA", default=False)
    enable_kalman_filter = prompt_yes_no("Enable Kalman Filter", default=False)
    calibration_points = prompt_choice_int("How many Calibration Points", (5, 9, 16, 25), default=9)
    samples_per_calibration = prompt_int(
        "How many samples per calibration",
        default=90,
        minimum=1,
    )
    test_points = prompt_int("How many Test Points", default=9, minimum=1)
    accuracy_samples_per_test_point = prompt_int(
        "How many accuracy samples per test point",
        default=30,
        minimum=1,
    )

    return PixelErrorConfig(
        enable_ema=enable_ema,
        enable_kalman_filter=enable_kalman_filter,
        calibration_points=calibration_points,
        samples_per_calibration=samples_per_calibration,
        test_points=test_points,
        accuracy_samples_per_test_point=accuracy_samples_per_test_point,
    )


def main() -> None:
    """
    Standalone entry point for real calibration/testing or CSV-only analysis.
    """
    config = prompt_pixel_error_config()
    visual_angle_settings = prompt_visual_angle_settings()
    canvas_size = prompt_canvas_size()

    print("\nMode")
    print("1. Run calibration and accuracy test")
    print("2. Analyze target/gaze coordinates from CSV")
    print("3. No coordinate data yet")
    try:
        choice = input("Choose an option [1]: ").strip() or "1"
    except EOFError:
        print()
        choice = "1"

    if choice == "1":
        GazePixelErrorRunner(
            config,
            visual_angle_settings=visual_angle_settings,
            canvas_size=canvas_size,
        ).run()
        return

    if choice == "2":
        try:
            csv_path = input("CSV path: ").strip()
        except EOFError:
            print()
            csv_path = ""
        if csv_path:
            try:
                targets, gaze_points = load_points_from_csv(csv_path)
            except OSError as error:
                print(f"Could not read CSV file: {error}")
                targets = []
                gaze_points = []
        else:
            targets = []
            gaze_points = []
    else:
        targets = []
        gaze_points = []

    screen_info = ScreenInfo.detect()
    if visual_angle_settings:
        screen_width_cm, screen_height_cm, viewing_distance_cm = visual_angle_settings
        canvas_width_px, canvas_height_px = canvas_size or (
            screen_info.width_px,
            screen_info.height_px,
        )
        screen_info = ScreenInfo(
            width_px=canvas_width_px,
            height_px=canvas_height_px,
            width_cm=screen_width_cm,
            height_cm=screen_height_cm,
            viewing_distance_cm=viewing_distance_cm,
            display_width_px=screen_info.width_px,
            display_height_px=screen_info.height_px,
        )
    elif canvas_size:
        canvas_width_px, canvas_height_px = canvas_size
        screen_info = ScreenInfo(
            width_px=canvas_width_px,
            height_px=canvas_height_px,
            display_width_px=screen_info.width_px,
            display_height_px=screen_info.height_px,
        )
    report = calculate_pixel_error_report(
        targets, gaze_points, config=config, screen_info=screen_info
    )
    print_pixel_error_summary(report)
    report_path = save_pixel_error_report(report)
    samples_path = save_pixel_error_samples(report)
    display_report_path = save_display_pixel_error_report(report)
    graph_path = save_pixel_error_graph(report)
    display_graph_path = save_display_pixel_error_graph(report)
    print(f"\nSaved report: {report_path}")
    print(f"Saved samples: {samples_path}")
    if display_report_path:
        print(f"Saved display px report: {display_report_path}")
    if graph_path:
        print(f"Saved graph (canvas px): {graph_path}")
    if display_graph_path:
        print(f"Saved graph (display px): {display_graph_path}")


if __name__ == "__main__":
    main()