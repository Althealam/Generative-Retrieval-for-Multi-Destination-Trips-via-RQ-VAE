"""Accuracy@4 evaluation (same logic as data/evaluation_demo.ipynb)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import data_dir


def load_ground_truth(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else data_dir() / "ground_truth.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Ground truth not found: {p}")
    return pd.read_csv(p)


def evaluate_accuracy_at_4(submission: pd.DataFrame, ground_truth: pd.DataFrame) -> tuple[float, int]:
    """True next city appears in any of the four predicted slots. Returns (mean, n_trips)."""
    sub = submission.copy()
    gt = ground_truth.copy()
    if "utrip_id" not in sub.columns:
        sub = sub.reset_index()
    if "utrip_id" not in gt.columns:
        gt = gt.reset_index()
    data = sub.merge(gt[["utrip_id", "city_id"]], on="utrip_id", how="inner")
    if data.empty:
        raise ValueError("No overlapping utrip_id between submission and ground truth.")
    hits = (
        (data["city_id"] == data["city_id_1"])
        | (data["city_id"] == data["city_id_2"])
        | (data["city_id"] == data["city_id_3"])
        | (data["city_id"] == data["city_id_4"])
    )
    return float(hits.mean()), int(len(data))


def print_accuracy_at_4_report(
    submission_df: pd.DataFrame,
    *,
    skip: bool = False,
    ground_truth_path: Path | str | None = None,
) -> None:
    """Print Accuracy@4 after training; no-op if skip or ground truth missing."""
    if skip:
        return
    try:
        gt = load_ground_truth(ground_truth_path)
    except FileNotFoundError as e:
        print(f"⚠️ 跳过评估（未找到 ground truth）: {e}")
        return
    acc, n = evaluate_accuracy_at_4(submission_df, gt)
    print(f"📊 Accuracy@4: {acc:.6f}  (trips evaluated: {n})")
