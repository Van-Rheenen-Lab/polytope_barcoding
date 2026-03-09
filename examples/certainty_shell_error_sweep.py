from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from polytope_barcoding.post_barcoding_analysis.hamming_error_inference import (  # noqa: E402
    compute_shell_metrics_quadratic,
)


def ordered_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(
        cols,
        key=lambda c: int("".join(ch for ch in c[len(prefix):] if ch.isdigit()) or 10**9),
    )


def global_hamming_compact(barcodes: np.ndarray) -> float:
    arr = np.asarray(barcodes, dtype=np.int8)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")

    n = arr.shape[0]
    unique_barcodes, counts = np.unique(arr, axis=0, return_counts=True)
    if unique_barcodes.shape[0] < 2:
        return float("nan")

    hd = (unique_barcodes[:, None, :] != unique_barcodes[None, :, :]).sum(axis=2).astype(np.float64)
    weighted_sum = (hd * counts[None, :]).sum(axis=1)
    denom = n - counts
    mean_per_group = np.full(unique_barcodes.shape[0], np.nan, dtype=np.float64)
    np.divide(weighted_sum, denom, out=mean_per_group, where=denom > 0)
    return float((mean_per_group * counts).sum() / n)


def sweep(
    df: pd.DataFrame,
    thresholds: np.ndarray,
    shell_um: float = 20.0,
    certainty_prefix: str = "certainty_channel_",
    barcode_prefix: str = "barcode_channel_",
    x_col: str = "x",
    y_col: str = "y",
    ignore_negative_barcodes: bool = False,
    cell_fraction: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    barcode_cols = ordered_cols(df, barcode_prefix)
    certainty_cols = ordered_cols(df, certainty_prefix)
    if not barcode_cols or not certainty_cols:
        raise ValueError("Missing barcode/certainty columns in csv.")
    if ignore_negative_barcodes:
        df = df.loc[(df[barcode_cols].to_numpy(dtype=np.int8, copy=False) != 0).any(axis=1)].copy()

    cert = df[certainty_cols].to_numpy(dtype=np.float64, copy=False)
    cell_certainty = np.nanmin(cert, axis=1)

    rows: list[dict] = []
    for thr in thresholds:
        keep = np.isfinite(cell_certainty) & (cell_certainty >= float(thr))
        kept = df.loc[keep]
        n = int(len(kept))
        e = np.nan

        if n >= 3:
            b = kept[barcode_cols].to_numpy(dtype=np.int8, copy=False)
            coords = kept[[x_col, y_col]].to_numpy(dtype=np.float64, copy=False)
            h_hat = global_hamming_compact(b)
            if not np.isfinite(h_hat):
                rows.append(
                    {
                        "threshold": float(thr),
                        "cells_retained": n,
                        "fraction_retained": float(n / len(df)) if len(df) else np.nan,
                        "error_rate_shell": np.nan,
                    }
                )
                print(f"thr={thr:.2f} cells={n} error=nan")
                continue
            tree = KDTree(coords)
            _, _, e_shell = compute_shell_metrics_quadratic(
                np.array([0.0, shell_um], dtype=float),
                b,
                coords,
                tree,
                h_hat,
                cell_fraction=cell_fraction,
                random_seed=random_seed,
            )
            e = float(e_shell[0]) if e_shell else np.nan

        rows.append(
            {
                "threshold": float(thr),
                "cells_retained": n,
                "fraction_retained": float(n / len(df)) if len(df) else np.nan,
                "error_rate_shell": e,
            }
        )
        print(f"thr={thr:.2f} cells={n} error={e}")

    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, shell_um: float = 20.0) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(results["threshold"], results["error_rate_shell"], marker="o")
    ax1.set_ylabel(f"error rate E(0,{shell_um:g}]")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(results["threshold"], results["cells_retained"], marker="o", color="tab:orange")
    ax2.set_xlabel("certainty threshold")
    ax2.set_ylabel("cells retained")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def run_certainty_shell_error_sweep() -> None:
    csv_path = repo_root / "example_data" / "HEK_cells_200mgTAM" / "barcodes_gmm_triplet.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run examples/barcoding_with_gmm.py first to generate it."
        )

    shell_um = 40.0
    thresholds = np.arange(0.0, 1.01, 0.05)
    ignore_negative_barcodes = True

    df = pd.read_csv(csv_path)
    results = sweep(
        df,
        thresholds=thresholds,
        shell_um=shell_um,
        ignore_negative_barcodes=ignore_negative_barcodes,
    )
    plot_results(results, shell_um=shell_um)

    save_threshold = 0.25
    certainty_cols = ordered_cols(df, "certainty_channel_")

    cert = df[certainty_cols].to_numpy(dtype=np.float64, copy=False)
    cell_certainty = np.nanmin(cert, axis=1)
    keep = np.isfinite(cell_certainty) & (cell_certainty >= save_threshold)
    filtered_df = df.loc[keep].copy()

    thr_label = f"{save_threshold:.2f}".replace(".", "p")
    filtered_csv_path = csv_path.with_name(f"{csv_path.stem}_certainty_{thr_label}.csv")
    filtered_df.to_csv(filtered_csv_path, index=False)
    print(
        f"saved filtered csv ({len(filtered_df)} / {len(df)} cells, "
        f"threshold={save_threshold:.2f}): {filtered_csv_path}"
    )


if __name__ == "__main__":
    run_certainty_shell_error_sweep()
