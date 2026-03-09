import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform


def _barcodes_to_array(barcodes):
    """Map any iterable barcode to a 0/1 numpy array."""
    n, L = len(barcodes), len(barcodes[0])
    arr = np.zeros((n, L), dtype=np.int8)
    for i, bc in enumerate(barcodes):
        for j, c in enumerate(bc):
            arr[i, j] = c if isinstance(c, (int, np.integer)) else (1 if c in ("+", "1") else 0)
    return arr


def hamming_distance(bc1, bc2):
    return sum(a != b for a, b in zip(bc1, bc2))


def build_neighbor_dict(tree: KDTree, max_radius: float):
    """Return {i: [(j, dist_ij), …]} for all pairs with dist ≤ max_radius."""
    sparse = tree.sparse_distance_matrix(tree, max_radius, output_type="dict")
    n = tree.data.shape[0]
    neigh = {i: [] for i in range(n)}
    for (i, j), d in sparse.items():
        if i == j:
            continue
        neigh[i].append((j, d))
        neigh[j].append((i, d))
    return neigh


def compute_global_hamming(barcodes, cell_fraction=1.0, random_seed=None):
    """Average Hamming distance between *non-identical* pairs (global)."""
    arr = _barcodes_to_array(barcodes)
    n, L = arr.shape

    if cell_fraction < 1.0:
        if random_seed is not None:
            np.random.seed(random_seed)
        idx = np.random.choice(n, size=int(n * cell_fraction), replace=False)
    else:
        idx = np.arange(n)

    D = squareform(pdist(arr, metric="hamming")) * L
    means = [
        D[i][(D[i] > 0) & (np.arange(n) != i)].mean()
        for i in tqdm(idx, desc="Calculating H_hat", ncols=100)
    ]
    return float(np.mean(means))


def compute_global_occurrence_density(barcodes):
    """Non-spatial P_same (uncorrected for errors)."""
    N = len(barcodes)
    counts = Counter(tuple(bc) for bc in barcodes)
    return sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))


def compute_shell_metrics_quadratic(shell_edges, barcodes, coords, tree, H_hat,
                                    cell_fraction=1.0, random_seed=None):
    """
    For each shell (r_in, r_out] compute H_d_shell, P_obs_shell, E_shell using the quadratic inversion.

    Here, we simply model the local non-same hamming rate as H(d) = w(d) * Ĥ + (1 - w(d)) * 1, with w(d) being the
    ratio between erroneous same barcodes and random other non-same barcodes in that shell.
    
    W(d) is then estimated as W(d) = (1−P_cl(d))/((1−P_cl(d))+E P_cl(d))
    where P_cl(d) is the true same-barcode fraction in that shell, and E is the error rate. We express
    P_cl(d) = P_obs(d) / (1 - E)^2 because both the "same" and "different" pairs are affected by errors, and the chance
    of a pair being observed as "same" is reduced by (1 - E)^2 due to the possibility of either barcode being misread.

    Inverting this gives the quadratic inversion used to estimate E(d) per shell to the first order.
    We expect E(d) to be roughly constant across shells.

    """
    arr = _barcodes_to_array(barcodes)
    n, L = arr.shape
    max_r = shell_edges[-1]
    neigh_dict = build_neighbor_dict(tree, max_r)

    if cell_fraction < 1.0:
        if random_seed is not None:
            np.random.seed(random_seed)
        sample_idx = np.random.choice(n, size=int(n * cell_fraction), replace=False)
    else:
        sample_idx = np.arange(n)

    avg_hamming_no_equal = []
    occurrence_density = []

    for r_in, r_out in tqdm(
        list(zip(shell_edges[:-1], shell_edges[1:])),
        desc="Shells",
        ncols=100,
    ):
        h_sum = o_sum = 0.0
        h_cnt = o_cnt = 0

        for i in sample_idx:
            js = [j for j, d in neigh_dict[i] if r_in < d <= r_out]
            if not js:
                continue

            js = np.asarray(js, dtype=int)
            diffs = (arr[js] != arr[i])
            hd = diffs.sum(axis=1)

            nz = hd > 0
            if nz.any():
                h_sum += hd[nz].mean()
                h_cnt += 1

            same = np.count_nonzero(~nz)
            o_sum += same / hd.size
            o_cnt += 1

        avg_hamming_no_equal.append(h_sum / h_cnt if h_cnt else np.nan)
        occurrence_density.append(o_sum / o_cnt if o_cnt else np.nan)

    # Quadratic inversion to estimate E(d) per shell:
    E_shell = []
    for H_d, P in zip(avg_hamming_no_equal, occurrence_density):
        if np.isnan(H_d) or np.isnan(P) or H_d >= H_hat:
            E_shell.append(np.nan)
            continue
        D = H_hat - H_d
        term = P * (H_d - 1) + 2 * D
        disc = term**2 - 4 * (D**2) * (1 - P)
        if disc < 0:
            E_shell.append(np.nan)
            continue
        s = np.sqrt(disc)
        c1 = (term - s) / (2 * D)
        if 0 <= c1 <= 1:
            E_shell.append(c1)
        else:
            c2 = (term + s) / (2 * D)
            E_shell.append(c2 if 0 <= c2 <= 1 else np.nan)

    return avg_hamming_no_equal, occurrence_density, E_shell

def compute_average_hamming_histogram(barcodes, coords, tree, max_non_clone=100, query_k=200):
    avg_hd = []
    n = len(coords)
    for i in range(n):
        dists, idxs = tree.query(coords[i], k=query_k)
        hd_list = []
        for j in idxs:
            if j == i:
                continue
            if barcodes[j] != barcodes[i]:
                hd_list.append(hamming_distance(barcodes[i], barcodes[j]))
            if len(hd_list) >= max_non_clone:
                break
        avg_hd.append(np.mean(hd_list) if hd_list else np.nan)
    return avg_hd


def compute_clone_threshold_histogram(barcodes, coords, tree, max_neighbors=100):
    clone_threshold = []
    n = len(coords)
    for i in range(n):
        dists, idxs = tree.query(coords[i], k=max_neighbors + 1)
        neighs = [j for j in idxs if j != i]
        cumul = 0
        threshold = None
        for count, j in enumerate(neighs[:max_neighbors], start=1):
            if barcodes[j] == barcodes[i]:
                cumul += 1
            if (cumul / count) < 0.5:
                threshold = cumul
                break
        clone_threshold.append(cumul if threshold is None else threshold)
    return clone_threshold

def main():
    # ── parameters ── #
    SHELL_WIDTH = 10        # µm
    MAX_RADIUS = 100       # µm – outer edge of last shell
    shell_edges = np.arange(10, MAX_RADIUS + SHELL_WIDTH, SHELL_WIDTH)
    shell_midpoints = (shell_edges[:-1] + shell_edges[1:]) / 2

    repo_root = Path(__file__).resolve().parents[3]
    path_csv = repo_root / "example_data" / "HEK_cells_200mgTAM" / "barcodes_gmm_triplet.csv"
    if not path_csv.exists():
        raise FileNotFoundError(
            f"{path_csv} not found. Run examples/barcoding_with_gmm.py first to generate it."
        )
    df = pd.read_csv(path_csv)
    # take a smaller subset for testing
    df = df.sample(frac=0.6, random_state=42).reset_index(drop=True)

    L = len(df["barcode_string"].iloc[0])
    df = df[df["barcode_string"] != "-" * L]

    barcodes = [
        [row[f"barcode_channel_{i}"] for i in range(1, L + 1)]
        for _, row in df.iterrows()
    ]
    coords = df[["x", "y"]].values

    cell_fraction = 1.0
    random_seed = 42

    H_hat = compute_global_hamming(barcodes, cell_fraction, random_seed)
    print("Global H_hat:", H_hat)
    tree = KDTree(coords)

    avg_hamm_hist = compute_average_hamming_histogram(barcodes, coords, tree, 20, 200)
    clone_thresh_hist = compute_clone_threshold_histogram(barcodes, coords, tree, 1000)

    plt.figure(figsize=(7, 4.5))
    plt.hist(avg_hamm_hist, bins=50, edgecolor="black", alpha=0.8)
    plt.xlabel("Average Hamming Distance (first 100 non-clone neighbours)")
    plt.ylabel("Cells")
    plt.title("Histogram of Average Hamming Distances")
    plt.grid(True)

    plt.figure(figsize=(7, 4.5))
    plt.hist(
        clone_thresh_hist,
        bins=range(min(clone_thresh_hist), max(clone_thresh_hist) + 2),
        align="left",
        edgecolor="black",
        alpha=0.8,
    )
    plt.xlabel("Clone Threshold (until clone-fraction < 50 %)")
    plt.ylabel("Cells")
    plt.title("Histogram of Clone Thresholds")
    plt.grid(True)

    avg_hamming_shell, P_obs_shell, E_shell = compute_shell_metrics_quadratic(
        shell_edges,
        barcodes,
        coords,
        tree,
        H_hat,
        cell_fraction,
        random_seed,
    )

    P_obs_shell = np.array(P_obs_shell)
    E_shell = np.array(E_shell)
    P_true_shell = P_obs_shell / (1 - E_shell) ** 2

    # global baseline, error-corrected (option A)
    P_global_raw = compute_global_occurrence_density(barcodes)
    valid_E = E_shell[~np.isnan(E_shell)]
    E_global = valid_E[-1] if valid_E.size else 0.0
    P_global_true = P_global_raw / (1 - E_global) ** 2

    P_excess_shell = (P_true_shell - P_global_true) / (1 - P_global_true)

    # ── scatter plots ── #
    plt.figure(figsize=(7, 4.5))
    plt.scatter(shell_midpoints, avg_hamming_shell, label="H(d) excluding equal barcodes")
    plt.xlabel("Shell midpoint (µm)")
    plt.ylabel("Average Hamming Distance")
    plt.title("Hamming Distance per Shell")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(shell_midpoints, P_true_shell, label="Corrected P_same")
    plt.hlines(P_global_true, shell_midpoints[0], shell_midpoints[-1],
               label="Global baseline", linestyles="dashed")
    plt.xlabel("Shell midpoint (µm)")
    plt.ylabel("Probability")
    plt.title("Same-barcode Probability per Shell")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(shell_midpoints, P_excess_shell, label="Excess clustering (normalised)")
    plt.xlabel("Shell midpoint (µm)")
    plt.ylabel("P_excess(d)")
    plt.title("Excess Same-barcode Clustering per Shell")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(shell_midpoints, E_shell, label="E(d) per shell", marker="s")
    plt.xlabel("Shell midpoint (µm)")
    plt.ylabel("Error rate E(d)")
    plt.title("Inferred Error Rate per Shell")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
