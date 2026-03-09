from .core import Binarization, CellData
from skimage.filters import threshold_otsu
from typing import Literal
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from itertools import combinations
from typing import Optional
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

def _store_avg_intensities_in_properties(
        cell_data: CellData,
        cell_ids: np.ndarray,
        avg_intensities: np.ndarray,
) -> None:
    """
    Persist per-cell, per-channel average intensities on cell_data.properties.

    Parameters
    ----------
    cell_data : CellData
        Target cell data object.
    cell_ids : np.ndarray
        Label ids corresponding to rows in avg_intensities.
    avg_intensities : np.ndarray
        Array of shape (K, C), where K is number of cells and C number of channels.
    """
    props = cell_data.properties
    if not isinstance(props, pd.DataFrame):
        raise RuntimeError("cell_data.properties must be a pandas DataFrame")

    ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    values = np.asarray(avg_intensities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("avg_intensities must have shape (K, C)")
    if values.shape[0] != ids.shape[0]:
        raise ValueError("cell_ids length must match avg_intensities rows")

    n_rows = len(props)
    n_channels = values.shape[1]

    if n_rows == 0:
        for ch in range(n_channels):
            props[f"avg_intensity_channel_{ch + 1}"] = np.array([], dtype=np.float64)
        return

    if "mask_number" in props.columns and ids.size:
        mask_numbers = props["mask_number"].to_numpy(dtype=np.int64, copy=False)
        max_id = int(max(int(ids.max()), int(mask_numbers.max()) if mask_numbers.size else 0))
        label_to_row = np.full(max_id + 1, -1, dtype=np.int32)
        valid_ids = (ids >= 0) & (ids <= max_id)
        label_to_row[ids[valid_ids]] = np.arange(ids.size, dtype=np.int32)[valid_ids]
        row_idx = np.full(n_rows, -1, dtype=np.int32)
        in_bounds = (mask_numbers >= 0) & (mask_numbers < label_to_row.size)
        row_idx[in_bounds] = label_to_row[mask_numbers[in_bounds]]
        valid_rows = row_idx >= 0

        for ch in range(n_channels):
            column = np.full(n_rows, np.nan, dtype=np.float64)
            column[valid_rows] = values[row_idx[valid_rows], ch]
            props[f"avg_intensity_channel_{ch + 1}"] = column
        return

    if n_rows == values.shape[0]:
        for ch in range(n_channels):
            props[f"avg_intensity_channel_{ch + 1}"] = values[:, ch]
        return

    raise RuntimeError(
        "Cannot align avg intensities to cell_data.properties: "
        "missing 'mask_number' and row counts differ."
    )


class OtsuBinarize(Binarization):
    def __init__(self, fluorophore_channels, cell_data):
        super().__init__(fluorophore_channels, cell_data)
        self.channel_means: np.ndarray | None = None
        self.average_masks = self.calculate_avg_intensity_per_cell()
        self.valid_mask = self.cell_data.masks > 0  # only cell pixels

        labels = np.asarray(self.cell_data.masks)
        cell_ids = np.unique(labels)
        cell_ids = cell_ids[cell_ids != 0]
        if self.channel_means is not None and cell_ids.size:
            avg_intensities = self.channel_means[:, cell_ids].T
            _store_avg_intensities_in_properties(self.cell_data, cell_ids, avg_intensities)

    def binarize_channels(self, channel=None, thresholds=None):
        if channel is None:
            channels = range(self.average_masks.shape[0])
        else:
            channels = [channel]

        if thresholds is None:
            thresholds = [
                threshold_otsu(self.average_masks[ch][self.valid_mask])
                for ch in channels
            ]

        binary_masks = []
        for ch, thr in zip(channels, thresholds):
            binary_mask = self.average_masks[ch] > thr
            binary_masks.append(binary_mask)
        return np.array(binary_masks)

    def calculate_avg_intensity_per_cell(self):
        masks = np.asarray(self.cell_data.masks)
        fluor = np.asarray(self.fluorophore_channels, dtype=np.float64)

        average_masks = np.zeros_like(fluor, dtype=np.float64)
        if masks.size == 0 or fluor.size == 0:
            return average_masks

        mask_flat = masks.reshape(-1)
        if mask_flat.dtype.kind != "i":
            mask_flat = mask_flat.astype(np.int32, copy=False)

        valid = mask_flat > 0
        if not np.any(valid):
            return average_masks

        label_ids = mask_flat[valid]
        max_label = int(label_ids.max())
        counts = np.bincount(label_ids, minlength=max_label + 1).astype(np.float64)
        counts[counts == 0] = 1.0  # avoid division by zero

        flattened_channels = fluor.reshape(fluor.shape[0], -1)
        valid_indices = np.flatnonzero(valid)

        means_per_channel = np.zeros((fluor.shape[0], max_label + 1), dtype=np.float64)
        for ch in range(fluor.shape[0]):
            sums = np.bincount(
                label_ids,
                weights=flattened_channels[ch, valid_indices],
                minlength=max_label + 1,
            )
            means = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums),
                where=counts != 0,
            )
            means_per_channel[ch] = means

        self.channel_means = means_per_channel
        mapped = means_per_channel[:, mask_flat]
        return mapped.reshape((fluor.shape[0],) + masks.shape)


def entropy_split(hist: np.ndarray) -> int:
    """
    O(L) maximum-entropy split index using prefix sums.
    - MaxEntropy ported from ImageJ MaxEntropy method: https://imagej.net/ij/plugins/download/Entropy_Threshold.java
        Original description:
        "Automatic thresholding technique based on the entopy of the histogram.
        See: P.K. Sahoo, S. Soltani, K.C. Wong and, Y.C. Chen "A Survey of
        Thresholding Techniques", Computer Vision, Graphics, and Image
        Processing, Vol. 41, pp.233-260, 1988.

        @author Jarek Sacha"
    Parameters
    ----------
    hist : np.ndarray
        1D array of nonnegative counts (length L).

    Returns
    -------
    int
        Histogram bin index t* maximizing H(<=t) + H(>t).
    """
    h = np.asarray(hist, dtype=float)
    total = h.sum()
    if total == 0:
        raise ValueError("Empty histogram: sum of all bins is zero.")

    # Probabilities and their cumulative sum
    p = h / total
    pT = np.cumsum(p)  # P(T <= t)
    pW = 1.0 - pT  # P(T > t)

    # s_i = p_i * log(p_i); define 0*log(0) := 0
    s = np.zeros_like(p)
    nz = p > 0
    s[nz] = p[nz] * np.log(p[nz])

    # Prefix sum S_t = sum_{i<=t} p_i log p_i; total S = sum_i p_i log p_i
    S_prefix = np.cumsum(s)
    S_total = S_prefix[-1]
    S_white = S_total - S_prefix

    # Entropy formulas:
    # hB[t] = - S_prefix[t] / pT[t] + log(pT[t])   (if pT[t] > 0 else 0)
    # hW[t] = - S_white[t]  / pW[t] + log(pW[t])   (if pW[t] > 0 else 0)
    eps = np.finfo(float).tiny
    hB = np.zeros_like(pT)
    hW = np.zeros_like(pW)

    maskB = pT > eps
    maskW = pW > eps

    hB[maskB] = -S_prefix[maskB] / pT[maskB] + np.log(pT[maskB])
    hW[maskW] = -S_white[maskW] / pW[maskW] + np.log(pW[maskW])

    j = hB + hW
    # Guard against any numerical oddities
    j = np.nan_to_num(j, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)

    return int(np.argmax(j))


class GMMTripletBinarize(Binarization):
    """Triplet GMM voting binarizer. Generalized to any multiplet voting."""

    def __init__(
            self,
            fluorophore_channels: np.ndarray,
            cell_data: CellData,
            *,
            sensitivity: float = 0.66,
            confidence_threshold: float = 0.0,
            subset_size: int = 3,
            log_floor: Optional[float] = -2,
            kde_grid_size: int = 512,
            gmm_max_iter: int = 2000,
            gmm_tol: float = 1e-8,
            reg_covar: float = 1e-6,
            random_state: int = 0,
            strict_bimodal: bool = True,
            strict_fit: bool = True,
    ) -> None:
        super().__init__(fluorophore_channels, cell_data)

        if fluorophore_channels.ndim != 3:
            raise ValueError("Expected fluorophore_channels with shape (C, Y, X)")
        if self.cell_data.masks.ndim != 2:
            raise ValueError("Expected masks with shape (Y, X)")
        if fluorophore_channels.shape[1:] != self.cell_data.masks.shape:
            raise ValueError("Image and mask shapes must match")

        self.num_channels = int(fluorophore_channels.shape[0])
        self.subset_size = int(subset_size)
        if self.subset_size < 2:
            raise ValueError("subset_size must be >= 2")
        if self.num_channels < self.subset_size:
            raise ValueError("Need at least subset_size channels for GMM voting")

        self.sensitivity = float(sensitivity)
        self.confidence_threshold = float(confidence_threshold)
        self.log_floor = log_floor
        self.kde_grid_size = int(kde_grid_size)
        self.gmm_max_iter = int(gmm_max_iter)
        self.gmm_tol = float(gmm_tol)
        self.reg_covar = float(reg_covar)
        self.random_state = int(random_state)
        self.strict_bimodal = bool(strict_bimodal)
        self.strict_fit = bool(strict_fit)

        if not (0.0 <= self.sensitivity <= 1.0):
            raise ValueError("sensitivity must be in [0, 1]")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")

        self._labels = self.cell_data.masks.astype(np.int32, copy=False)
        self._cell_ids = np.unique(self._labels)
        self._cell_ids = self._cell_ids[self._cell_ids != 0]
        if self._cell_ids.size == 0:
            raise ValueError("No labeled cells found in masks")

        self.cell_ids = self._cell_ids.copy()
        self.cell_means = self._compute_cell_means()
        self.avg_intensities = self.cell_means
        _store_avg_intensities_in_properties(self.cell_data, self._cell_ids, self.avg_intensities)
        self._last_log_floor_used: Optional[float] = None
        self.binary_matrix: Optional[np.ndarray] = None
        self.likelihood: Optional[np.ndarray] = None
        self.confidence_scores: Optional[np.ndarray] = None
        self.mean_confidence_scores: Optional[np.ndarray] = None
        self.trusted_cells: Optional[np.ndarray] = None
        self.colorcodes: Optional[np.ndarray] = None
        self.thresholds: Optional[np.ndarray] = None

    def _compute_cell_means(self) -> np.ndarray:
        means = [
            np.asarray(ndi.mean(ch, labels=self._labels, index=self._cell_ids), dtype=np.float64)
            for ch in self.fluorophore_channels
        ]
        return np.stack(means, axis=0).T if means else np.empty((0, 0), dtype=np.float64)

    def _fallback_gmm_threshold(self, values: np.ndarray) -> Optional[float]:
        """
        Force a 2-component split when KDE is not clearly bimodal.
        Returns None if a stable split cannot be estimated.
        """
        x = np.asarray(values, dtype=np.float64)
        if x.size < 4 or not np.all(np.isfinite(x)):
            return None

        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return None

        try:
            gmm_1d = GaussianMixture(
                n_components=2,
                covariance_type="full",
                n_init=5,
                max_iter=self.gmm_max_iter,
                tol=self.gmm_tol,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            )
            gmm_1d.fit(x[:, None])
        except Exception:
            return None

        means = gmm_1d.means_.reshape(-1)
        order = np.argsort(means)
        means = means[order]
        weights = gmm_1d.weights_.reshape(-1)[order]
        if np.any(weights <= 1e-6):
            return None

        # Pick the equal-posterior crossing between both component means.
        grid = np.linspace(lo, hi, self.kde_grid_size, dtype=np.float64)
        post = gmm_1d.predict_proba(grid[:, None])[:, order]
        between = (grid >= means[0]) & (grid <= means[1])
        if np.any(between):
            delta = np.abs(post[between, 0] - post[between, 1])
            return float(grid[between][int(np.argmin(delta))])

        return float(np.mean(means))

    def _kde_threshold(self, values: np.ndarray) -> float:
        x = np.asarray(values, dtype=np.float64)
        if x.size < 2 or not np.all(np.isfinite(x)):
            raise ValueError("Input data contains NaN or Inf values")
        if float(np.max(x)) <= float(np.min(x)):
            if self.strict_bimodal:
                raise ValueError("Less than two peaks detected, data may not be bimodal.")
            return float(np.median(x))

        grid = np.linspace(float(np.min(x)), float(np.max(x)), self.kde_grid_size, dtype=np.float64)
        try:
            y = np.asarray(gaussian_kde(x)(grid), dtype=np.float64)
        except Exception:
            fallback_thr = self._fallback_gmm_threshold(x)
            if fallback_thr is not None:
                return fallback_thr
            if self.strict_bimodal:
                raise
            return float(np.median(x))

        peaks, _ = find_peaks(y)

        if peaks.size < 2:
            fallback_thr = self._fallback_gmm_threshold(x)
            if fallback_thr is not None:
                return fallback_thr
            if self.strict_bimodal:
                raise ValueError("Less than two peaks detected, data may not be bimodal.")
            return float(np.median(x))

        top = np.sort(peaks[np.argsort(y[peaks])[-2:]])
        valley = top[0] + int(np.argmin(y[top[0]: top[1] + 1]))
        return float(grid[valley])

    @staticmethod
    def _initial_labels(cc: np.ndarray) -> np.ndarray:
        bits = cc.astype(np.int32, copy=False)
        return np.sum(bits * (1 << np.arange(bits.shape[1], dtype=np.int32)), axis=1).astype(np.int32, copy=False)

    def _fit_subset(self, x_sub: np.ndarray, init0: np.ndarray, seed_offset: int) -> np.ndarray:
        n_comp = 1 << x_sub.shape[1]
        means_init = np.zeros((n_comp, x_sub.shape[1]), dtype=np.float64)
        weights_init = np.zeros(n_comp, dtype=np.float64)
        global_mean = np.mean(x_sub, axis=0)

        for comp in range(n_comp):
            mask = init0 == comp
            if np.any(mask):
                means_init[comp] = np.mean(x_sub[mask], axis=0)
                weights_init[comp] = float(np.mean(mask))
            else:
                means_init[comp] = global_mean
                weights_init[comp] = 1e-6

        weights_init /= np.sum(weights_init)
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="tied",  # full + shared covariance
            n_init=1,
            max_iter=self.gmm_max_iter,
            tol=self.gmm_tol,
            reg_covar=self.reg_covar,
            random_state=self.random_state + seed_offset,
            means_init=means_init,
            weights_init=weights_init,
        )
        gmm.fit(x_sub)
        return gmm.predict(x_sub).astype(np.int32, copy=False)

    def binarize_channels(self) -> np.ndarray:
        if self.cell_means.size == 0:
            return np.zeros_like(self.fluorophore_channels, dtype=bool)

        x = np.asarray(self.cell_means, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("Input data contains NaN or Inf values.")

        floor = np.finfo(np.float64).tiny if self.log_floor is None else float(self.log_floor)
        # Cut away everything below log10 = 0 by enforcing raw intensity >= 1.
        floor = max(floor, 1.0)
        x = np.log10(np.maximum(x, floor))
        self._last_log_floor_used = floor

        thresholds = np.array([self._kde_threshold(x[:, ch]) for ch in range(self.num_channels)], dtype=np.float64)
        input_cc = x >= thresholds[None, :]

        vote_sum = np.zeros((x.shape[0], self.num_channels), dtype=np.float64)
        vote_count = np.zeros((x.shape[0], self.num_channels), dtype=np.int32)

        for t_idx, subset in enumerate(combinations(range(self.num_channels), self.subset_size)):
            subset = np.array(subset, dtype=np.int32)
            init0 = self._initial_labels(input_cc[:, subset])
            try:
                idx_gm = self._fit_subset(x[:, subset], init0, t_idx)
            except Exception as exc:
                if self.strict_fit:
                    raise RuntimeError(f"GMM fit failed for subset {tuple(int(v) for v in subset)}") from exc
                continue

            for local_idx, ch in enumerate(subset):
                vote_sum[:, ch] += ((idx_gm >> local_idx) & 1).astype(np.float64, copy=False)
                vote_count[:, ch] += 1

        if np.any(vote_count == 0):
            raise RuntimeError("Some channel votes are missing, likely because too many subset fits failed.")

        likelihood = vote_sum / vote_count
        binary_matrix = likelihood >= self.sensitivity
        confidence_scores = np.abs(likelihood - 0.5) * 2.0
        mean_confidence_scores = np.mean(confidence_scores, axis=1)
        trusted_cells = np.all(confidence_scores >= self.confidence_threshold, axis=1)
        colorcodes = np.array(
            [int(code) if code else 0 for code in
             ("".join(str(i + 1) for i, b in enumerate(r) if b) for r in binary_matrix)],
            dtype=object,
        )

        self.binary_matrix = binary_matrix
        self.likelihood = likelihood
        self.confidence_scores = confidence_scores
        self.mean_confidence_scores = mean_confidence_scores
        self.trusted_cells = trusted_cells
        self.colorcodes = colorcodes
        self.thresholds = thresholds

        props = self.cell_data.properties
        if "mask_number" in props.columns:
            label_to_row = np.full(int(self._labels.max()) + 1, -1, dtype=np.int32)
            label_to_row[self._cell_ids] = np.arange(self._cell_ids.size, dtype=np.int32)
            mask_numbers = props["mask_number"].to_numpy(dtype=np.int64, copy=False)
            row_idx = np.full(len(props), -1, dtype=np.int32)
            in_bounds = (mask_numbers >= 0) & (mask_numbers < label_to_row.size)
            row_idx[in_bounds] = label_to_row[mask_numbers[in_bounds]]
            valid_rows = row_idx >= 0
        elif len(props) == confidence_scores.shape[0]:
            row_idx = np.arange(confidence_scores.shape[0], dtype=np.int32)
            valid_rows = np.ones(confidence_scores.shape[0], dtype=bool)
        else:
            raise RuntimeError(
                "Cannot align certainty outputs to cell_data.properties: "
                "missing 'mask_number' and row counts differ."
            )

        for ch in range(self.num_channels):
            vals = np.full(len(props), np.nan, dtype=np.float64)
            vals[valid_rows] = confidence_scores[row_idx[valid_rows], ch]
            props[f"certainty_channel_{ch + 1}"] = vals

        mean_vals = np.full(len(props), np.nan, dtype=np.float64)
        mean_vals[valid_rows] = mean_confidence_scores[row_idx[valid_rows]]
        props["mean_certainty"] = mean_vals

        binary_image = np.zeros((self.num_channels, *self._labels.shape), dtype=bool)
        label_to_row = np.full(int(self._labels.max()) + 1, -1, dtype=np.int32)
        label_to_row[self._cell_ids] = np.arange(self._cell_ids.size, dtype=np.int32)
        row_index_image = label_to_row[self._labels]
        valid = row_index_image >= 0
        if np.any(valid):
            flat_valid = valid.ravel()
            rows = row_index_image.ravel()[flat_valid]
            binary_image.reshape(self.num_channels, -1)[:, flat_valid] = binary_matrix[rows].T

        return binary_image