import numpy as np
import pandas as pd
import tifffile
import warnings
from abc import ABC, abstractmethod
from scipy import ndimage as ndi


class CellData:
    """
    Base class for cells.

    This object should contain the properties of each cell, as well as the cell_data that localize the cells in the image.
    All other classes should use this to pass around the cell information.
    """

    def __init__(self, masks: np.ndarray, crop: tuple = None):
        """
        Initialize the CellData object with masks and optionally crop the masks.

        Args:
            masks: A numpy array containing the mask data.
            crop: A tuple specifying the crop region (y1, y2, x1, x2). If None, no cropping is applied.
        """
        if crop:
            masks = self._crop_masks(masks, crop)
            warnings.warn(
                "Currently, cropping & reloading barcodes is not supported. Please reannotate."
            )
        if masks.dtype == "float32":
            masks = masks.astype("int32")
        self.masks = masks
        self.properties = self._generate_properties()
        self.binary_channels = None

    def _crop_masks(self, masks: np.ndarray, crop: tuple) -> np.ndarray:
        """
        Crop the masks array to the specified region.

        Args:
            masks: The original masks array.
            crop: A tuple (y1, y2, x1, x2) specifying the region to crop.

        Returns:
            np.ndarray: The cropped masks array.
        """
        y1, y2, x1, x2 = crop
        return masks[y1:y2, x1:x2]

    def _generate_properties(self) -> pd.DataFrame:
        """
        Generate properties of the cells from the cell_data. The properties extracted are the x & y location and the mask
        number. It creates a dataframe with these properties. The dataframe will later store other properties such as
        barcodes, cluster labels, etc.

        Returns: A pandas DataFrame with the properties of the cells.
        """
        labels = np.asarray(self.masks)
        if labels.size == 0:
            if labels.ndim == 3:
                return pd.DataFrame(columns=["mask_number", "z", "y", "x"])
            return pd.DataFrame(columns=["mask_number", "y", "x"])

        mask_numbers = np.unique(labels)
        mask_numbers = mask_numbers[mask_numbers != 0]
        if mask_numbers.size == 0:
            if labels.ndim == 3:
                return pd.DataFrame(columns=["mask_number", "z", "y", "x"])
            return pd.DataFrame(columns=["mask_number", "y", "x"])

        # Use the label image itself as intensity input to avoid allocating large temporary arrays.
        # For each region, label values are constant, so weighted and unweighted centroids are identical.
        centroids = ndi.center_of_mass(labels, labels=labels, index=mask_numbers)
        centroids_arr = np.atleast_2d(np.asarray(centroids, dtype=np.float64))

        if labels.ndim == 3:
            coord_cols = ["z", "y", "x"]
        elif labels.ndim == 2:
            coord_cols = ["y", "x"]
        else:
            coord_cols = [f"axis_{axis}" for axis in range(labels.ndim)]

        df = pd.DataFrame(
            np.column_stack([mask_numbers, centroids_arr]),
            columns=["mask_number", *coord_cols],
        )
        return df

    def add_string_barcodes_to_df(self):

        extracted_barcodes = self.properties.filter(like="barcode_channel").values
        concat = []
        for barcode in extracted_barcodes:
            concat.append(["".join("+" if bit else "-" for bit in barcode)][0])
        self.properties["barcode_string"] = concat

    def add_barcodes_to_df(self, barcodes: np.ndarray):
        """
        Add barcodes to the properties DataFrame.

        Args:
            barcodes: A numpy array containing the barcodes for each mask.
        """
        for i in range(barcodes.shape[1]):
            self.properties[f"barcode_channel_{i + 1}"] = barcodes[:, i]

        self.add_string_barcodes_to_df()

    def save(self, properties_path: str, masks_path: str):
        """
        Save the properties to a CSV file and the cell_data to a TIFF file.

        Args:
            properties_path: Path to save the properties CSV.
            masks_path: Path to save the cell_data TIFF.
        """
        self.properties.to_csv(properties_path, index=False)
        if masks_path:
            tifffile.imwrite(masks_path, self.masks)

    def signal_filter(self, img_path=None, ch=None, thr=None):
        """
        Filter masks by an external signal source. Automatically selects the cell masks that exceed the set threshold
        for further processing.

        Args:
            img_path: Path to an external image, must have the same XYZ shape as the input image.
            ch: Which channel(s) to analyze. Can be an integer or tuple of integers.
            thr: Threshold for the signal(s). Can be an integer or tuple of integers. Any cells with average values
            exceeding the threshold will be used in further processing.

        Returns:
            Optionally returns a pandas DataFrame with all the mean values for all cells (regardless of threshold) per
            channel.
        """

        if type(ch) != tuple:
            ch = [ch]
        if type(thr) != tuple:
            thr = [thr]
        if len(ch) != len(thr):
            raise ValueError('ch and thr must have same length')

        if img_path is not None:
            cell_ids = np.unique(self.masks)
            cell_ids = cell_ids[cell_ids != 0]
            keep = np.zeros((len(ch), len(cell_ids)), dtype=bool)
            cell_means = []
            with tifffile.TiffFile(img_path) as tif:
                for i, c in enumerate(ch):
                    img = tif.asarray()[:,c]  # ZCYX
                    means = ndi.mean(img, labels=self.masks, index=cell_ids)
                    cell_means.append(means)
                    keep[i] = means > thr[i]
            keep = keep.prod(axis=0, dtype=bool)
            keep_ids = cell_ids[keep]  # Which cells to keep
            drop_ids = cell_ids[~keep]  # And which to drop
            self.masks[np.isin(self.masks, drop_ids)] = 0  # Set dropped cells to 0
            # Fancy line to keep the cells with the passing grade, aligning the filtered masks and properties
            self.properties = self.properties.loc[self.properties['mask_number'].isin(keep_ids)]
            cell_means = np.array(cell_means).T

            return pd.DataFrame(cell_means)

    @classmethod
    def load(
        cls,
        masks_path: str = None,
        barcode_path: str = None,
        z=None,
        crop: tuple = None,
    ):
        """
        Load cell_data from a TIFF file and optionally barcodes from a CSV file, with optional cropping.

        Barcodes are assigned to the cell_data based on the x and y coordinates in the CSV file, using the masks check
        if the x and y coordinates fall within a mask. If 2 labels are present in the same mask, the barcode is assigned
        to the last label.

        Args:
            masks_path: Optional path to the cell_data TIFF file.
            barcode_path: Optional path to the CSV file containing barcodes.
            z: What z-slice to use. Set to 0 for 2D usage.
            crop: Optional tuple specifying the crop region (y1, y2, x1, x2).

        Returns:
            CellData: A new instance of CellData with loaded properties and cell_data.
        """
        if masks_path:
            masks = tifffile.imread(masks_path)
            if masks.dtype == "float32":
                masks = masks.astype("int32")
            if len(masks.shape) > 2:
                if z is not None:
                    masks = masks[z, ...]
            instance = cls(masks, crop=crop)
        else:
            raise ValueError(
                "masks_path must be provided to load masks if no existing mask is present."
            )

        if barcode_path:
            instance.load_barcodes(barcode_path)

        return instance

    def load_barcodes(self,
                      barcode_path: str | None = None,
                      barcodes: pd.DataFrame | None = None) -> None:
        if barcode_path is not None:
            barcodes = pd.read_csv(barcode_path)

        if barcodes is None or barcodes.empty:
            return

        # Add any new columns to self.properties once, up front
        new_cols = [c for c in barcodes.columns if c not in {"x", "y"}]
        for c in new_cols:
            if c not in self.properties.columns:
                self.properties[c] = np.nan

        # Cast coordinates to int32 so we can use them as indices
        xy = barcodes[["y", "x"]].astype(np.int32).to_numpy()  # shape (n, 2)
        y_arr, x_arr = xy[:, 0], xy[:, 1]

        h, w = self.masks.shape
        in_bounds = (x_arr >= 0) & (x_arr < w) & (y_arr >= 0) & (y_arr < h)

        # Warn once (not per row) about out‑of‑image annotations
        if (~in_bounds).any():
            bad_idx = np.flatnonzero(~in_bounds).tolist()
            warnings.warn(
                f"Annotations at indices {bad_idx} are out of image bounds."
            )

        # Keep only in‑bounds records
        barcodes = barcodes.loc[in_bounds].copy()
        y_arr, x_arr = y_arr[in_bounds], x_arr[in_bounds]

        mask_numbers = self.masks[y_arr, x_arr]
        barcodes["mask_number"] = mask_numbers

        # Discard mask_number == 0 in one shot
        barcodes = barcodes.loc[barcodes["mask_number"] != 0]

        # -------------------------------------------
        # 3. Build a frame keyed by mask_number
        # -------------------------------------------
        # We assume at most one barcode per mask; if not, decide which row wins
        value_cols = [c for c in barcodes.columns
                      if c not in {"x", "y", "mask_number", "barcode_string"}]

        # make sure only those go into self.properties (add if missing)
        for c in value_cols:
            if c not in self.properties.columns:
                self.properties[c] = np.nan

        merged = (
            barcodes
            .loc[~barcodes.duplicated("mask_number", keep="first"),
            ["mask_number"] + value_cols]
            .set_index("mask_number")
        )

        # --------------------------------------------
        # 4. Left‑join onto self.properties in bulk
        # --------------------------------------------
        self.properties.set_index("mask_number", inplace=True)
        self.properties.update(merged)  # inplace, keeps dtype
        self.properties.reset_index(inplace=True)

        # --------------------------------------------
        # 5. Report unused / unassigned information
        # --------------------------------------------
        used_masks = set(merged.index)
        all_masks = set(self.properties["mask_number"])
        unassigned_masks = all_masks - used_masks
        if unassigned_masks:
            warnings.warn(
                f"Masks {sorted(unassigned_masks)} were not assigned any barcodes. They will be set to 0"
            )

        # Any barcodes that fell out above are “unused”
        unused_annotations = barcodes.index.difference(merged.index)
        if len(unused_annotations):
            warnings.warn(
                f"Annotations at indices {sorted(unused_annotations)} were not used."
            )

        self.properties.fillna(0, inplace=True)

        # --------------------------------------------------
        # 6. Recompute barcode_string from channel columns
        # --------------------------------------------------
        channel_cols = sorted(
            [c for c in self.properties.columns if c.startswith("barcode_channel_")],
            key=lambda c: int(c.rsplit("_", 1)[1])  # numeric order 1,2,3,...
        )

        if channel_cols:
            self.properties["barcode_string"] = (
                self.properties[channel_cols]
                .astype(int)
                .apply(lambda r: "".join("+" if v else "-" for v in r), axis=1)
            )


class CellSegmentation(ABC):
    """
    Base class for cell segmentation
    """

    def __init__(self, dna_channel, metadata=None):
        self.dna_channel = dna_channel
        self.metadata = metadata

    @abstractmethod
    def segment(self): ...


class Binarization(ABC):
    """
    Abstract base class for the binarization of fluorescence signal.
    """

    def __init__(self, fluorophore_channels, cell_data: CellData):
        self.fluorophore_channels = fluorophore_channels
        self.cell_data = cell_data

    @abstractmethod
    def binarize_channels(self):
        """

        Returns: A binary list of 'cells' by 'channels'

        """
        ...


class Barcoding(ABC):
    """
    Abstract base class for determining barcodes on segmented cell_data using binary fluorescence channels.

    """

    def __init__(self, binary_channels, cell_data: CellData):
        self.binary_channels = binary_channels
        self.cell_data = cell_data

    @abstractmethod
    def compute_barcodes(self):
        """

        Returns: A list of barcodes per mask.

        """
        ...


class Clustering(ABC):
    """
    Abstract baseclass for Clustering groups of the same barcodes into seperate clusters.
    This can be used when certain barcodes
    """
