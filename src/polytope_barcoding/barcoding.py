from .core import Barcoding, CellData
from skimage.measure import regionprops_table
import numpy as np

"""
I need to decide if we want to build this into the binarization

Implementations I still want to do:
- Some barcoding implementation that checks proximal barcodes to improve binarization


"""


class GreedyBarcoding(Barcoding):
    def __init__(self, binary_channels, cell_data: CellData):
        super().__init__(binary_channels, cell_data)

    def compute_barcodes(self):
        """
        Computes the barcodes for each mask based on the binary channels.

        Returns:
            None
        """
        # Get the properties of each region (cell), including coordinates
        # in GreedyBarcoding.compute_barcodes()
        cell_props = regionprops_table(self.cell_data.masks, properties=["label", "coords"])
        unique_labels = cell_props["label"]
        num_labels = len(unique_labels)
        num_channels = self.binary_channels.shape[0]
        barcode_per_mask = np.zeros((num_labels, num_channels), dtype=bool)

        is_3d = (self.binary_channels.ndim == 4 and self.cell_data.masks.ndim == 3)

        for idx, coords in enumerate(cell_props["coords"]):
            coords = np.asarray(coords)
            if coords.ndim != 2 or coords.size == 0:
                continue

            if is_3d:
                z_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]
                cell_pixels = self.binary_channels[:, z_idx, y_idx, x_idx]  # (C, N)
            else:
                y_idx, x_idx = coords[:, 0], coords[:, 1]
                cell_pixels = self.binary_channels[:, y_idx, x_idx]  # (C, N)

            barcode_per_mask[idx, :] = np.any(cell_pixels, axis=1)

        # write barcode_per_mask back to self.cell_data.properties as before

        # Add the barcodes to the DataFrame
        self.cell_data.add_barcodes_to_df(barcode_per_mask)
