from __future__ import annotations

from pathlib import Path

from polytope_barcoding import GMMTripletBinarize, GreedyBarcoding
from polytope_barcoding.core import CellData
from polytope_barcoding.utils.visualisation_tools import (
    read_metadata,
    read_tiff_with_tifffile,
)


def main() -> None:
    """
    Example workflow that binarizes real fluorescence data with the GMM triplet voting model.

    Update the paths below to match your dataset before running.
    """

    tags = [
        "FLAG",
        "HA",
        "V5",
        "T7",
        "VSV-G",
        "AU1",
        "MYC",
        "STAG",
        "HSV",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    experiment_path = repo_root / "example_data" / "HEK_cells_200mgTAM"
    image_name = "W03_200.tif"
    masks_name = "dapi_R3_cp_masks.tif"

    image_path = experiment_path / image_name
    masks_path = experiment_path / masks_name
    mask_z = 0  # Use a single 2D mask plane.

    images = read_tiff_with_tifffile(
        str(image_path),
        tags,
        include_nontags=False,
        order_mode="construct",
    )

    if images is None:
        raise RuntimeError(f"Could not read image at '{image_path}'.")
    print(f"Channel number: {images.shape[0]}")
    _metadata = read_metadata(str(image_path))
    fluorophore_images = images
    cell_data = CellData.load(masks_path=str(masks_path), z=mask_z)

    binarizer = GMMTripletBinarize(
        fluorophore_channels=fluorophore_images,
        cell_data=cell_data,
        random_state=0,
        sensitivity=0.5,
        confidence_threshold=0.0,
        strict_bimodal=True,
        strict_fit=True,
    )
    binary_channels = binarizer.binarize_channels()
    print("GMM triplet binarization complete.")

    GreedyBarcoding(binary_channels=binary_channels, cell_data=cell_data).compute_barcodes()
    print("Greedy barcoding complete.")

    output_csv = experiment_path / "barcodes_gmm_triplet.csv"
    output_masks = experiment_path / "masks_pipeline.tif"
    cell_data.save(str(output_csv), str(output_masks))
    print(f"Saved barcodes to {output_csv}")


if __name__ == "__main__":
    main()
