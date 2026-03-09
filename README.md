# Polytope Nuclear Barcoding

Utilities for binarization and barcode extraction on Polytope imaging data.

## Installation

```bash
pip install polytope_barcoding
```

## Getting Started

Run the examples from the repository root:

```bash
py -3 examples/barcoding_with_gmm.py
py -3 examples/certainty_shell_error_sweep.py
```

`certainty_shell_error_sweep.py` can also use your own CSV:

```bash
py -3 examples/certainty_shell_error_sweep.py --csv path/to/barcodes.csv
```

## Data Notes

- Input is expected to be multichannel Polytope tag images and a corresponding mask image.
- Typical tag panel: `DNA, FLAG, HA, V5, T7, VSV-G, AU1, Myc, S-tag, HSV`.
- Channels should be as clean/unmixed as possible before binarization.

## Contact

Jeroen Doornbos  
j.doornbos@nki.nl  
jeroendoornbos98@gmail.com

Tom van Leeuwen  
t.v.leeuwen@nki.nl  
tom@tleeuwen.nl