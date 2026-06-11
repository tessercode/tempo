# TEMPO: A Stochastic Benchmarking Protocol for Evaluating Temporal Robustness in Spiking Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](pyproject.toml)

TEMPO is a benchmarking protocol that encodes character classes (A–Z and 0–9) as dual-channel spike trains derived from International Morse code, with configurable human-inspired noise sources. It is designed to probe whether spiking neural networks (SNNs) exploit temporal spike structure rather than mere spike counts.

## Repository Structure

```
tempo/
├── tempo/                  # Python package
│   ├── dataset/
│   │   ├── generate_dataset.py   # Encoder, CLI entry point, HDF5 writer
│   │   └── tempo_dataset.py      # TEMPODataset (PyTorch Dataset)
│   └── utils/              # Utility functions (reserved for future use)
├── notebooks/              # Jupyter notebooks for paper experiments
│   ├── experiment1_ambiguity_calibration.ipynb
│   ├── experiment2_temporal_irreducibility.ipynb
│   ├── experiment3_stochastic_robustness.ipynb
│   ├── experiment4_operator_parameter_validation.ipynb
│   ├── experiment5_architecture_ablation.ipynb
│   ├── visualize_spike_trains.ipynb
│   └── data/               # CW recording label files for experiment 4
│       ├── README.md             # Audio preprocessing and labelling guide
│       ├── 20260312_it9etc iz8vkw_qrs_7030 khz.txt   # Audacity label export (7030 kHz)
│       ├── 20260312_iu2udw iz8vkw_qrs_3557 khz.txt   # Audacity label export (3557 kHz)
│       ├── 20260312_iu7qci iz8vkw_qrs_3557 khz.txt   # Audacity label export (3557 kHz)
│       ├── 20260312_iw5dua iz8vkw_qrs_3557 khz.txt   # Audacity label export (3557 kHz)
│       ├── 20260312_iz0rga iz8vkw_qrs_7030 khz.txt   # Audacity label export (7030 kHz)
│       ├── 20260312_iz4dyx iz8vkw_qrs_3557 khz.txt   # Audacity label export (3557 kHz)
│       └── QRQcw - CFOnet 12-20-2011.txt             # Audacity label export (QRQcw Net)
├── data/                   # Wordlists and pre-built datasets
│   ├── alpha.txt                 # 26-letter alphabet wordlist (A–Z)
│   ├── alphanum.txt              # Alphanumeric wordlist (A–Z, 0–9)
│   ├── top50.txt                 # Top-50 common English words (from NeuroMorse dataset)
│   ├── alpha_stochastic.h5       # Pre-built: 26 classes × 500 samples
│   ├── alphanum_stochastic.h5    # Pre-built: 36 classes × 500 samples
│   └── top50_stochastic.h5       # Pre-built: 50 classes × 500 samples
├── tests/                  # Pytest regression test suite
├── USERGUIDE.md            # Detailed usage guide with API reference
├── CONTRIBUTING.md         # Contributor guidelines
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lhindman/tempo.git
cd tempo
pip install -r requirements.txt
```

For development (adds linting, testing, and notebook tools):

```bash
pip install -r requirements-dev.txt
```

Or install the package in editable mode so that `import tempo` always reflects the current source:

```bash
pip install -e .
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Spike-train encoding and array operations |
| `h5py` | Read/write TEMPO HDF5 dataset files |
| `torch` | PyTorch tensor operations and training |
| `snntorch` | Leaky Integrate-and-Fire neurons and surrogate gradients |

## Pre-built Datasets

Three datasets are included in the `data/` directory. All were generated at 20 WPM, 500 samples per class, with all noise sources enabled (`--all-noise --seed 42`).

| File | Wordlist | Classes | Samples | Size |
|------|----------|---------|---------|------|
| `alpha_stochastic.h5` | A–Z | 26 | 13,000 | 1.6 MB |
| `alphanum_stochastic.h5` | A–Z, 0–9 | 36 | 18,000 | 2.2 MB |
| `top50_stochastic.h5` | Top-50 common English words | 50 | 25,000 | 3.9 MB |

These files can be loaded directly with `TEMPODataset` — no generation step required. To produce datasets with different parameters (WPM, noise configuration, multiplier), see [Generating Datasets](#generating-datasets).

## Quick Start

### Load a pre-built dataset

```python
from tempo.dataset.tempo_dataset import TEMPODataset

ds = TEMPODataset('data/alpha_stochastic.h5')
print(f"Samples: {len(ds)}")           # 13,000
print(f"Classes: {ds.num_classes}")    # 26
spikes, label = ds[0]
print(f"Spike tensor shape: {spikes.shape}")   # [T, 2]
```

For a complete walkthrough including the reference TempoSNN model and training loop, see [USERGUIDE.md](USERGUIDE.md).

## Generating Datasets

The `generate_dataset` module is both a CLI tool and a Python API.

### Command-Line Interface

```bash
python -m tempo.dataset.generate_dataset WORDLIST OUTPUT [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--wpm N` | 20 | Words per minute |
| `--multiplier N` | 1 | Spike trains per word |
| `--weighting` | off | Per-word speed bias ω ~ LogNormal(0.0360, 0.2446) |
| `--dash-ratio` | off | Dash-to-dot ratio variation r ~ LogNormal(1.2269, 0.2916) |
| `--jitter` | off | Gaussian timing jitter σ = 0.575 × T_u |
| `--all-noise` | off | Enable all three noise sources |
| `--seed N` | — | RNG seed for reproducibility |

### Python API

```python
from tempo.dataset.generate_dataset import encode_word, split_channels, generate_dataset

T_U = 1200.0 / 20   # 60 ms at WPM=20

# Encode a single word (deterministic)
spikes = encode_word('SOS', t_u=T_U)
dot_times, dash_times = split_channels(spikes)

# Generate a full dataset
all_dot, all_dash, labels = generate_dataset(
    words=['A', 'B', 'C'],
    wpm=20,
    multiplier=500,
    weighting=True,
    dash_ratio=True,
    jitter=True,
    seed=42,
)
```

## Experiment Notebooks

| Notebook | Description |
|----------|-------------|
| `experiment1_ambiguity_calibration.ipynb` | Verifies that the jointly calibrated parameters σ = 0.575 × T_u and T_thresh = 1.92 × T_u produce balanced cross-channel misclassification within the 8–12% target range |
| `experiment2_temporal_irreducibility.ipynb` | Trains an SNN on standard TEMPO data and evaluates it under conditions that progressively remove temporal structure, confirming that >74% accuracy requires genuine temporal pattern learning |
| `experiment3_stochastic_robustness.ipynb` | Compares models trained on deterministic vs. stochastic TEMPO data across a range of noise levels |
| `experiment4_operator_parameter_validation.ipynb` | Validates TEMPO's stochastic noise parameters against authentic human-keyed Morse code recordings, measuring dot-dash ratio, fist weight, temporal jitter, and sending speed across multiple operators; label files for the analyzed recordings are in `notebooks/data/` |
| `experiment5_architecture_ablation.ipynb` | Ablation study across three SNN architectures (full multi-timescale, no-slow-group, single-timescale) to confirm that the slow LIF group drives graceful degradation under noise and that TEMPO results generalize beyond a single architecture |
| `visualize_spike_trains.ipynb` | Raster plot visualization of TEMPO spike trains |

## Running Tests

```bash
pytest
```

The test suite covers encoding correctness, deterministic timestamps, noise flags, HDF5 round-trips, the CLI, and `TEMPODataset` loading.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use TEMPO in your research, please cite:

```bibtex
@article{hindman2026tempo,
  title={TEMPO: A Stochastic Benchmarking Protocol for Evaluating Temporal Robustness in Spiking Neural Networks},
  author={Hindman, Lucas S. and Cantley, Kurtis D.},
  year={2026}
}
```

## Contact

Luke Hindman — [lukehindman@boisestate.edu](mailto:lukehindman@boisestate.edu)
