# TEMPO User Guide

**TEMPO** — A Stochastic Benchmarking Protocol for Evaluating Temporal Robustness in Spiking Neural Networks

This guide walks you through installing the `tempo` package, generating datasets, loading them into PyTorch / snnTorch, and training the reference TempoSNN model.

---

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Generating Datasets](#generating-datasets)
   - [Command-line Interface](#command-line-interface)
   - [Python API](#python-api)
4. [The HDF5 File Format](#the-hdf5-file-format)
5. [Loading Datasets with TEMPODataset](#loading-datasets-with-tempodataset)
6. [In-Memory Encoding (No HDF5)](#in-memory-encoding-no-hdf5)
7. [Training a TempoSNN](#training-a-temposnn)
8. [Evaluation](#evaluation)
9. [API Reference](#api-reference)

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/tessercode/tempo.git
cd tempo
pip install -r requirements.txt
```

For development (adds linting, testing, and notebook tools):

```bash
pip install -r requirements-dev.txt
```

Or install the package in editable mode so that `import tempo` always reflects
the current source:

```bash
pip install -e .
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `h5py` | Read/write TEMPO HDF5 dataset files |
| `numpy` | Spike-train encoding and array operations |
| `torch` | PyTorch tensor operations and training |
| `snntorch` | Leaky Integrate-and-Fire neurons and surrogate gradients |

---

## Core Concepts

### The TEMPO Encoding Protocol

TEMPO encodes each character class (A–Z and 0–9) as a **dual-channel spike train** derived from International Morse code.  Each Morse mark (dot or dash) produces exactly one spike:

- **Channel 0 (dot channel)**: spikes whose mark duration is shorter than the threshold `T_thresh = 1.92 × T_u`
- **Channel 1 (dash channel)**: spikes whose mark duration is equal to or longer than `T_thresh`

The base time unit `T_u = 1200 / WPM` milliseconds sets the overall speed.  At the default `WPM = 20`, `T_u = 60 ms`.

```
Standard timing:
  Dot  duration:         1.0 × T_u   =  60 ms
  Dash duration:         3.0 × T_u   = 180 ms
  Intra-character gap:   1.0 × T_u   =  60 ms
  Inter-character gap:   3.0 × T_u   = 180 ms
  Inter-word gap:        7.0 × T_u   = 420 ms
```

### Human-Inspired Noise Sources

Three optional noise sources model the variability present in human-operated Morse:

| Flag | Parameter | Description |
|------|-----------|-------------|
| `weighting` | ω ~ LogNormal(0.0360, 0.2446) | Per-word speed bias: a single scale factor applied to every timing interval in that word |
| `dash_ratio` | r ~ LogNormal(1.2269, 0.2916) | Dash-to-dot ratio variation: replaces the standard 3× ratio with a random value |
| `jitter` | σ = 0.575 × T_u | Gaussian timing noise: added independently to each mark and gap |

All three flags together produce the **full stochastic TEMPO encoding**, which is the recommended setting for training temporally robust SNNs.

---

## Generating Datasets

### Command-line Interface

The fastest way to produce a TEMPO dataset is via the `generate_dataset` module entry point.  It reads a wordlist, encodes every word, and writes the result to an HDF5 file.

**Basic usage — 26-letter alphabet, deterministic timing:**

```bash
python -m tempo.dataset.generate_dataset \
    data/alpha.txt \
    data/alpha_det.h5 \
    --wpm 20 \
    --multiplier 500
```

**Full stochastic encoding (recommended for training):**

```bash
python -m tempo.dataset.generate_dataset \
    data/alpha.txt \
    data/alpha_stochastic.h5 \
    --wpm 20 \
    --multiplier 500 \
    --all-noise \
    --seed 42
```

`--all-noise` is shorthand for `--weighting --dash-ratio --jitter`.

**Including digits (A–Z plus 0–9):**

```bash
python -m tempo.dataset.generate_dataset \
    data/alphanum.txt \
    data/alphanum_stochastic.h5 \
    --wpm 20 \
    --multiplier 500 \
    --all-noise \
    --seed 42
```

**Using a custom wordlist:**

The wordlist file is plain text with words separated by whitespace (spaces, newlines, or tabs).  All characters must appear in the Morse table (A–Z, 0–9); any word containing unsupported characters is skipped with a warning.

```
# my_words.txt
HELLO WORLD TEMPO SNN SPIKE TRAIN
```

```bash
python -m tempo.dataset.generate_dataset my_words.txt data/custom.h5 --multiplier 100 --all-noise
```

**Full CLI options:**

```
positional arguments:
  wordlist              Path to wordlist file (whitespace-separated words)
  output                Output HDF5 file path

options:
  --wpm WPM             Words per minute (default: 20)
  --multiplier N        Spike trains per word (default: 1)
  --weighting           Enable per-word speed bias (ω ~ LogNormal(0.0360, 0.2446))
  --dash-ratio          Enable dash-to-dot ratio variation (r ~ LogNormal(1.2269, 0.2916))
  --jitter              Enable Gaussian timing jitter (σ = 0.575 × T_u)
  --all-noise           Enable all three noise sources
  --seed SEED           Integer RNG seed for reproducibility
```

---

### Python API

The same functionality is available programmatically via `tempo.dataset.generate_dataset`.

#### Encoding a single word

```python
import numpy as np
from tempo.dataset.generate_dataset import encode_word, split_channels, MORSE_TABLE

WPM = 20
T_U = 1200.0 / WPM   # 60 ms

# Deterministic encoding (no noise)
spikes = encode_word('HELLO', t_u=T_U)
dot_times, dash_times = split_channels(spikes)

print(f"'HELLO' → {len(spikes)} spikes total")
print(f"  dot channel:  {len(dot_times)} spikes at {dot_times.round(1)} ms")
print(f"  dash channel: {len(dash_times)} spikes at {dash_times.round(1)} ms")
```

Expected output:
```
'HELLO' → 9 spikes total
  dot channel:  6 spikes at [  60.  300.  360.  420.  840.  900.] ms
  dash channel: 3 spikes at [ 180.  540.  720.] ms
```

#### Encoding with noise

```python
rng = np.random.default_rng(seed=42)

spikes_noisy = encode_word(
    'HELLO',
    t_u=T_U,
    weighting=True,    # ω ~ LogNormal(0.0360, 0.2446)
    dash_ratio=True,   # r ~ LogNormal(1.2269, 0.2916)
    jitter=True,       # σ = 0.575 × T_u
    rng=rng,
)
dot_times, dash_times = split_channels(spikes_noisy)
print(f"Noisy: {len(dot_times)} dot spikes, {len(dash_times)} dash spikes")
```

#### Generating a full dataset in Python

```python
from tempo.dataset.generate_dataset import generate_dataset, write_hdf5

# Load wordlist from the bundled alpha.txt
import pathlib
alpha_path = pathlib.Path('data/alpha.txt')
words = [w.upper() for w in alpha_path.read_text().split()]

# Generate 500 stochastic samples per letter (A–Z)
all_dot_times, all_dash_times, labels = generate_dataset(
    words=words,
    wpm=20,
    multiplier=500,
    weighting=True,
    dash_ratio=True,
    jitter=True,
    seed=42,
)

print(f"Total samples: {len(labels)}")          # 13,000
print(f"Unique classes: {len(set(labels))}")    # 26
print(f"Dot spikes for first sample: {all_dot_times[0]}")
```

#### Writing to HDF5

```python
write_hdf5(
    filepath='data/alpha_stochastic.h5',
    all_dot_times=all_dot_times,
    all_dash_times=all_dash_times,
    labels=labels,
    wpm=20,
    multiplier=500,
    weighting=True,
    dash_ratio=True,
    jitter=True,
    seed=42,
    wordlist_path=str(alpha_path),
)
print("Dataset written to data/alpha_stochastic.h5")
```

---

## The HDF5 File Format

TEMPO datasets are stored in HDF5 with the following structure:

```
/spikes/
    dot_channel   (N_samples,)  vlen float64  — dot spike timestamps in ms
    dash_channel  (N_samples,)  vlen float64  — dash spike timestamps in ms
/labels           (N_samples,)  UTF-8 string  — class label (e.g. 'A', 'HELLO')
/metadata/                      group          — dataset provenance
    protocol_version  "TEMPO"
    wpm               int
    unit_ms           float  (= 1200 / wpm)
    multiplier        int
    weighting_enabled bool
    dash_ratio_enabled bool
    jitter_enabled    bool
    seed              int  (-1 if not set)
    wordlist_path     str
    timescale         "1ms"
    encoding_type     "Mark Termination (Dual Channel)"
    timestamp         ISO 8601 creation time
```

Because each spike train has a different number of spikes, the channel arrays use
HDF5 variable-length (`vlen`) dtype — each element is a ragged float64 array.
All timestamps are in **milliseconds** relative to the start of the first Morse
symbol in that sample.

You can inspect a file with the `h5py` library directly:

```python
import h5py

with h5py.File('data/alpha_stochastic.h5', 'r') as f:
    # Read metadata
    for key, val in f['metadata'].attrs.items():
        print(f"  {key}: {val}")

    # Inspect one sample
    i = 0
    print(f"\nSample {i}: label='{f['labels'][i].decode()}'")
    print(f"  dot  spikes: {f['spikes/dot_channel'][i]}")
    print(f"  dash spikes: {f['spikes/dash_channel'][i]}")
```

---

## Loading Datasets with TEMPODataset

`TEMPODataset` is a `torch.utils.data.Dataset` subclass that reads an HDF5 file
and converts the sparse spike timestamps into dense binary tensors suitable for
snnTorch.

### Basic usage

```python
from tempo.dataset.tempo_dataset import TEMPODataset
from torch.utils.data import DataLoader

# Load the dataset — time dimension is derived automatically from the data
dataset = TEMPODataset('data/alpha_stochastic.h5')

print(f"Samples:        {len(dataset)}")           # 13,000
print(f"Classes:        {dataset.num_classes}")    # 26
print(f"Word → ID map:  {dataset.word_to_id}")
print(f"Time steps:     {dataset.spikes.shape[1]}")  # max T in the data

# Single sample
spikes, label = dataset[0]
print(f"\nspikes shape: {spikes.shape}")   # [T, 2]
print(f"label:        {label.item()} → '{dataset.id_to_word[label.item()]}'")
```

### Fixing the time dimension

When training on multiple datasets (e.g. train / validation / test), all sets
must share the same time dimension.  Pass `max_time_steps` explicitly:

```python
T_MAX = 1600   # 1600 ms at 1 ms resolution

train_ds = TEMPODataset('data/train.h5',      max_time_steps=T_MAX)
val_ds   = TEMPODataset('data/val.h5',        max_time_steps=T_MAX)
test_ds  = TEMPODataset('data/test_sigma0.h5', max_time_steps=T_MAX)
```

Spikes beyond `max_time_steps` are silently discarded; samples are zero-padded
to the specified length if they are shorter.

### Creating DataLoaders

```python
train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    pin_memory=True,     # faster CPU→GPU transfer
    num_workers=4,
    persistent_workers=True,
)

val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
```

Each batch yields:
- `spikes`:  `torch.float32` tensor of shape `[Batch, Time, 2]`
- `labels`:  `torch.long` tensor of shape `[Batch]`

**snnTorch expects time as the first dimension.**  Permute before passing to
the network:

```python
for spikes, labels in train_loader:
    # spikes: [B, T, 2]  →  [T, B, 2]  (snnTorch convention)
    spikes = spikes.permute(1, 0, 2).to(device)
    labels = labels.to(device)
    output = model(spikes)   # [B, n_classes]
```

---

## In-Memory Encoding (No HDF5)

You can build tensors directly from `encode_word` or `generate_dataset`, skipping the HDF5 intermediate entirely.  This is useful for rapid prototyping, online data augmentation, and custom noise schedules.

### Converting spike times to a dense tensor

```python
import numpy as np
import torch
from tempo.dataset.generate_dataset import encode_word, split_channels

def spikes_to_tensor(dot_times, dash_times, max_time_steps):
    """Convert dot/dash timestamp arrays into a [T, 2] binary tensor."""
    tensor = torch.zeros(max_time_steps, 2, dtype=torch.float32)
    for t in dot_times:
        idx = int(round(t))
        if 0 <= idx < max_time_steps:
            tensor[idx, 0] = 1.0
    for t in dash_times:
        idx = int(round(t))
        if 0 <= idx < max_time_steps:
            tensor[idx, 1] = 1.0
    return tensor

T_MAX = 1600
T_U   = 60.0     # WPM = 20
rng   = np.random.default_rng(seed=0)

spikes    = encode_word('MORSE', t_u=T_U, weighting=True, dash_ratio=True, jitter=True, rng=rng)
dot_t, dash_t = split_channels(spikes)
x = spikes_to_tensor(dot_t, dash_t, max_time_steps=T_MAX)  # [1600, 2]
```

### Building a complete in-memory dataset

```python
from tempo.dataset.generate_dataset import generate_dataset
import pathlib

alpha_path = pathlib.Path('data/alpha.txt')
words = [w.upper() for w in alpha_path.read_text().split()]

# Generate dataset (returns lists of numpy arrays)
all_dot, all_dash, labels = generate_dataset(
    words=words,
    wpm=20,
    multiplier=500,
    weighting=True,
    dash_ratio=True,
    jitter=True,
    seed=42,
)

# Build word → integer mapping (sorted for reproducibility)
unique_words = sorted(set(labels))
word_to_id   = {w: i for i, w in enumerate(unique_words)}
id_to_word   = {i: w for w, i in word_to_id.items()}

T_MAX = 1600
n_samples = len(labels)

# Allocate tensors
X = torch.zeros(n_samples, T_MAX, 2, dtype=torch.float32)
y = torch.tensor([word_to_id[w] for w in labels], dtype=torch.long)

for i, (dot_t, dash_t) in enumerate(zip(all_dot, all_dash)):
    for t in dot_t:
        idx = int(round(t))
        if 0 <= idx < T_MAX:
            X[i, idx, 0] = 1.0
    for t in dash_t:
        idx = int(round(t))
        if 0 <= idx < T_MAX:
            X[i, idx, 1] = 1.0

print(f"X shape: {X.shape}")   # [13000, 1600, 2]
print(f"y shape: {y.shape}")   # [13000]
```

Once you have `X` and `y` you can wrap them in a `TensorDataset`:

```python
from torch.utils.data import TensorDataset, DataLoader

ds     = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True)
```

---

## Training a TempoSNN

The reference TempoSNN architecture from the TEMPO paper is reproduced below
(Option B — multi-timescale LIF with soft reset, 3,742 parameters).

### Model definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

spike_grad = surrogate.fast_sigmoid()


class TempoSNN(nn.Module):
    """Multi-timescale LIF network for TEMPO classification.

    Three neuron groups share a single input projection (fc1) but operate
    at different membrane decay constants, giving the network simultaneous
    access to fast (~20 ms), medium (~100 ms), and slow (~500 ms) temporal
    scales.

    Input:  [T, B, 2]   (time × batch × channels)
    Output: [B, n_out]  (accumulated output spike counts, one per class)
    """

    # (n_neurons, beta, threshold)
    GROUPS = [
        (42,  0.95,  0.3),    # fast:   τ ~  20 ms  — local mark detection
        (43,  0.99,  0.4),    # medium: τ ~ 100 ms  — intra-character context
        (43,  0.998, 0.5),    # slow:   τ ~ 500 ms  — full-letter integration
    ]

    def __init__(self, n_in=2, n_out=26):
        super().__init__()
        n_hid = sum(g[0] for g in self.GROUPS)   # 128

        self.fc1 = nn.Linear(n_in, n_hid)
        # Wide uniform init ensures neurons fire on the first spike
        # (necessary for the sparse 2-channel input)
        nn.init.uniform_(self.fc1.weight, -2.0, 2.0)
        nn.init.zeros_(self.fc1.bias)

        self.lifs = nn.ModuleList([
            snn.Leaky(
                beta=b,
                learn_beta=True,
                threshold=thr,
                reset_mechanism='subtract',   # soft reset
                spike_grad=spike_grad,
            )
            for (_, b, thr) in self.GROUPS
        ])

        self.fc2  = nn.Linear(n_hid, n_out)
        self.lif2 = snn.Leaky(
            beta=0.95,
            learn_beta=True,
            reset_mechanism='subtract',
            spike_grad=spike_grad,
        )
        self._sizes = [g[0] for g in self.GROUPS]

    def forward(self, x):
        """Forward pass.

        Args:
            x: [T, B, 2] input spike tensor (float32).

        Returns:
            [B, n_out] accumulated output spike count (rate code).
        """
        T, B, C = x.shape

        # Batch fc1 across all time steps: one matmul instead of T small ones
        cur_all = self.fc1(x.reshape(T * B, C)).reshape(T, B, -1)   # [T, B, 128]

        mems     = [lif.init_leaky() for lif in self.lifs]
        spk1_all = []

        for t in range(T):
            cur = cur_all[t]           # [B, 128]
            groups, offset = [], 0
            for i, (lif, n) in enumerate(zip(self.lifs, self._sizes)):
                spk_i, mems[i] = lif(cur[:, offset:offset + n], mems[i])
                groups.append(spk_i)
                offset += n
            spk1_all.append(torch.cat(groups, dim=1))   # [B, 128]

        # Batch fc2 across all time steps
        spk1_seq = torch.stack(spk1_all, dim=0)                          # [T, B, 128]
        cur2_all = self.fc2(spk1_seq.reshape(T * B, -1)).reshape(T, B, -1)

        mem2     = self.lif2.init_leaky()
        spk2_acc = None
        for t in range(T):
            spk2, mem2 = self.lif2(cur2_all[t], mem2)
            spk2_acc   = spk2 if spk2_acc is None else spk2_acc + spk2

        return spk2_acc   # [B, 26] — total output spike count


# Quick sanity check
model = TempoSNN()
n_params = sum(p.numel() for p in model.parameters())
print(f"TempoSNN parameters: {n_params:,}")   # 3,742
```

### Training loop

```python
import contextlib
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model     = TempoSNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-5
)

# Assumes X [N, T, 2] and y [N] built from the in-memory section above,
# or loaded via TEMPODataset into tensors.
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256,
    shuffle=True,
    pin_memory=(device.type == 'cuda'),
    num_workers=4,
    persistent_workers=(device.type == 'cuda'),
)

N_EPOCHS = 100

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    total_loss = correct = total = 0

    for xb, yb in train_loader:
        # Permute to snnTorch convention: [B, T, 2] → [T, B, 2]
        xb = xb.permute(1, 0, 2).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        out  = model(xb)                        # [B, 26]
        loss = F.cross_entropy(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)

    scheduler.step()

    if epoch % 10 == 0:
        train_acc = correct / total
        print(f"Epoch {epoch:3d}/{N_EPOCHS}  "
              f"loss={total_loss/total:.4f}  acc={train_acc*100:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
```

---

## Evaluation

```python
def evaluate(model, dataset, device, batch_size=512):
    """Return accuracy (float in [0, 1]) for a TEMPODataset."""
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         pin_memory=(device.type == 'cuda'))
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.permute(1, 0, 2).to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out     = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total


# Example: evaluate on multiple noise levels
test_files = {
    'σ=0.00': 'data/test_sigma_0.00.h5',
    'σ=0.84': 'data/test_sigma_0.84.h5',
    'σ=1.20': 'data/test_sigma_1.20.h5',
}

for label, path in test_files.items():
    ds  = TEMPODataset(path, max_time_steps=T_MAX)
    acc = evaluate(model, ds, device)
    print(f"  {label}: {acc*100:.1f}%")
```

### Interpreting results

| Accuracy range | Interpretation |
|----------------|----------------|
| ~3.85% | Chance level (uniform random over 26 classes) |
| ~26.9% | Count-only ceiling (using only dot/dash spike counts, no timing) |
| >74%   | Temporal pattern learning confirmed (Exp2 intact benchmark) |

A model that scores well above the count-only ceiling (26.9%) is demonstrably
exploiting the temporal structure of the spike trains, not just the rate code.

---

## API Reference

### `tempo.dataset.generate_dataset`

#### `encode_word(word, t_u, weighting=False, dash_ratio=False, jitter=False, rng=None)`

Encode a single uppercase word into a dual-channel spike train.

| Parameter | Type | Description |
|-----------|------|-------------|
| `word` | `str` | Uppercase word (characters must be in `MORSE_TABLE`) |
| `t_u` | `float` | Base time unit in ms (`= 1200 / WPM`) |
| `weighting` | `bool` | Enable per-word speed bias ω ~ LogNormal(0.0360, 0.2446) |
| `dash_ratio` | `bool` | Enable dash ratio variation r ~ LogNormal(1.2269, 0.2916) |
| `jitter` | `bool` | Enable Gaussian jitter σ = 0.575 × T_u |
| `rng` | `np.random.Generator` | NumPy RNG; created fresh if `None` |

Returns `List[Tuple[float, int]]` — a list of `(timestamp_ms, channel)` pairs,
where `channel` is `0` (dot) or `1` (dash).

---

#### `split_channels(spikes)`

Split a spike list into per-channel timestamp arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `spikes` | `List[Tuple[float, int]]` | Output of `encode_word` |

Returns `(dot_times, dash_times)` as `numpy.ndarray` of `float64`.

---

#### `generate_dataset(words, wpm, multiplier=1, weighting=False, dash_ratio=False, jitter=False, seed=None)`

Generate a full dataset for a list of words.

| Parameter | Type | Description |
|-----------|------|-------------|
| `words` | `List[str]` | Uppercase word strings |
| `wpm` | `int` | Words per minute |
| `multiplier` | `int` | Spike trains per word (dataset size = `len(words) × multiplier`) |
| `weighting` | `bool` | Enable speed bias noise |
| `dash_ratio` | `bool` | Enable dash ratio noise |
| `jitter` | `bool` | Enable Gaussian jitter |
| `seed` | `int \| None` | RNG seed for reproducibility |

Returns `(all_dot_times, all_dash_times, labels)` where each of the first two is
a `List[np.ndarray]` and `labels` is a `List[str]`.

---

#### `write_hdf5(filepath, all_dot_times, all_dash_times, labels, wpm, multiplier, weighting, dash_ratio, jitter, seed, wordlist_path)`

Write a dataset to a TEMPO HDF5 file.  All parameters match the outputs
and settings from `generate_dataset`.

---

#### `MORSE_TABLE`

`Dict[str, str]` — International Morse code for A–Z and 0–9.

```python
from tempo.dataset.generate_dataset import MORSE_TABLE
print(MORSE_TABLE['S'])   # '...'
print(MORSE_TABLE['O'])   # '---'
```

---

### `tempo.dataset.tempo_dataset`

#### `TEMPODataset(file_path, max_time_steps=None)`

PyTorch `Dataset` that loads a TEMPO HDF5 file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str \| Path` | Path to a TEMPO HDF5 file |
| `max_time_steps` | `int \| None` | Fixed time dimension; auto-derived from data if `None` |

**Attributes after construction:**

| Attribute | Type | Shape / Description |
|-----------|------|---------------------|
| `spikes` | `torch.float32` | `[N, T, 2]` — dense binary spike tensor |
| `labels` | `torch.long` | `[N]` — integer class IDs |
| `word_to_id` | `dict[str, int]` | Word string → integer ID (sorted alphabetically) |
| `id_to_word` | `dict[int, str]` | Integer ID → word string |
| `num_classes` | `int` | Number of unique word classes in the dataset |

`__getitem__(idx)` returns `(spikes[idx], labels[idx])` with shapes `[T, 2]`
and `[]` (scalar), respectively.

---

### Protocol Constants (WPM = 20)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `T_u` | 60 ms | Base time unit |
| `T_thresh` | 115.2 ms | Channel assignment threshold (1.92 × T_u) |
| σ (jitter) | 34.5 ms | Gaussian jitter std dev (0.575 × T_u) |
| ω (weighting) | LogNormal(0.0360, 0.2446) | Per-word speed bias |
| r (dash_ratio) | LogNormal(1.2269, 0.2916) | Dash-to-dot duration ratio |
| Chance | 3.85% | 1/26 random baseline |
| Count ceiling | 26.9% | 7/26 rate-code only baseline |
