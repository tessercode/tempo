# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Luke Hindman
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TEMPODataset(Dataset):
    """
    Custom Loader for TEMPO v1.1 HDF5 files.
    Assumes HDF5 structure:
    - 'spikes/dot_channel': (N_samples,) variable-length float64 timestamps
    - 'spikes/dash_channel': (N_samples,) variable-length float64 timestamps
    - 'labels': (N_samples,) variable-length UTF-8 strings

    Reads sparse per-channel spike timestamps and converts them to dense
    binary tensors [Time, 2] for SNNTorch compatibility.

    Builds a deterministic word-to-integer mapping so that all spike trains
    for the same word share the same integer label. Words are assigned IDs
    in sorted order for reproducibility.

    Args:
        file_path: Path to a TEMPO v1.1 HDF5 file.
        max_time_steps: Fixed time dimension length. If None, derived from
            the latest spike across all samples. Samples are zero-padded
            to this length; spikes beyond it are discarded.
    """
    def __init__(self, file_path, max_time_steps=None):
        with h5py.File(file_path, 'r') as f:
            raw_dots = [f['spikes/dot_channel'][i] for i in range(len(f['labels']))]
            raw_dashes = [f['spikes/dash_channel'][i] for i in range(len(f['labels']))]
            raw_labels = f['labels'][:]

        # Decode byte strings to Python str
        str_labels = [l.decode() if isinstance(l, bytes) else l for l in raw_labels]

        # Build deterministic word -> integer mapping (sorted for reproducibility)
        unique_words = sorted(set(str_labels))
        self.word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.num_classes = len(unique_words)

        # Convert string labels to integer IDs
        self.labels = torch.tensor(
            [self.word_to_id[w] for w in str_labels], dtype=torch.long,
        )

        # Determine time dimension
        if max_time_steps is None:
            max_t = 0
            for dots, dashes in zip(raw_dots, raw_dashes):
                if len(dots) > 0:
                    max_t = max(max_t, int(round(dots.max())))
                if len(dashes) > 0:
                    max_t = max(max_t, int(round(dashes.max())))
            max_time_steps = max_t + 1  # +1 for zero-indexing

        # Convert sparse timestamps to dense binary tensors
        n_samples = len(str_labels)
        spikes = np.zeros((n_samples, max_time_steps, 2), dtype=np.float32)
        for i, (dots, dashes) in enumerate(zip(raw_dots, raw_dashes)):
            for t in dots:
                idx = int(round(t))
                if 0 <= idx < max_time_steps:
                    spikes[i, idx, 0] = 1.0
            for t in dashes:
                idx = int(round(t))
                if 0 <= idx < max_time_steps:
                    spikes[i, idx, 1] = 1.0

        self.spikes = torch.tensor(spikes, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns spike tensor [Time, Channels] and integer label
        return self.spikes[idx], self.labels[idx]


# Example Usage:
# ds = TEMPODataset("tempo_dataset.h5")
# print(f"Classes: {ds.num_classes}, Mapping: {ds.word_to_id}")
# loader = DataLoader(ds, batch_size=64, shuffle=True)
#
# for spikes, targets in loader:
#     # spikes shape: [Batch, Time, Channels]
#     # SNNTorch expects [Time, Batch, Channels]
#     spikes = spikes.transpose(0, 1)
#     spk_out, mem_out = net(spikes)
