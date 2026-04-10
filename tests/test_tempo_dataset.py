"""
Regression tests for TEMPODataset.

Covers:
  - Correct loading from HDF5
  - Tensor shapes and dtypes
  - Label mapping (word_to_id / id_to_word) determinism and invertibility
  - Spike tensor is binary (0.0 / 1.0)
  - max_time_steps auto-derivation and explicit override
  - __getitem__ shapes
  - DataLoader batch shapes and permute convention
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tempo.dataset.tempo_dataset import TEMPODataset

WPM = 20
WORDS = ['A', 'E', 'S', 'T', 'O']  # 5 words
MULTIPLIER = 3                       # 15 samples total


# ---------------------------------------------------------------------------
# Loading and basic attributes
# ---------------------------------------------------------------------------

class TestLoading:
    def test_loads_without_error(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds is not None

    def test_len_equals_total_samples(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert len(ds) == len(WORDS) * MULTIPLIER

    def test_num_classes_equals_unique_words(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds.num_classes == len(WORDS)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class TestLabelMapping:
    def test_word_to_id_has_all_classes(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        for word in WORDS:
            assert word in ds.word_to_id

    def test_id_to_word_is_inverse_of_word_to_id(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        for word, idx in ds.word_to_id.items():
            assert ds.id_to_word[idx] == word

    def test_mapping_is_sorted_alphabetically(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        sorted_words = sorted(WORDS)
        for i, word in enumerate(sorted_words):
            assert ds.word_to_id[word] == i, (
                f"'{word}' should map to {i}, got {ds.word_to_id[word]}"
            )

    def test_mapping_is_deterministic_across_loads(self, hdf5_path):
        ds1 = TEMPODataset(str(hdf5_path))
        ds2 = TEMPODataset(str(hdf5_path))
        assert ds1.word_to_id == ds2.word_to_id

    def test_label_ids_are_valid_class_indices(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        n = ds.num_classes
        for label in ds.labels:
            assert 0 <= label.item() < n


# ---------------------------------------------------------------------------
# Tensor shapes and dtypes
# ---------------------------------------------------------------------------

class TestTensorShapes:
    def test_spikes_tensor_is_3d(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds.spikes.ndim == 3

    def test_spikes_tensor_shape_N_T_2(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        N, T, C = ds.spikes.shape
        assert N == len(WORDS) * MULTIPLIER
        assert C == 2

    def test_spikes_tensor_dtype_float32(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds.spikes.dtype == torch.float32

    def test_labels_tensor_is_1d(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds.labels.ndim == 1

    def test_labels_tensor_length(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert len(ds.labels) == len(WORDS) * MULTIPLIER

    def test_labels_tensor_dtype_long(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds.labels.dtype == torch.long


# ---------------------------------------------------------------------------
# Spike values are binary
# ---------------------------------------------------------------------------

class TestSpikeBinary:
    def test_spike_values_are_zero_or_one(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        unique_vals = ds.spikes.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Unexpected spike value: {v.item()}"

    def test_at_least_some_spikes_are_nonzero(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        assert ds.spikes.sum().item() > 0


# ---------------------------------------------------------------------------
# max_time_steps behaviour
# ---------------------------------------------------------------------------

class TestMaxTimeSteps:
    def test_auto_derived_time_covers_all_spikes(self, hdf5_path):
        """Auto-derived T must be >= largest spike timestamp in any sample."""
        import h5py
        ds = TEMPODataset(str(hdf5_path))
        T = ds.spikes.shape[1]
        with h5py.File(str(hdf5_path), 'r') as f:
            for i in range(len(ds)):
                dots = f['spikes/dot_channel'][i]
                dashes = f['spikes/dash_channel'][i]
                for t in dots:
                    assert int(round(t)) < T
                for t in dashes:
                    assert int(round(t)) < T

    def test_explicit_max_time_steps_sets_tensor_size(self, hdf5_path):
        T_FIXED = 2000
        ds = TEMPODataset(str(hdf5_path), max_time_steps=T_FIXED)
        assert ds.spikes.shape[1] == T_FIXED

    def test_explicit_max_time_steps_smaller_truncates(self, hdf5_path):
        """Spikes beyond max_time_steps are silently discarded; no error."""
        ds = TEMPODataset(str(hdf5_path), max_time_steps=10)
        assert ds.spikes.shape[1] == 10   # shape is respected

    def test_two_datasets_same_max_time_steps_are_compatible(self, hdf5_path,
                                                               stochastic_hdf5_path):
        T = 1600
        ds1 = TEMPODataset(str(hdf5_path), max_time_steps=T)
        ds2 = TEMPODataset(str(stochastic_hdf5_path), max_time_steps=T)
        assert ds1.spikes.shape[1] == ds2.spikes.shape[1] == T


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

class TestGetItem:
    def test_returns_tuple_of_two(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        item = ds[0]
        assert len(item) == 2

    def test_spikes_item_shape_is_T_2(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        spikes, label = ds[0]
        T = ds.spikes.shape[1]
        assert spikes.shape == (T, 2)

    def test_label_item_is_scalar_long(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        _, label = ds[0]
        assert label.dtype == torch.long
        assert label.ndim == 0  # scalar tensor

    def test_getitem_consistent_with_direct_access(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        spikes, label = ds[3]
        assert torch.equal(spikes, ds.spikes[3])
        assert torch.equal(label, ds.labels[3])


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------

class TestDataLoaderCompatibility:
    def test_dataloader_batch_shape(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        T = ds.spikes.shape[1]
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_spikes, batch_labels = next(iter(loader))
        assert batch_spikes.shape == (4, T, 2)
        assert batch_labels.shape == (4,)

    def test_permute_to_snntorch_convention(self, hdf5_path):
        """After permute(1,0,2), shape is [T, B, 2] as snnTorch expects."""
        ds = TEMPODataset(str(hdf5_path))
        T = ds.spikes.shape[1]
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        spikes, _ = next(iter(loader))
        spikes_t = spikes.permute(1, 0, 2)
        assert spikes_t.shape == (T, 4, 2)

    def test_full_epoch_without_error(self, hdf5_path):
        ds = TEMPODataset(str(hdf5_path))
        loader = DataLoader(ds, batch_size=5, shuffle=True)
        count = 0
        for spikes, labels in loader:
            count += len(labels)
        assert count == len(ds)
