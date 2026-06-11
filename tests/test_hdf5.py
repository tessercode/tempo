"""
Regression tests for write_hdf5().

Covers:
  - File creation and HDF5 structure (groups, datasets, attributes)
  - Metadata attribute values match arguments
  - Spike data round-trips correctly (dot/dash timestamps preserved)
  - Labels round-trip correctly
  - Variable-length arrays preserved for ragged data
  - None seed stored as -1
"""

import h5py
import numpy as np
import pytest

from tempo.dataset.generate_dataset import generate_dataset, write_hdf5

WPM = 20
WORDS = ['A', 'E', 'S', 'T', 'O']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset_and_write(tmp_path, words=WORDS, multiplier=2, seed=0,
                            weighting=False, dash_ratio=False, jitter=False):
    dots, dashes, labels = generate_dataset(
        words, wpm=WPM, multiplier=multiplier,
        weighting=weighting, dash_ratio=dash_ratio, jitter=jitter, seed=seed,
    )
    path = tmp_path / "ds.h5"
    write_hdf5(
        str(path), dots, dashes, labels,
        wpm=WPM, multiplier=multiplier,
        weighting=weighting, dash_ratio=dash_ratio, jitter=jitter,
        seed=seed, wordlist_path="test_wordlist.txt",
    )
    return path, dots, dashes, labels


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestFileCreation:
    def test_file_is_created(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        assert path.exists()

    def test_file_is_valid_hdf5(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert f is not None   # opens without error


# ---------------------------------------------------------------------------
# HDF5 structure
# ---------------------------------------------------------------------------

class TestHDF5Structure:
    def test_spikes_group_exists(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert 'spikes' in f

    def test_dot_channel_dataset_exists(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert 'spikes/dot_channel' in f

    def test_dash_channel_dataset_exists(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert 'spikes/dash_channel' in f

    def test_labels_dataset_exists(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert 'labels' in f

    def test_metadata_group_exists(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert 'metadata' in f

    def test_dataset_lengths_match_sample_count(self, tmp_path):
        n_samples = len(WORDS) * 2
        path, *_ = make_dataset_and_write(tmp_path, multiplier=2)
        with h5py.File(str(path), 'r') as f:
            assert len(f['spikes/dot_channel']) == n_samples
            assert len(f['spikes/dash_channel']) == n_samples
            assert len(f['labels']) == n_samples


# ---------------------------------------------------------------------------
# Metadata attributes
# ---------------------------------------------------------------------------

class TestMetadataAttributes:
    REQUIRED_ATTRS = [
        'protocol_version', 'wpm', 'unit_ms', 'multiplier',
        'weighting_enabled', 'dash_ratio_enabled', 'jitter_enabled',
        'seed', 'wordlist_path', 'timescale', 'encoding_type', 'timestamp',
    ]

    def test_all_required_attributes_present(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            attrs = f['metadata'].attrs
            for key in self.REQUIRED_ATTRS:
                assert key in attrs, f"Missing metadata attribute: '{key}'"

    def test_protocol_version(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['protocol_version'] == 'TEMPO'

    def test_wpm_stored_correctly(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['wpm'] == WPM

    def test_unit_ms_equals_1200_over_wpm(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert abs(f['metadata'].attrs['unit_ms'] - 1200.0 / WPM) < 1e-9

    def test_multiplier_stored_correctly(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path, multiplier=7)
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['multiplier'] == 7

    def test_noise_flags_stored_correctly_all_false(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            attrs = f['metadata'].attrs
            assert not attrs['weighting_enabled']
            assert not attrs['dash_ratio_enabled']
            assert not attrs['jitter_enabled']

    def test_noise_flags_stored_correctly_all_true(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path,
                                           weighting=True, dash_ratio=True, jitter=True)
        with h5py.File(str(path), 'r') as f:
            attrs = f['metadata'].attrs
            assert attrs['weighting_enabled']
            assert attrs['dash_ratio_enabled']
            assert attrs['jitter_enabled']

    def test_seed_stored_correctly(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path, seed=99)
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['seed'] == 99

    def test_none_seed_stored_as_minus_one(self, tmp_path):
        dots, dashes, labels = generate_dataset(WORDS, wpm=WPM)
        path = tmp_path / "ds_noseed.h5"
        write_hdf5(
            str(path), dots, dashes, labels,
            wpm=WPM, multiplier=1,
            weighting=False, dash_ratio=False, jitter=False,
            seed=None, wordlist_path="test",
        )
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['seed'] == -1

    def test_encoding_type(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['encoding_type'] == 'Mark Termination (Dual Channel)'

    def test_timescale(self, tmp_path):
        path, *_ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            assert f['metadata'].attrs['timescale'] == '1ms'


# ---------------------------------------------------------------------------
# Data round-trip
# ---------------------------------------------------------------------------

class TestDataRoundTrip:
    def test_labels_round_trip(self, tmp_path):
        path, _, _, labels_written = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            raw = f['labels'][:]
            labels_read = [l.decode() if isinstance(l, bytes) else l for l in raw]
        assert labels_read == labels_written

    def test_dot_channel_round_trip(self, tmp_path):
        path, dots_written, _, _ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            for i, expected in enumerate(dots_written):
                read = f['spikes/dot_channel'][i]
                np.testing.assert_array_almost_equal(read, expected, decimal=9)

    def test_dash_channel_round_trip(self, tmp_path):
        path, _, dashes_written, _ = make_dataset_and_write(tmp_path)
        with h5py.File(str(path), 'r') as f:
            for i, expected in enumerate(dashes_written):
                read = f['spikes/dash_channel'][i]
                np.testing.assert_array_almost_equal(read, expected, decimal=9)

    def test_ragged_arrays_preserved(self, tmp_path):
        """E has 1 spike; O has 3 — different lengths must survive the round-trip."""
        words = ['E', 'O']
        dots, dashes, labels = generate_dataset(words, wpm=WPM, multiplier=1)
        path = tmp_path / "ragged.h5"
        write_hdf5(str(path), dots, dashes, labels,
                   wpm=WPM, multiplier=1,
                   weighting=False, dash_ratio=False, jitter=False,
                   seed=0, wordlist_path="test")
        with h5py.File(str(path), 'r') as f:
            e_dots = f['spikes/dot_channel'][0]   # E='.' → 1 dot
            o_dashes = f['spikes/dash_channel'][1] # O='---' → 3 dashes
            assert len(e_dots) == 1
            assert len(o_dashes) == 3
