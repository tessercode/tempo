"""
Shared pytest fixtures for the TEMPO regression test suite.
"""

import pytest
import numpy as np

from tempo.dataset.generate_dataset import generate_dataset, write_hdf5

# ---------------------------------------------------------------------------
# Protocol constants used across multiple test modules
# ---------------------------------------------------------------------------
WPM = 20
T_U = 1200.0 / WPM          # 60.0 ms
T_THRESH = 1.92 * T_U       # 115.2 ms — channel assignment boundary
SIGMA = 0.575 * T_U         # ~34.5 ms — Gaussian jitter std dev

SIMPLE_WORDS = ['A', 'E', 'S', 'T', 'O']


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def t_u():
    return T_U


@pytest.fixture
def simple_words():
    return SIMPLE_WORDS


@pytest.fixture
def small_dataset():
    """Deterministic dataset: SIMPLE_WORDS × 3 samples each, no noise."""
    return generate_dataset(SIMPLE_WORDS, wpm=WPM, multiplier=3, seed=0)


@pytest.fixture
def stochastic_dataset():
    """Full-noise dataset: SIMPLE_WORDS × 3 samples each."""
    return generate_dataset(
        SIMPLE_WORDS, wpm=WPM, multiplier=3,
        weighting=True, dash_ratio=True, jitter=True, seed=42,
    )


@pytest.fixture
def hdf5_path(tmp_path, small_dataset):
    """Write small_dataset to a temp HDF5 file and return the path."""
    all_dot, all_dash, labels = small_dataset
    filepath = tmp_path / "test_dataset.h5"
    write_hdf5(
        str(filepath), all_dot, all_dash, labels,
        wpm=WPM, multiplier=3,
        weighting=False, dash_ratio=False, jitter=False,
        seed=0, wordlist_path="test",
    )
    return filepath


@pytest.fixture
def stochastic_hdf5_path(tmp_path, stochastic_dataset):
    """Write stochastic_dataset to a temp HDF5 file and return the path."""
    all_dot, all_dash, labels = stochastic_dataset
    filepath = tmp_path / "test_stochastic.h5"
    write_hdf5(
        str(filepath), all_dot, all_dash, labels,
        wpm=WPM, multiplier=3,
        weighting=True, dash_ratio=True, jitter=True,
        seed=42, wordlist_path="test",
    )
    return filepath
