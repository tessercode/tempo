"""
Regression tests for generate_dataset().

Covers:
  - Sample count (words × multiplier)
  - Label ordering and correctness
  - Return types
  - RNG reproducibility via seed
  - Noise flag propagation
"""

import numpy as np
import pytest

from tempo.dataset.generate_dataset import generate_dataset

WPM = 20
WORDS = ['A', 'E', 'S', 'T', 'O']


# ---------------------------------------------------------------------------
# Sample count and label structure
# ---------------------------------------------------------------------------

class TestSampleCount:
    def test_default_multiplier_one_sample_per_word(self):
        dots, dashes, labels = generate_dataset(WORDS, wpm=WPM)
        assert len(dots) == len(WORDS)
        assert len(dashes) == len(WORDS)
        assert len(labels) == len(WORDS)

    def test_multiplier_scales_sample_count(self):
        M = 7
        dots, dashes, labels = generate_dataset(WORDS, wpm=WPM, multiplier=M)
        assert len(dots) == len(WORDS) * M
        assert len(dashes) == len(WORDS) * M
        assert len(labels) == len(WORDS) * M

    def test_single_word_single_multiplier(self):
        dots, dashes, labels = generate_dataset(['S'], wpm=WPM, multiplier=1)
        assert len(labels) == 1

    def test_single_word_multiple_multiplier(self):
        dots, dashes, labels = generate_dataset(['S'], wpm=WPM, multiplier=5)
        assert len(labels) == 5
        assert all(l == 'S' for l in labels)


class TestLabelCorrectness:
    def test_labels_match_words_order(self):
        words = ['S', 'O', 'S']
        _, _, labels = generate_dataset(words, wpm=WPM, multiplier=1)
        assert labels == words

    def test_multiplier_labels_are_grouped_by_word(self):
        words = ['A', 'B']
        _, _, labels = generate_dataset(words, wpm=WPM, multiplier=3)
        assert labels == ['A', 'A', 'A', 'B', 'B', 'B']

    def test_all_labels_are_strings(self):
        _, _, labels = generate_dataset(WORDS, wpm=WPM, multiplier=2)
        for lbl in labels:
            assert isinstance(lbl, str)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_returns_three_items(self):
        result = generate_dataset(['E'], wpm=WPM)
        assert len(result) == 3

    def test_dot_and_dash_are_lists(self):
        dots, dashes, _ = generate_dataset(['E'], wpm=WPM)
        assert isinstance(dots, list)
        assert isinstance(dashes, list)

    def test_each_element_is_numpy_array(self):
        dots, dashes, _ = generate_dataset(WORDS, wpm=WPM)
        for arr in dots + dashes:
            assert isinstance(arr, np.ndarray)

    def test_spike_array_dtype_float64(self):
        dots, dashes, _ = generate_dataset(WORDS, wpm=WPM)
        for arr in dots + dashes:
            assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# WPM scaling
# ---------------------------------------------------------------------------

class TestWPMScaling:
    def test_higher_wpm_produces_shorter_spike_times(self):
        """At higher WPM, T_u is smaller, so spike times should be smaller."""
        dots_slow, _, _ = generate_dataset(['E'], wpm=10, multiplier=1, seed=0)
        dots_fast, _, _ = generate_dataset(['E'], wpm=40, multiplier=1, seed=0)
        # E has one dot spike; fast WPM → smaller timestamp
        assert dots_fast[0][0] < dots_slow[0][0]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_deterministic_output(self):
        d1, dsh1, l1 = generate_dataset(WORDS, wpm=WPM, jitter=True,
                                         multiplier=3, seed=42)
        d2, dsh2, l2 = generate_dataset(WORDS, wpm=WPM, jitter=True,
                                         multiplier=3, seed=42)
        assert l1 == l2
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(dsh1, dsh2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_spikes(self):
        words = ['MORSE', 'CODE']
        d1, _, _ = generate_dataset(words, wpm=WPM, jitter=True,
                                     multiplier=5, seed=1)
        d2, _, _ = generate_dataset(words, wpm=WPM, jitter=True,
                                     multiplier=5, seed=2)
        assert any(not np.array_equal(a, b) for a, b in zip(d1, d2))

    def test_no_noise_is_deterministic_without_seed(self):
        """Deterministic encoding (no noise) needs no seed to be reproducible."""
        d1, _, _ = generate_dataset(WORDS, wpm=WPM)
        d2, _, _ = generate_dataset(WORDS, wpm=WPM)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Noise flag propagation
# ---------------------------------------------------------------------------

class TestNoisePropagation:
    def test_jitter_produces_variability_across_multiplier(self):
        """With jitter and multiplier>1, copies of the same word should differ."""
        words = ['S']
        dots, _, _ = generate_dataset(words, wpm=WPM, multiplier=10,
                                       jitter=True, seed=0)
        # At least some pair of copies should differ
        all_same = all(np.array_equal(dots[0], dots[i]) for i in range(1, 10))
        assert not all_same

    def test_no_noise_all_copies_identical(self):
        """Without noise, every copy of the same word must be identical."""
        words = ['S']
        dots, _, _ = generate_dataset(words, wpm=WPM, multiplier=5, seed=0)
        for i in range(1, 5):
            np.testing.assert_array_equal(dots[0], dots[i])

    def test_weighting_produces_variability(self):
        words = ['MORSE']
        dots_det, _, _ = generate_dataset(words, wpm=WPM)
        dots_w, _, _ = generate_dataset(words, wpm=WPM, weighting=True, seed=7)
        assert not np.array_equal(dots_det[0], dots_w[0])
