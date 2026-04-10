"""
Regression tests for the generate_dataset CLI (python -m tempo.dataset.generate_dataset).

Covers:
  - Basic invocation creates a valid, non-empty HDF5 file
  - --wpm, --multiplier flags are respected
  - --all-noise flag enables all three noise sources in metadata
  - --seed makes output reproducible (spike data identical across runs)
  - Unencodable words are skipped without crashing (warning to stderr)
  - An empty/all-invalid wordlist exits with a non-zero return code
"""

import subprocess
import sys

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cli(args, wordlist_content, tmp_path):
    """Write a temp wordlist file, run the CLI, return (returncode, stdout, stderr, h5_path)."""
    wl_path = tmp_path / "wordlist.txt"
    h5_path = tmp_path / "output.h5"
    wl_path.write_text(wordlist_content)
    result = subprocess.run(
        [sys.executable, '-m', 'tempo.dataset.generate_dataset',
         str(wl_path), str(h5_path)] + args,
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout, result.stderr, h5_path


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------

class TestBasicInvocation:
    def test_exit_code_zero_on_success(self, tmp_path):
        rc, _, _, _ = run_cli([], 'A B C', tmp_path)
        assert rc == 0

    def test_output_file_is_created(self, tmp_path):
        _, _, _, h5_path = run_cli([], 'A B C', tmp_path)
        assert h5_path.exists()

    def test_output_is_valid_hdf5(self, tmp_path):
        _, _, _, h5_path = run_cli([], 'A B C', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            assert 'spikes' in f
            assert 'labels' in f
            assert 'metadata' in f

    def test_lowercase_words_are_normalised(self, tmp_path):
        """CLI should uppercase input words before encoding."""
        _, _, _, h5_path = run_cli([], 'a b c', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            labels = [l.decode() if isinstance(l, bytes) else l for l in f['labels'][:]]
        assert all(l.isupper() for l in labels)


# ---------------------------------------------------------------------------
# --multiplier flag
# ---------------------------------------------------------------------------

class TestMultiplierFlag:
    def test_default_multiplier_one_sample_per_word(self, tmp_path):
        _, _, _, h5_path = run_cli([], 'A B C', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            assert len(f['labels']) == 3

    def test_explicit_multiplier(self, tmp_path):
        _, _, _, h5_path = run_cli(['--multiplier', '5'], 'A B C', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            assert len(f['labels']) == 15   # 3 words × 5


# ---------------------------------------------------------------------------
# --wpm flag
# ---------------------------------------------------------------------------

class TestWPMFlag:
    def test_wpm_stored_in_metadata(self, tmp_path):
        _, _, _, h5_path = run_cli(['--wpm', '15'], 'E T', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            assert f['metadata'].attrs['wpm'] == 15

    def test_unit_ms_matches_wpm(self, tmp_path):
        _, _, _, h5_path = run_cli(['--wpm', '15'], 'E T', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            expected_unit = 1200.0 / 15
            assert abs(f['metadata'].attrs['unit_ms'] - expected_unit) < 1e-9


# ---------------------------------------------------------------------------
# --all-noise flag
# ---------------------------------------------------------------------------

class TestAllNoiseFlag:
    def test_all_noise_sets_metadata_flags(self, tmp_path):
        _, _, _, h5_path = run_cli(['--all-noise', '--seed', '0'], 'A B C', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            attrs = f['metadata'].attrs
            assert attrs['weighting_enabled']
            assert attrs['dash_ratio_enabled']
            assert attrs['jitter_enabled']

    def test_all_noise_produces_nonzero_variance(self, tmp_path):
        """With --all-noise and multiplier>1, copies of the same word should differ."""
        _, _, _, h5_path = run_cli(
            ['--all-noise', '--multiplier', '5', '--seed', '42'], 'S', tmp_path
        )
        with h5py.File(str(h5_path), 'r') as f:
            dots = [f['spikes/dot_channel'][i] for i in range(5)]
        all_same = all(np.array_equal(dots[0], dots[i]) for i in range(1, 5))
        assert not all_same


# ---------------------------------------------------------------------------
# --seed flag (reproducibility)
# ---------------------------------------------------------------------------

class TestSeedFlag:
    def test_same_seed_same_dot_spikes(self, tmp_path):
        args = ['--all-noise', '--multiplier', '3', '--seed', '7']
        words = 'A E S T O'
        _, _, _, h5a = run_cli(args, words, tmp_path / "a" if False else tmp_path)

        tmp2 = tmp_path / "run2"
        tmp2.mkdir()
        _, _, _, h5b = run_cli(args, words, tmp2)

        with h5py.File(str(h5a), 'r') as fa, h5py.File(str(h5b), 'r') as fb:
            n = len(fa['labels'])
            for i in range(n):
                np.testing.assert_array_almost_equal(
                    fa['spikes/dot_channel'][i],
                    fb['spikes/dot_channel'][i],
                    decimal=9,
                    err_msg=f"dot_channel[{i}] differs between seeded runs",
                )

    def test_seed_stored_in_metadata(self, tmp_path):
        _, _, _, h5_path = run_cli(['--seed', '123'], 'A B', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            assert f['metadata'].attrs['seed'] == 123


# ---------------------------------------------------------------------------
# Invalid / unencodable words
# ---------------------------------------------------------------------------

class TestInvalidWords:
    def test_unencodable_word_is_skipped_not_crash(self, tmp_path):
        """A word containing '?' (not in Morse table) should be skipped."""
        rc, _, stderr, h5_path = run_cli([], 'A HELLO? B', tmp_path)
        assert rc == 0
        assert h5_path.exists()

    def test_unencodable_word_warning_in_stderr(self, tmp_path):
        rc, _, stderr, _ = run_cli([], 'A HELLO? B', tmp_path)
        assert 'skipped' in stderr.lower() or 'warning' in stderr.lower()

    def test_only_valid_words_are_encoded(self, tmp_path):
        """After skipping 'HELLO?', only A and B should appear in labels."""
        _, _, _, h5_path = run_cli([], 'A HELLO? B', tmp_path)
        with h5py.File(str(h5_path), 'r') as f:
            labels = {l.decode() if isinstance(l, bytes) else l
                      for l in f['labels'][:]}
        assert 'A' in labels
        assert 'B' in labels
        assert not any('?' in l for l in labels)

    def test_all_invalid_words_exits_nonzero(self, tmp_path):
        """A wordlist with no encodable words should exit with a non-zero code."""
        rc, _, _, _ = run_cli([], '??? !!! @@@', tmp_path)
        assert rc != 0
