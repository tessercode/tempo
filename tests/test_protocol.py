"""
Regression tests for TEMPO protocol invariants.

These tests encode the properties that must hold regardless of future
implementation changes, ensuring protocol correctness is preserved.

Protocol reference (WPM=20, T_u=60 ms):
  T_thresh     = 1.92 × T_u = 115.2 ms   (channel assignment boundary)
  σ (jitter)   = 0.575 × T_u ≈ 34.5 ms
  ω (weight)   ~ LogNormal(0.0360, 0.2446)
  r (dash)     ~ LogNormal(1.2269, 0.2916)
  dot_ideal    = 1.0 × T_u    →  < T_thresh  → channel 0
  dash_ideal   = 3.0 × T_u    →  > T_thresh  → channel 1
"""

import numpy as np
import pytest

from tempo.dataset.generate_dataset import MORSE_TABLE, encode_word, split_channels

WPM = 20
T_U = 1200.0 / WPM       # 60.0 ms
T_THRESH = 1.92 * T_U    # 115.2 ms
SIGMA = 0.575 * T_U      # ~34.5 ms


# ---------------------------------------------------------------------------
# Threshold and channel boundary
# ---------------------------------------------------------------------------

class TestChannelBoundary:
    def test_dot_duration_below_threshold(self):
        """Standard dot (1.0 × T_u) must be below the channel threshold."""
        assert 1.0 * T_U < T_THRESH

    def test_dash_duration_above_threshold(self):
        """Standard dash (3.0 × T_u) must be above the channel threshold."""
        assert 3.0 * T_U > T_THRESH

    def test_dot_margin_at_least_half_t_u(self):
        """There should be meaningful separation between dot and T_thresh."""
        margin = T_THRESH - 1.0 * T_U
        assert margin >= 0.5 * T_U

    def test_dash_margin_at_least_half_t_u(self):
        """There should be meaningful separation between dash and T_thresh."""
        margin = 3.0 * T_U - T_THRESH
        assert margin >= 0.5 * T_U


# ---------------------------------------------------------------------------
# Deterministic encoding invariants
# ---------------------------------------------------------------------------

class TestDeterministicEncoding:
    def test_e_produces_exactly_one_dot_spike(self):
        spikes = encode_word('E', T_U)
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) == 1
        assert len(dash_t) == 0

    def test_t_produces_exactly_one_dash_spike(self):
        spikes = encode_word('T', T_U)
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) == 0
        assert len(dash_t) == 1

    def test_s_produces_three_dot_spikes(self):
        dot_t, dash_t = split_channels(encode_word('S', T_U))
        assert len(dot_t) == 3
        assert len(dash_t) == 0

    def test_o_produces_three_dash_spikes(self):
        dot_t, dash_t = split_channels(encode_word('O', T_U))
        assert len(dot_t) == 0
        assert len(dash_t) == 3

    def test_sos_spike_count(self):
        """SOS = '... --- ...' → 9 spikes: 6 dots, 3 dashes."""
        dot_t, dash_t = split_channels(encode_word('SOS', T_U))
        assert len(dot_t) == 6
        assert len(dash_t) == 3

    @pytest.mark.parametrize("char", list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    def test_dot_count_matches_morse_dots(self, char):
        n_dots_expected = MORSE_TABLE[char].count('.')
        dot_t, _ = split_channels(encode_word(char, T_U))
        assert len(dot_t) == n_dots_expected, (
            f"'{char}': expected {n_dots_expected} dot spikes, got {len(dot_t)}"
        )

    @pytest.mark.parametrize("char", list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    def test_dash_count_matches_morse_dashes(self, char):
        n_dashes_expected = MORSE_TABLE[char].count('-')
        _, dash_t = split_channels(encode_word(char, T_U))
        assert len(dash_t) == n_dashes_expected, (
            f"'{char}': expected {n_dashes_expected} dash spikes, got {len(dash_t)}"
        )


# ---------------------------------------------------------------------------
# Timing inter-character gap correctness
# ---------------------------------------------------------------------------

class TestInterCharacterGap:
    def test_se_fourth_spike_is_separated_by_inter_char_gap(self):
        """
        S = '...' ends with a spike at 5*T_u.
        Inter-character gap = 3*T_u → next character (E) starts at 5*T_u + 3*T_u = 8*T_u.
        E = '.' → spike at 8*T_u + 1*T_u = 9*T_u.
        """
        spikes = encode_word('SE', T_U)
        # S spikes at 1, 3, 5 × T_u; E spike at 9 × T_u
        times = [ts for ts, _ in spikes]
        assert abs(times[-1] - 9.0 * T_U) < 1e-9, (
            f"SE 4th spike: expected {9*T_U:.1f} ms, got {times[-1]:.6f} ms"
        )


# ---------------------------------------------------------------------------
# Noise parameter bounds
# ---------------------------------------------------------------------------

class TestNoiseBounds:
    N_TRIALS = 50

    def test_weighting_omega_strictly_positive(self):
        """E with weighting only: t_spike = omega*T_u, omega ~ LogNormal > 0.

        LogNormal is unbounded above, so the invariant is strict positivity
        rather than a fixed range.
        """
        rng = np.random.default_rng(0)
        for _ in range(self.N_TRIALS):
            ts = encode_word('E', T_U, weighting=True, rng=rng)[0][0]
            assert ts > 0

    def test_dash_ratio_t_strictly_positive(self):
        """T with dash_ratio only: t_spike = r*T_u, r ~ LogNormal > 0.

        LogNormal is unbounded above, so the invariant is strict positivity
        rather than a fixed range.
        """
        rng = np.random.default_rng(0)
        for _ in range(self.N_TRIALS):
            ts = encode_word('T', T_U, dash_ratio=True, rng=rng)[0][0]
            assert ts > 0

    def test_clamp_floor_never_violated_under_jitter(self):
        """No spike timestamp is ever below 0.1 * T_u, even under heavy jitter."""
        rng = np.random.default_rng(0)
        floor = 0.1 * T_U
        for _ in range(self.N_TRIALS):
            for ts, _ in encode_word('E', T_U, jitter=True, rng=rng):
                assert ts >= floor - 1e-9, (
                    f"Spike at {ts:.4f} ms violates clamp floor {floor:.1f} ms"
                )


# ---------------------------------------------------------------------------
# WPM scaling
# ---------------------------------------------------------------------------

class TestWPMScaling:
    def test_t_u_scales_inversely_with_wpm(self):
        t_u_10 = 1200.0 / 10
        t_u_20 = 1200.0 / 20
        t_u_40 = 1200.0 / 40
        assert t_u_10 == 120.0
        assert t_u_20 == 60.0
        assert t_u_40 == 30.0

    def test_spike_times_proportional_to_t_u(self):
        """E spike at WPM=10 should be exactly 2x the spike at WPM=20."""
        ts_20 = encode_word('E', 1200.0 / 20)[0][0]
        ts_10 = encode_word('E', 1200.0 / 10)[0][0]
        assert abs(ts_10 / ts_20 - 2.0) < 1e-9
