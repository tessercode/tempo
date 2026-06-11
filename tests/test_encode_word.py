"""
Regression tests for encode_word() and split_channels().

Covers:
  - Output structure (types, channels, positivity)
  - Spike counts matching Morse element counts
  - Exact deterministic timestamps for simple letters
  - Channel assignment relative to T_thresh
  - Clipping floor (0.1 * t_u)
  - Noise flags (weighting, dash_ratio, jitter)
  - RNG reproducibility
  - split_channels round-trip and edge cases
"""

import numpy as np
import pytest

from tempo.dataset.generate_dataset import MORSE_TABLE, encode_word, split_channels

WPM = 20
T_U = 1200.0 / WPM       # 60.0 ms
T_THRESH = 1.92 * T_U    # 115.2 ms

# LogNormal log-space parameters (must match generate_dataset.encode_word)
MU_OMEGA, SIGMA_OMEGA = 0.0360, 0.2446
MU_R, SIGMA_R = 1.2269, 0.2916


def _lognormal_mean(mu, sigma):
    """Analytic mean of LogNormal(mu, sigma): exp(mu + sigma^2 / 2)."""
    return np.exp(mu + 0.5 * sigma ** 2)


# ---------------------------------------------------------------------------
# encode_word — output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_returns_list(self):
        assert isinstance(encode_word('E', T_U), list)

    def test_each_element_is_length_two(self):
        for item in encode_word('HELLO', T_U):
            assert len(item) == 2

    def test_channels_are_binary(self):
        for _, ch in encode_word('MORSE', T_U):
            assert ch in (0, 1), f"Unexpected channel value: {ch}"

    def test_timestamps_are_positive(self):
        for ts, _ in encode_word('SOS', T_U):
            assert ts > 0, f"Non-positive timestamp: {ts}"

    def test_timestamps_strictly_increasing(self):
        spikes = encode_word('TEMPO', T_U)
        times = [ts for ts, _ in spikes]
        for a, b in zip(times, times[1:]):
            assert b > a, f"Timestamps not strictly increasing: {a:.3f} >= {b:.3f}"

    def test_empty_word_returns_empty_list(self):
        assert encode_word('', T_U) == []

    def test_unencodable_chars_are_silently_skipped(self):
        """Characters outside MORSE_TABLE (e.g. space) are filtered out."""
        spikes_ab = encode_word('AB', T_U)
        spikes_a_b = encode_word('A B', T_U)  # space is not encodable
        assert len(spikes_ab) == len(spikes_a_b)


# ---------------------------------------------------------------------------
# encode_word — spike counts
# ---------------------------------------------------------------------------

class TestSpikeCounts:
    @pytest.mark.parametrize("char", list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    def test_single_char_count_matches_morse_length(self, char):
        expected = len(MORSE_TABLE[char])
        got = len(encode_word(char, T_U))
        assert got == expected, (
            f"'{char}' ({MORSE_TABLE[char]}): expected {expected} spikes, got {got}"
        )

    def test_multi_char_word_total_count(self):
        word = 'SOS'
        expected = sum(len(MORSE_TABLE[c]) for c in word)
        assert len(encode_word(word, T_U)) == expected

    def test_long_word_count(self):
        word = 'TEMPO'
        expected = sum(len(MORSE_TABLE[c]) for c in word)
        assert len(encode_word(word, T_U)) == expected


# ---------------------------------------------------------------------------
# encode_word — deterministic timestamps
# ---------------------------------------------------------------------------
#
# At WPM=20 (T_U=60 ms), no noise, omega=1.0, r=3.0:
#
#   E = '.'   → dot at 1*T_U = 60 ms
#   T = '-'   → dash at 3*T_U = 180 ms
#
#   S = '...' timing:
#     t=0:   dot(60) → spike @  60ms; intra-gap(60) → t_cur=120
#     t=120: dot(60) → spike @ 180ms; intra-gap(60) → t_cur=240
#     t=240: dot(60) → spike @ 300ms; intra-gap(60) → t_cur=360
#     inter-char supersedes: t_cur = 300+180 = 480ms
#
#   A = '.-' timing:
#     t=0:   dot(60)  → spike @  60ms; intra-gap(60) → t_cur=120
#     t=120: dash(180)→ spike @ 300ms; intra-gap(60) → t_cur=360
#     inter-char supersedes: t_cur = 300+180 = 480ms
#
# ---------------------------------------------------------------------------

class TestDeterministicTimestamps:
    TOL = 1e-9

    def test_e_single_dot_at_one_t_u(self):
        spikes = encode_word('E', T_U)
        assert len(spikes) == 1
        assert abs(spikes[0][0] - T_U) < self.TOL

    def test_t_single_dash_at_three_t_u(self):
        spikes = encode_word('T', T_U)
        assert len(spikes) == 1
        assert abs(spikes[0][0] - 3.0 * T_U) < self.TOL

    def test_s_three_dots(self):
        spikes = encode_word('S', T_U)
        expected_times = [1 * T_U, 3 * T_U, 5 * T_U]
        assert len(spikes) == 3
        for i, (ts, _) in enumerate(spikes):
            assert abs(ts - expected_times[i]) < self.TOL, (
                f"S spike {i}: expected {expected_times[i]:.1f}, got {ts:.6f}"
            )

    def test_a_dot_then_dash(self):
        spikes = encode_word('A', T_U)
        assert len(spikes) == 2
        assert abs(spikes[0][0] - 1.0 * T_U) < self.TOL   # dot  @ 60ms
        assert abs(spikes[1][0] - 5.0 * T_U) < self.TOL   # dash @ 300ms

    def test_deterministic_is_identical_across_calls(self):
        s1 = encode_word('SOS', T_U)
        s2 = encode_word('SOS', T_U)
        assert s1 == s2


# ---------------------------------------------------------------------------
# encode_word — channel assignment
# ---------------------------------------------------------------------------

class TestChannelAssignment:
    def test_e_is_channel_0(self):
        spikes = encode_word('E', T_U)
        assert spikes[0][1] == 0

    def test_t_is_channel_1(self):
        spikes = encode_word('T', T_U)
        assert spikes[0][1] == 1

    def test_s_all_channel_0(self):
        spikes = encode_word('S', T_U)
        assert all(ch == 0 for _, ch in spikes)

    def test_o_all_channel_1(self):
        spikes = encode_word('O', T_U)
        assert all(ch == 1 for _, ch in spikes)

    def test_a_first_dot_second_dash(self):
        spikes = encode_word('A', T_U)
        assert spikes[0][1] == 0  # '.'
        assert spikes[1][1] == 1  # '-'

    def test_standard_dot_duration_below_threshold(self):
        """1.0 * T_u < T_thresh — dots always land on ch0 without noise."""
        assert 1.0 * T_U < T_THRESH

    def test_standard_dash_duration_above_threshold(self):
        """3.0 * T_u > T_thresh — dashes always land on ch1 without noise."""
        assert 3.0 * T_U > T_THRESH


# ---------------------------------------------------------------------------
# encode_word — timestamp clamping floor (0.1 * T_u)
# ---------------------------------------------------------------------------

class TestTimestampClamping:
    def test_no_spike_below_clamp_floor_with_extreme_jitter(self):
        """With jitter enabled, timestamps must stay >= 0.1 * T_u."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            spikes = encode_word('E', T_U, jitter=True, rng=rng)
            for ts, _ in spikes:
                # Allow a small float tolerance but no spike should be near 0
                assert ts >= 0.1 * T_U - 1e-6, (
                    f"Spike at {ts:.3f} ms is below clamp floor {0.1 * T_U:.1f} ms"
                )


# ---------------------------------------------------------------------------
# encode_word — noise flags
# ---------------------------------------------------------------------------

class TestNoiseFlags:
    def test_weighting_changes_timestamps(self):
        base = encode_word('MORSE', T_U)
        noisy = encode_word('MORSE', T_U, weighting=True, rng=np.random.default_rng(0))
        assert [ts for ts, _ in base] != [ts for ts, _ in noisy]

    def test_jitter_changes_timestamps(self):
        base = encode_word('MORSE', T_U)
        noisy = encode_word('MORSE', T_U, jitter=True, rng=np.random.default_rng(0))
        assert [ts for ts, _ in base] != [ts for ts, _ in noisy]

    def test_dash_ratio_changes_dash_timestamps(self):
        base = encode_word('T', T_U)                        # single dash
        noisy = encode_word('T', T_U, dash_ratio=True, rng=np.random.default_rng(0))
        assert base[0][0] != noisy[0][0]

    def test_weighting_e_spike_is_positive_lognormal(self):
        """E='.' with only weighting: t_spike = omega * T_u, omega ~ LogNormal.

        LogNormal is strictly positive and unbounded above, so we check
        positivity per draw and that the empirical mean of omega = t_spike / T_u
        matches the analytic LogNormal mean.
        """
        rng = np.random.default_rng(0)
        omegas = []
        for _ in range(5000):
            ts = encode_word('E', T_U, weighting=True, rng=rng)[0][0]
            assert ts > 0, f"Non-positive E spike under weighting: {ts}"
            omegas.append(ts / T_U)
        empirical = float(np.mean(omegas))
        expected = _lognormal_mean(MU_OMEGA, SIGMA_OMEGA)
        assert abs(empirical - expected) < 0.03, (
            f"omega mean {empirical:.4f} differs from LogNormal mean {expected:.4f}"
        )

    def test_dash_ratio_t_spike_is_positive_lognormal(self):
        """T='-' with only dash_ratio: t_spike = r * T_u, r ~ LogNormal.

        LogNormal is strictly positive and unbounded above, so we check
        positivity per draw and that the empirical mean of r = t_spike / T_u
        matches the analytic LogNormal mean.
        """
        rng = np.random.default_rng(0)
        ratios = []
        for _ in range(5000):
            ts = encode_word('T', T_U, dash_ratio=True, rng=rng)[0][0]
            assert ts > 0, f"Non-positive T spike under dash_ratio: {ts}"
            ratios.append(ts / T_U)
        empirical = float(np.mean(ratios))
        expected = _lognormal_mean(MU_R, SIGMA_R)
        assert abs(empirical - expected) < 0.1, (
            f"r mean {empirical:.4f} differs from LogNormal mean {expected:.4f}"
        )


# ---------------------------------------------------------------------------
# encode_word — RNG reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = encode_word('HELLO', T_U, weighting=True, dash_ratio=True, jitter=True, rng=rng1)
        s2 = encode_word('HELLO', T_U, weighting=True, dash_ratio=True, jitter=True, rng=rng2)
        assert s1 == s2

    def test_different_seeds_give_different_output(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        s1 = encode_word('HELLO', T_U, weighting=True, dash_ratio=True, jitter=True, rng=rng1)
        s2 = encode_word('HELLO', T_U, weighting=True, dash_ratio=True, jitter=True, rng=rng2)
        assert [ts for ts, _ in s1] != [ts for ts, _ in s2]


# ---------------------------------------------------------------------------
# split_channels
# ---------------------------------------------------------------------------

class TestSplitChannels:
    def test_empty_input_returns_two_empty_float64_arrays(self):
        dot_t, dash_t = split_channels([])
        assert isinstance(dot_t, np.ndarray)
        assert isinstance(dash_t, np.ndarray)
        assert dot_t.dtype == np.float64
        assert dash_t.dtype == np.float64
        assert len(dot_t) == 0
        assert len(dash_t) == 0

    def test_only_dots(self):
        spikes = [(60.0, 0), (180.0, 0), (300.0, 0)]
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) == 3
        assert len(dash_t) == 0
        np.testing.assert_array_equal(dot_t, [60.0, 180.0, 300.0])

    def test_only_dashes(self):
        spikes = [(180.0, 1), (420.0, 1)]
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) == 0
        assert len(dash_t) == 2
        np.testing.assert_array_equal(dash_t, [180.0, 420.0])

    def test_mixed_spikes(self):
        spikes = [(60.0, 0), (300.0, 1), (540.0, 0)]
        dot_t, dash_t = split_channels(spikes)
        np.testing.assert_array_equal(dot_t, [60.0, 540.0])
        np.testing.assert_array_equal(dash_t, [300.0])

    def test_returns_float64(self):
        dot_t, dash_t = split_channels([(60.0, 0), (180.0, 1)])
        assert dot_t.dtype == np.float64
        assert dash_t.dtype == np.float64

    def test_preserves_exact_timestamps(self):
        spikes = [(123.456789, 0), (987.654321, 1)]
        dot_t, dash_t = split_channels(spikes)
        assert abs(dot_t[0] - 123.456789) < 1e-9
        assert abs(dash_t[0] - 987.654321) < 1e-9

    def test_total_spike_count_preserved(self):
        spikes = encode_word('A', T_U)
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) + len(dash_t) == len(spikes)

    def test_e_roundtrip(self):
        """E='.' → 1 dot, 0 dashes."""
        spikes = encode_word('E', T_U)
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) == 1
        assert len(dash_t) == 0

    def test_t_roundtrip(self):
        """T='-' → 0 dots, 1 dash."""
        spikes = encode_word('T', T_U)
        dot_t, dash_t = split_channels(spikes)
        assert len(dot_t) == 0
        assert len(dash_t) == 1

    def test_s_o_roundtrip(self):
        """S='...' → 3 dots; O='---' → 3 dashes."""
        s_dot, s_dash = split_channels(encode_word('S', T_U))
        o_dot, o_dash = split_channels(encode_word('O', T_U))
        assert len(s_dot) == 3 and len(s_dash) == 0
        assert len(o_dot) == 0 and len(o_dash) == 3
