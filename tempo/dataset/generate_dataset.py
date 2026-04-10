#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Luke Hindman
"""TEMPO v1.1 Dataset Generator

Generates dual-channel Morse code spike train datasets in HDF5 format
following the TEMPO v1.1 encoding protocol.

Usage:
    python -m tempo.dataset.generate_dataset wordlist.txt output.h5 [options]
"""

import argparse
import sys
import time

import h5py
import numpy as np

# International Morse Code: A-Z, 0-9
MORSE_TABLE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
    'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
    'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
    'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.',
}


def encode_word(word, t_u, weighting=False, dash_ratio=False, jitter=False, rng=None):
    """Encode a single word into a dual-channel spike train.

    Follows Algorithm 1 (TEMPO v1.1 Spike Train Generation) from the paper.

    Args:
        word: Uppercase string to encode.
        t_u: Base time unit in ms (1200 / WPM).
        weighting: Enable systematic bias (omega ~ U[0.8, 1.3]).
        dash_ratio: Enable dash ratio variation (r ~ U[2.5, 4.5]).
        jitter: Enable Gaussian jitter (sigma = 0.838 * t_u).
        rng: numpy RandomState for reproducibility.

    Returns:
        List of (timestamp_ms, channel) tuples where channel 0=dot, 1=dash.
    """
    if rng is None:
        rng = np.random.default_rng()

    t_thresh = 2.17 * t_u
    sigma = 0.838 * t_u if jitter else 0.0
    omega = rng.uniform(0.8, 1.3) if weighting else 1.0
    r = rng.uniform(2.5, 4.5) if dash_ratio else 3.0

    t_current = 0.0
    spikes = []
    chars = [c for c in word if c in MORSE_TABLE]

    for ci, char in enumerate(chars):
        pattern = MORSE_TABLE[char]

        for ei, element in enumerate(pattern):
            # Mark duration
            if element == '.':
                t_ideal = 1.0 * t_u
            else:
                t_ideal = r * t_u

            t_weighted = t_ideal * omega
            t_noisy = t_weighted + rng.normal(0, sigma) if sigma > 0 else t_weighted
            t_final = max(t_noisy, 0.1 * t_u)

            t_spike = t_current + t_final
            channel = 0 if t_final < t_thresh else 1
            spikes.append((t_spike, channel))

            # Intra-character gap (after every element)
            gap_ideal = 1.0 * t_u
            gap_noisy = gap_ideal * omega + (rng.normal(0, sigma) if sigma > 0 else 0.0)
            gap_final = max(gap_noisy, 0.1 * t_u)
            t_current = t_spike + gap_final

        # Inter-character gap (supersedes the last intra-character gap)
        # Per the paper note: final intra-char gap of each char is replaced
        # by the inter-character gap. We rewind the last intra-char advance
        # and apply inter-char instead.
        t_current = spikes[-1][0]  # back to last spike time
        gap_ideal = 3.0 * t_u
        gap_noisy = gap_ideal * omega + (rng.normal(0, sigma) if sigma > 0 else 0.0)
        gap_final = max(gap_noisy, 0.1 * t_u)
        t_current += gap_final

    # Inter-word gap at end (supersedes last inter-character gap)
    # Rewind to last spike and apply inter-word gap
    if spikes:
        t_current = spikes[-1][0]
        gap_ideal = 7.0 * t_u
        gap_noisy = gap_ideal * omega + (rng.normal(0, sigma) if sigma > 0 else 0.0)
        gap_final = max(gap_noisy, 0.1 * t_u)
        t_current += gap_final

    return spikes


def split_channels(spikes):
    """Split a spike list into per-channel timestamp arrays.

    Timestamps are relative to t=0 at the start of the first Morse symbol,
    as produced by encode_word().

    Args:
        spikes: List of (timestamp_ms, channel) tuples.

    Returns:
        (dot_times, dash_times) as numpy float64 arrays.
    """
    if not spikes:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    dot_times = np.array([t for t, ch in spikes if ch == 0], dtype=np.float64)
    dash_times = np.array([t for t, ch in spikes if ch == 1], dtype=np.float64)
    return dot_times, dash_times


def generate_dataset(words, wpm, multiplier=1, weighting=False, dash_ratio=False, jitter=False, seed=None):
    """Generate the full dataset of spike trains.

    Args:
        words: List of uppercase word strings.
        wpm: Words per minute.
        multiplier: Number of spike trains per word.
        weighting: Enable systematic bias.
        dash_ratio: Enable dash ratio variation.
        jitter: Enable Gaussian jitter.
        seed: Optional RNG seed.

    Returns:
        all_dot_times: list of numpy arrays (one per sample).
        all_dash_times: list of numpy arrays (one per sample).
        labels: list of word strings.
    """
    rng = np.random.default_rng(seed)
    t_u = 1200.0 / wpm

    all_dot_times = []
    all_dash_times = []
    labels = []
    n_total = len(words) * multiplier

    for i, word in enumerate(words):
        for j in range(multiplier):
            spike_list = encode_word(word, t_u, weighting, dash_ratio, jitter, rng)
            dot_times, dash_times = split_channels(spike_list)
            all_dot_times.append(dot_times)
            all_dash_times.append(dash_times)
            labels.append(word)

            done = i * multiplier + j + 1
            print(f"\r  Encoding: {done}/{n_total} samples", end="", file=sys.stderr)

    print(file=sys.stderr)
    return all_dot_times, all_dash_times, labels


def write_hdf5(filepath, all_dot_times, all_dash_times, labels, wpm, multiplier,
               weighting, dash_ratio, jitter, seed, wordlist_path):
    """Write the dataset to an HDF5 file.

    Spike data is organized as spikes/<channel>[sample_index] rather than
    spikes/<sample>/<channel> because each sample has a variable number of
    spikes per channel. HDF5 variable-length datasets (vlen_dtype) require
    a flat 1D dataset where each element is a ragged array, so the channel
    split must occur at the dataset level. The two channel datasets act as
    parallel arrays indexed by sample. This avoids creating N per-sample
    groups, which is slow to create and traverse in HDF5.

    Args:
        filepath: Output file path.
        all_dot_times: List of numpy arrays (one per sample) with dot spike times.
        all_dash_times: List of numpy arrays (one per sample) with dash spike times.
        labels: List of word strings.
        wpm: Words per minute used.
        multiplier: Samples per word.
        weighting: Whether weighting noise was enabled.
        dash_ratio: Whether dash ratio noise was enabled.
        jitter: Whether jitter noise was enabled.
        seed: RNG seed used (or None).
        wordlist_path: Path to the source wordlist.
    """
    str_dt = h5py.string_dtype(encoding='utf-8')
    vlen_dt = h5py.vlen_dtype(np.float64)

    with h5py.File(filepath, 'w') as f:
        spikes = f.create_group('spikes')
        dot_ds = spikes.create_dataset('dot_channel', shape=(len(labels),), dtype=vlen_dt)
        dash_ds = spikes.create_dataset('dash_channel', shape=(len(labels),), dtype=vlen_dt)
        for i in range(len(labels)):
            dot_ds[i] = all_dot_times[i]
            dash_ds[i] = all_dash_times[i]

        f.create_dataset('labels', data=np.array(labels, dtype=object), dtype=str_dt)

        meta = f.create_group('metadata')
        meta.attrs['protocol_version'] = 'TEMPO v1.1'
        meta.attrs['wpm'] = wpm
        meta.attrs['unit_ms'] = 1200.0 / wpm
        meta.attrs['multiplier'] = multiplier
        meta.attrs['weighting_enabled'] = weighting
        meta.attrs['dash_ratio_enabled'] = dash_ratio
        meta.attrs['jitter_enabled'] = jitter
        meta.attrs['seed'] = seed if seed is not None else -1
        meta.attrs['wordlist_path'] = str(wordlist_path)
        meta.attrs['timescale'] = '1ms'
        meta.attrs['encoding_type'] = 'Mark Termination (Dual Channel)'
        meta.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(
        description='TEMPO v1.1 Dataset Generator — generate dual-channel '
                    'Morse spike train datasets in HDF5 format.',
    )
    parser.add_argument('wordlist', help='Path to wordlist file (whitespace-separated words)')
    parser.add_argument('output', help='Output HDF5 file path')
    parser.add_argument('--wpm', type=int, default=20,
                        help='Words per minute (default: 20)')
    parser.add_argument('--multiplier', type=int, default=1,
                        help='Number of spike trains per word (default: 1)')
    parser.add_argument('--weighting', action='store_true',
                        help='Enable systematic bias (omega ~ U[0.8, 1.3])')
    parser.add_argument('--dash-ratio', action='store_true',
                        help='Enable dash ratio variation (r ~ U[2.5, 4.5])')
    parser.add_argument('--jitter', action='store_true',
                        help='Enable Gaussian jitter (sigma = 0.838 * T_u)')
    parser.add_argument('--all-noise', action='store_true',
                        help='Enable all noise sources')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.all_noise:
        args.weighting = True
        args.dash_ratio = True
        args.jitter = True

    # Read wordlist
    with open(args.wordlist) as f:
        raw = f.read().split()

    # Normalize to uppercase, filter to encodable characters, deduplicate
    words = []
    seen = set()
    skipped = []
    for w in raw:
        upper = w.upper()
        # Check all characters are in the Morse table
        if all(c in MORSE_TABLE for c in upper):
            if upper not in seen:
                words.append(upper)
                seen.add(upper)
        else:
            bad = [c for c in upper if c not in MORSE_TABLE]
            skipped.append((w, bad))

    if skipped:
        print(f"Warning: skipped {len(skipped)} word(s) with " f"unencodable characters:", file=sys.stderr)
        for w, bad in skipped[:10]:
            print(f"  '{w}' (chars: {bad})", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)

    if not words:
        print("Error: no encodable words found in wordlist.", file=sys.stderr)
        sys.exit(1)

    t_u = 1200.0 / args.wpm
    n_total = len(words) * args.multiplier

    print(f"TEMPO v1.1 Dataset Generator", file=sys.stderr)
    print(f"  Words: {len(words)} unique", file=sys.stderr)
    print(f"  WPM: {args.wpm} (T_u = {t_u:.1f} ms)", file=sys.stderr)
    print(f"  Multiplier: {args.multiplier} ({n_total} total samples)", file=sys.stderr)
    print(f"  Noise: weighting={args.weighting}, dash_ratio={args.dash_ratio}, " f"jitter={args.jitter}", file=sys.stderr)
    if args.seed is not None:
        print(f"  Seed: {args.seed}", file=sys.stderr)

    all_dot_times, all_dash_times, labels = generate_dataset(
        words, args.wpm, args.multiplier,
        args.weighting, args.dash_ratio, args.jitter, args.seed,
    )

    print(f"  Writing {args.output}...", file=sys.stderr)
    write_hdf5(
        args.output, all_dot_times, all_dash_times, labels, args.wpm,
        args.multiplier, args.weighting, args.dash_ratio, args.jitter,
        args.seed, args.wordlist,
    )

    print(f"  Done. {len(labels)} samples written.", file=sys.stderr)


if __name__ == '__main__':
    main()
