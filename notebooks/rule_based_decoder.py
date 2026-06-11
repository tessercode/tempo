#!/usr/bin/env python3
"""
rule_based_decoder.py  —  Rule-based Morse decoder baseline for TEMPO (R1.3 / R2.1)

Implements a channel-reading decoder that reconstructs the dot/dash sequence
directly from the TEMPO spike-train channel assignments (DOT_CH / DASH_CH)
and performs a reverse Morse table lookup.  No learning is used.

The decoder establishes:
  - How well classical threshold-based decoding performs on TEMPO data
  - A calibration of task difficulty
  - What learned SNN dynamics add beyond channel counting

Conditions evaluated:
  Experiment 2 (temporal irreducibility):
    intact    — reads channels in time order → limited only by cross-channel errors
    shuffled  — timestamps permuted → ordering lost → collapses toward count baseline
    collapsed — all spikes at t=0   → only channel counts available → count-only ceiling

  Experiment 3 (noise sweep):
    intact condition evaluated at each of 7 sigma_test levels

Usage:
    cd notebooks/
    python rule_based_decoder.py

Outputs:
    Console tables (matching paper Table 2 and Experiment 3 format)
    rule_based_results.csv  (per-seed, per-condition, per-sigma accuracies)
"""

import os
import csv
import numpy as np
from collections import defaultdict

# ── Morse table (ITU standard, A–Z only) ──────────────────────────────────────
MORSE_TABLE = {
    'A': '.-',   'B': '-...', 'C': '-.-.', 'D': '-..',
    'E': '.',    'F': '..-.', 'G': '--.',  'H': '....',
    'I': '..',   'J': '.---', 'K': '-.-',  'L': '.-..',
    'M': '--',   'N': '-.',   'O': '---',  'P': '.--.',
    'Q': '--.-', 'R': '.-.',  'S': '...',  'T': '-',
    'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-',
    'Y': '-.--', 'Z': '--..',
}

LETTERS   = sorted(MORSE_TABLE.keys())
N_CLASSES = len(LETTERS)

# Reverse lookup: dot/dash pattern → letter
REVERSE_MORSE = {v: k for k, v in MORSE_TABLE.items()}

# Count-signature table: (n_dots, n_dashes) → sorted list of letters
COUNT_TABLE = defaultdict(list)
for letter, pattern in MORSE_TABLE.items():
    COUNT_TABLE[(pattern.count('.'), pattern.count('-'))].append(letter)
COUNT_TABLE = {k: sorted(v) for k, v in COUNT_TABLE.items()}

# ── Protocol parameters (must match Experiment 1 calibration) ─────────────────
WPM          = 20
T_U          = 1200.0 / WPM     # 60.0 ms at 20 WPM
SIGMA        = 0.575             # fixed protocol sigma = 0.575 T_u
T_THRESH     = 1.92              # decision threshold (× T_u); = 115.2 ms
MU_R         = 1.2269            # LogNormal r: log-space mean
SIGMA_R      = 0.2916            # LogNormal r: log-space std
MU_OMEGA     = 0.0360            # LogNormal ω: log-space mean
SIGMA_OMEGA  = 0.2446            # LogNormal ω: log-space std
MAX_T        = 1600              # spike tensor length (1 bin = 1 ms)

# ── Experiment settings ────────────────────────────────────────────────────────
N_SEEDS          = 10
N_TEST           = 100           # test samples per class per condition
SEED             = 42
# σ_test sweep matches the updated Experiment 3 sweep (includes protocol σ)
SIGMA_TEST_FRACS = [0.00, 0.20, 0.40, 0.575, 0.80, 1.00, 1.20]

# ── TEMPO encoder ──────────────────────────────────────────────────────────────
def encode_letter(letter, t_u, sigma_ms, rng, omega=None, r=None):
    """
    Encode a single Morse letter as a list of (timestamp_ms, channel) spikes.

    Channel assignment: 0 = DOT_CH (duration < T_thresh), 1 = DASH_CH.
    Sigma, omega, and r match current paper parameters (Experiment 1 calibration).
    omega and r default to LogNormal draws; pass explicit values for deterministic
    encoding (e.g., omega=1.0, r=3.0 for the crossover ablation).
    """
    t_thresh = T_THRESH * t_u
    min_dur  = 0.1 * t_u
    _omega   = rng.lognormal(MU_OMEGA, SIGMA_OMEGA) if omega is None else float(omega)
    _r       = rng.lognormal(MU_R,     SIGMA_R)     if r     is None else float(r)

    pattern = MORSE_TABLE[letter]
    t_cur   = 0.0
    spikes  = []

    for element in pattern:
        t_ideal   = (1.0 if element == '.' else _r) * t_u
        t_weighted = t_ideal * _omega
        t_noisy   = t_weighted + (rng.normal(0, sigma_ms) if sigma_ms > 0 else 0.0)
        t_final   = max(t_noisy, min_dur)
        t_spike   = t_cur + t_final
        channel   = 0 if t_final < t_thresh else 1
        spikes.append((t_spike, channel))
        # Intra-character gap
        gap = max(1.0 * t_u * _omega
                  + (rng.normal(0, sigma_ms) if sigma_ms > 0 else 0.0), min_dur)
        t_cur = t_spike + gap

    # Replace last intra-char gap with inter-character gap
    if spikes:
        gap = max(3.0 * t_u * _omega
                  + (rng.normal(0, sigma_ms) if sigma_ms > 0 else 0.0), min_dur)
        t_cur = spikes[-1][0] + gap

    # Inter-word gap (single-char framing: letter is its own word)
    if spikes:
        gap = max(7.0 * t_u * _omega
                  + (rng.normal(0, sigma_ms) if sigma_ms > 0 else 0.0), min_dur)
        t_cur = spikes[-1][0] + gap  # noqa: F841

    return spikes


def make_spike_tensor(spikes, max_t):
    """Convert spike list to binary tensor shape [max_t, 2]."""
    tensor = np.zeros((max_t, 2), dtype=np.float32)
    for t_ms, ch in spikes:
        idx = int(round(t_ms))
        if 0 <= idx < max_t:
            tensor[idx, ch] = 1.0
    return tensor


def generate_test_split(n_per_class, sigma_ms, rng, omega=None, r=None):
    """Generate test tensors for all 26 letters. Returns (tensors, labels)."""
    tensors, labels = [], []
    for label_idx, letter in enumerate(LETTERS):
        for _ in range(n_per_class):
            spikes = encode_letter(letter, T_U, sigma_ms, rng,
                                   omega=omega, r=r)
            tensors.append(make_spike_tensor(spikes, MAX_T))
            labels.append(label_idx)
    return np.array(tensors, dtype=np.float32), np.array(labels, dtype=np.int32)

# ── Spike transforms (matching notebook implementations) ───────────────────────
def transform_intact(tensor):
    return tensor.copy()


def transform_shuffled(tensor, rng):
    """Permute time bins independently per channel; preserves spike counts."""
    t   = tensor.copy()
    T   = t.shape[0]
    t[:, 0] = t[rng.permutation(T), 0]
    t[:, 1] = t[rng.permutation(T), 1]
    return t


def transform_collapsed(tensor):
    """
    Replace each time bin with the total per-channel spike counts.
    After transformation every row = [n_dots, n_dashes] (the channel totals).
    """
    counts = tensor.sum(axis=0, keepdims=True)   # [1, 2]
    return np.broadcast_to(counts, tensor.shape).copy()

# ── Rule-based decoder ─────────────────────────────────────────────────────────
def decode_binary(tensor):
    """
    Channel-reading decoder for a binary spike tensor.

    Extracts spike events in timestamp order, reconstructs the dot/dash
    sequence from channel assignments, and performs a reverse Morse lookup.
    Returns the predicted letter index (0–25) or -1 on no-match.
    """
    dot_times  = np.where(tensor[:, 0] > 0)[0]
    dash_times = np.where(tensor[:, 1] > 0)[0]

    events = [(t, '.') for t in dot_times] + [(t, '-') for t in dash_times]
    events.sort(key=lambda x: x[0])

    if not events:
        return -1

    pattern = ''.join(e[1] for e in events)
    letter  = REVERSE_MORSE.get(pattern, None)
    return LETTERS.index(letter) if letter is not None else -1


def decode_count(tensor):
    """
    Count-based decoder for the collapsed condition.

    Reads the per-channel totals from the first row of the collapsed tensor
    and looks up (n_dots, n_dashes) in the count-signature table.
    For ambiguous signatures (multiple letters share the same counts), picks
    the alphabetically first candidate — equivalent to the optimal count
    classifier committing to one prediction per distinct signature group.
    Returns predicted letter index or -1.
    """
    n_dots   = int(round(float(tensor[0, 0])))
    n_dashes = int(round(float(tensor[0, 1])))
    candidates = COUNT_TABLE.get((n_dots, n_dashes), [])
    if not candidates:
        return -1
    return LETTERS.index(candidates[0])   # alphabetically first


def decode(tensor, condition, rng=None):
    """
    Dispatch to the appropriate decoding strategy for each test condition.

    intact    → channel-reading on binary tensor
    shuffled  → channel-reading after timestamp permutation
    collapsed → count-based lookup on channel totals
    """
    if condition == 'intact':
        return decode_binary(transform_intact(tensor))
    elif condition == 'shuffled':
        return decode_binary(transform_shuffled(tensor, rng))
    elif condition == 'collapsed':
        return decode_count(transform_collapsed(tensor))
    else:
        raise ValueError(f'Unknown condition: {condition}')

# ── Evaluation utilities ───────────────────────────────────────────────────────
def evaluate(tensors, labels, condition, rng=None):
    """Return accuracy (float 0–1) for a set of tensors."""
    correct = sum(
        decode(t, condition, rng=rng) == int(l)
        for t, l in zip(tensors, labels)
    )
    return correct / len(labels)


def wilson_ci(p, n, z=1.96):
    """Wilson 95% confidence interval for a proportion p observed over n trials."""
    lo = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    hi = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return lo, hi

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('Rule-based Morse Decoder Baseline  (R1.3 / R2.1)')
    print('=' * 62)
    print(f'  sigma        = {SIGMA} T_u = {SIGMA * T_U:.2f} ms  (fixed)')
    print(f'  T_thresh     = {T_THRESH} T_u = {T_THRESH * T_U:.2f} ms')
    print(f'  r            ~ LogNormal(mu={MU_R}, sigma={SIGMA_R})')
    print(f'  omega        ~ LogNormal(mu={MU_OMEGA}, sigma={SIGMA_OMEGA})')
    print(f'  Seeds        = {N_SEEDS}  (base seed {SEED})')
    print(f'  N_test       = {N_TEST} samples per class ({N_TEST * N_CLASSES} total)')
    print()

    n_total  = N_TEST * N_CLASSES   # samples per seed per condition
    csv_rows = []

    # ── Experiment 2: temporal irreducibility conditions ───────────────────────
    print('─' * 62)
    print(f'Experiment 2 — Intact / Shuffled / Collapsed')
    print(f'(sigma = {SIGMA} T_u, stochastic omega and r)')
    print('─' * 62)

    exp2_accs = {'intact': [], 'shuffled': [], 'collapsed': []}

    for si in range(N_SEEDS):
        seed = SEED + si
        # Test data RNG: matches exp2 notebook convention (seed + 10_000)
        rng_test  = np.random.default_rng(seed + 10_000)
        tensors, labels = generate_test_split(N_TEST, SIGMA * T_U, rng_test)

        acc_i = evaluate(tensors, labels, 'intact')
        # Shuffled transform RNG: matches exp2 notebook convention (seed + 20_000)
        rng_shuf  = np.random.default_rng(seed + 20_000)
        acc_s = evaluate(tensors, labels, 'shuffled', rng=rng_shuf)
        acc_c = evaluate(tensors, labels, 'collapsed')

        exp2_accs['intact'].append(acc_i)
        exp2_accs['shuffled'].append(acc_s)
        exp2_accs['collapsed'].append(acc_c)

        print(f'  Seed {seed}:  intact={acc_i*100:5.1f}%  '
              f'shuffled={acc_s*100:5.1f}%  collapsed={acc_c*100:5.1f}%')

        for cond, acc in [('intact', acc_i), ('shuffled', acc_s), ('collapsed', acc_c)]:
            csv_rows.append({'experiment': '2', 'condition': cond,
                             'sigma_frac': SIGMA, 'seed': seed, 'accuracy': acc})

    print()
    print(f'  {"Condition":<12}  {"Mean%":>7}  {"Std%":>5}  {"95% CI":>18}')
    print('  ' + '-' * 48)
    for cond in ['intact', 'shuffled', 'collapsed']:
        vals   = [v * 100 for v in exp2_accs[cond]]
        m, s   = np.mean(vals), np.std(vals)
        mean_p = np.mean(exp2_accs[cond])
        lo, hi = wilson_ci(mean_p, n_total * N_SEEDS)
        print(f'  {cond:<12}  {m:>7.2f}  {s:>5.2f}  [{lo*100:.2f}%, {hi*100:.2f}%]')

    print()
    chance  = 100.0 / N_CLASSES
    uc_base = 7.0  / N_CLASSES * 100
    ceiling = 13.0 / N_CLASSES * 100
    print(f'  Reference baselines:  chance={chance:.1f}%  '
          f'unique-count={uc_base:.1f}%  count-ceiling={ceiling:.1f}%')

    # Delta from intact to shuffled: the core temporal irreducibility statistic
    mean_i = np.mean(exp2_accs['intact'])   * 100
    mean_s = np.mean(exp2_accs['shuffled']) * 100
    mean_c = np.mean(exp2_accs['collapsed'])* 100
    print()
    print(f'  Δ(intact − shuffled)    = {mean_i - mean_s:+.1f} pp')
    print(f'  Δ(shuffled − collapsed) = {mean_s - mean_c:+.1f} pp')
    print(f'  Δ(intact − collapsed)   = {mean_i - mean_c:+.1f} pp')

    # ── Experiment 3: noise sweep ──────────────────────────────────────────────
    print()
    print('─' * 62)
    print('Experiment 3 — Noise sweep (intact condition, stochastic omega and r)')
    print('─' * 62)

    exp3_accs = {sf: [] for sf in SIGMA_TEST_FRACS}

    for si in range(N_SEEDS):
        seed = SEED + si
        row_str = f'  Seed {seed}:'
        for sf in SIGMA_TEST_FRACS:
            # RNG matches exp3 notebook: seed * 100_000 + int(round(sf * 1000))
            rng_test = np.random.default_rng(
                seed * 100_000 + int(round(sf * 1000)))
            tensors, labels = generate_test_split(N_TEST, sf * T_U, rng_test)
            acc = evaluate(tensors, labels, 'intact')
            exp3_accs[sf].append(acc)
            csv_rows.append({'experiment': '3', 'condition': 'intact',
                             'sigma_frac': sf, 'seed': seed, 'accuracy': acc})
            row_str += f'  {sf:.3f}→{acc*100:.1f}%'
        print(row_str)

    print()
    print(f'  {"sigma/Tu":>9}  {"Mean%":>7}  {"Std%":>5}  {"95% CI":>18}')
    print('  ' + '-' * 48)
    for sf in SIGMA_TEST_FRACS:
        vals   = [v * 100 for v in exp3_accs[sf]]
        m, s   = np.mean(vals), np.std(vals)
        mean_p = np.mean(exp3_accs[sf])
        lo, hi = wilson_ci(mean_p, n_total * N_SEEDS)
        marker = '  ← training σ' if sf == SIGMA else ''
        print(f'  {sf:>9.3f}  {m:>7.2f}  {s:>5.2f}  [{lo*100:.2f}%, {hi*100:.2f}%]{marker}')

    # ── Crossover ablation companion ───────────────────────────────────────────
    print()
    print('─' * 62)
    print('Experiment 3 crossover ablation companion')
    print('(fixed omega=1.0, r=3.0 — matches Model D training distribution)')
    print('─' * 62)

    abl_accs = {sf: [] for sf in SIGMA_TEST_FRACS}

    for si in range(N_SEEDS):
        seed = SEED + si
        for sf in SIGMA_TEST_FRACS:
            # Ablation RNG offset (+500_000) matches exp3 notebook convention
            rng_test = np.random.default_rng(
                seed * 100_000 + int(round(sf * 1000)) + 500_000)
            tensors, labels = generate_test_split(
                N_TEST, sf * T_U, rng_test, omega=1.0, r=3.0)
            acc = evaluate(tensors, labels, 'intact')
            abl_accs[sf].append(acc)
            csv_rows.append({'experiment': '3_ablation', 'condition': 'intact',
                             'sigma_frac': sf, 'seed': seed, 'accuracy': acc})

    print(f'  {"sigma/Tu":>9}  {"Mean%":>7}  {"Std%":>5}')
    print('  ' + '-' * 28)
    for sf in SIGMA_TEST_FRACS:
        vals = [v * 100 for v in abl_accs[sf]]
        m, s = np.mean(vals), np.std(vals)
        marker = '  ← training σ' if sf == SIGMA else ''
        print(f'  {sf:>9.3f}  {m:>7.2f}  {s:>5.2f}{marker}')

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'rule_based_results.csv')
    fieldnames = ['experiment', 'condition', 'sigma_frac', 'seed', 'accuracy']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print()
    print(f'Results saved → {out_path}')

    # ── Count-signature analysis ───────────────────────────────────────────────
    print()
    print('─' * 62)
    print('Count-signature analysis')
    print('─' * 62)
    print(f'  {"(dots,dashes)":>14}  {"Letters":>30}  {"Unique?"}')
    print('  ' + '-' * 56)
    for sig in sorted(COUNT_TABLE.keys()):
        letters   = COUNT_TABLE[sig]
        is_unique = 'yes' if len(letters) == 1 else f'no ({len(letters)} letters)'
        print(f'  {str(sig):>14}  {" ".join(letters):>30}  {is_unique}')

    n_unique = sum(1 for v in COUNT_TABLE.values() if len(v) == 1)
    print()
    print(f'  Unique signatures: {n_unique}/26 letters  → '
          f'unique-count baseline = {n_unique/N_CLASSES*100:.1f}%')
    print(f'  Distinct signatures: {len(COUNT_TABLE)}/26  → '
          f'count-only ceiling = {len(COUNT_TABLE)/N_CLASSES*100:.1f}%')


if __name__ == '__main__':
    main()
