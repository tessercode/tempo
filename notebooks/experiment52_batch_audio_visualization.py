"""Batch-generate audio cleaning figures for all calibration_dataset recordings.

Run from the repo root:
    .venv/bin/python notebooks/experiment52_batch_audio_visualization.py

Output: figures/exp52_audio_cleaning/<stem>.pdf  (150 DPI, one per recording)
Already-existing files are skipped so the script is safe to re-run.
"""

import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import librosa
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / 'calibration_dataset'

# ── Config ───────────────────────────────────────────────────────────────────
SR_TARGET  = 8000
N_FFT      = 1024
HOP        = 512       # larger hop reduces STFT size; fine for a paper figure
MAX_FRAMES = 4000      # downsample time axis to this many columns before plotting
FMAX       = 4000
DB_RANGE   = 80
BP_LO_HZ   = 350
BP_HI_HZ   = 1200
FIGSIZE    = (14, 9)
SEG_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63',
              '#9C27B0', '#00BCD4', '#8BC34A', '#FF5722']


# ── Figure function ──────────────────────────────────────────────────────────
def make_figure(orig_path, clean_path, label_path, out_path):
    orig_path, clean_path, label_path, out_path = (
        Path(p) for p in [orig_path, clean_path, label_path, out_path]
    )

    # Load
    orig,  _ = librosa.load(str(orig_path),  sr=SR_TARGET, mono=True)
    clean, _ = librosa.load(str(clean_path), sr=SR_TARGET, mono=True)

    # Labels
    segments = []
    with open(label_path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            t0, t1 = float(parts[0]), float(parts[1])
            lbl = parts[2].strip() if len(parts) > 2 else ''
            segments.append((t0, t1, lbl))

    # Crop to shorter of the two
    duration   = min(len(orig), len(clean)) / SR_TARGET
    orig_crop  = orig[:int(duration * SR_TARGET)]
    clean_crop = clean[:int(duration * SR_TARGET)]
    del orig, clean
    gc.collect()

    segs_crop = [(t0, t1, lbl) for t0, t1, lbl in segments if t0 < duration]

    # Spectrograms — compute, then immediately downsample time axis
    S_orig  = np.abs(librosa.stft(orig_crop,  n_fft=N_FFT, hop_length=HOP, window='hann'))
    ref_amp = float(S_orig.max())
    D_orig  = librosa.amplitude_to_db(S_orig, ref=ref_amp)
    del S_orig

    S_clean = np.abs(librosa.stft(clean_crop, n_fft=N_FFT, hop_length=HOP, window='hann'))
    D_clean = librosa.amplitude_to_db(S_clean, ref=ref_amp)
    del S_clean
    gc.collect()

    freqs      = librosa.fft_frequencies(sr=SR_TARGET, n_fft=N_FFT)
    freq_mask  = freqs <= FMAX
    freqs_plot = freqs[freq_mask]

    # Subsample time axis so we never feed more than MAX_FRAMES columns to imshow
    n_frames = D_orig.shape[1]
    if n_frames > MAX_FRAMES:
        idx = np.linspace(0, n_frames - 1, MAX_FRAMES, dtype=int)
        D_orig  = D_orig[:, idx]
        D_clean = D_clean[:, idx]
        frame_times = librosa.frames_to_time(idx, sr=SR_TARGET, hop_length=HOP)
    else:
        frame_times = librosa.frames_to_time(np.arange(n_frames),
                                              sr=SR_TARGET, hop_length=HOP)

    D_orig_plot  = D_orig[freq_mask, :]
    D_clean_plot = D_clean[freq_mask, :]
    del D_orig, D_clean
    gc.collect()

    # Build figure
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=FIGSIZE,
        gridspec_kw={'height_ratios': [2, 2, 1]}
    )
    fig.patch.set_facecolor('white')
    CBAR_SIZE, CBAR_PAD = '2%', 0.08

    def _specgram(D_plot, ax, title):
        # imshow renders at display resolution — no vertex mesh, low memory
        img = ax.imshow(
            D_plot,
            origin='lower', aspect='auto',
            extent=[frame_times[0], frame_times[-1],
                    freqs_plot[0],  freqs_plot[-1]],
            cmap='inferno', vmin=-DB_RANGE, vmax=0,
            interpolation='bilinear', rasterized=True
        )
        for hz, va in [(BP_LO_HZ, 'bottom'), (BP_HI_HZ, 'top')]:
            ax.axhline(hz, color='cyan', linewidth=0.9, linestyle='--', alpha=0.75)
            ax.text(duration * 0.002, hz, f'{hz} Hz',
                    color='cyan', fontsize=7, va=va, ha='left')
        ax.set_ylabel('Frequency (Hz)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
        ax.tick_params(labelsize=8)
        return img

    img1 = _specgram(D_orig_plot,  ax1,
                     f'Original Recording — {orig_path.stem}')
    img2 = _specgram(D_clean_plot, ax2,
                     f'Cleaned  ({BP_LO_HZ}–{BP_HI_HZ} Hz bandpass + notch filters)')
    del D_orig_plot, D_clean_plot
    gc.collect()

    for ax, img in [(ax1, img1), (ax2, img2)]:
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size=CBAR_SIZE, pad=CBAR_PAD)
        cb  = fig.colorbar(img, cax=cax)
        cb.set_label('dB', rotation=0, labelpad=8, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes('right', size=CBAR_SIZE, pad=CBAR_PAD)
    cax3.axis('off')

    # Waveform — downsample to ~100k points for display
    MAX_WAVE_PTS = 100_000
    if len(clean_crop) > MAX_WAVE_PTS:
        w_idx = np.linspace(0, len(clean_crop) - 1, MAX_WAVE_PTS, dtype=int)
        t_arr      = w_idx / SR_TARGET
        wave_plot  = clean_crop[w_idx]
    else:
        t_arr     = np.arange(len(clean_crop)) / SR_TARGET
        wave_plot = clean_crop
    del clean_crop
    gc.collect()

    ax3.plot(t_arr, wave_plot, linewidth=0.3, color='#1565C0', alpha=0.9, rasterized=True)
    yabs = max(np.abs(wave_plot).max(), 1e-6)
    ax3.set_ylim(-yabs * 1.15, yabs * 1.15)
    del wave_plot, t_arr
    gc.collect()

    for i, (t0, t1, lbl) in enumerate(segs_crop):
        color = SEG_COLORS[i % len(SEG_COLORS)]
        ax3.axvspan(t0, t1, alpha=0.22, color=color, linewidth=0)
        ax3.axvline(t0, color=color, linewidth=0.6, alpha=0.8)
        ax3.axvline(t1, color=color, linewidth=0.6, alpha=0.8)
        if lbl:
            ax3.text((t0 + t1) / 2, 0.95, lbl,
                     ha='center', va='top', fontsize=7, color='#222222',
                     clip_on=True, transform=ax3.get_xaxis_transform())

    ax3.axhline(0, color='gray', linewidth=0.3, alpha=0.5)
    ax3.set_ylabel('Amplitude', fontsize=9)
    ax3.set_xlabel('Time (s)', fontsize=9)
    ax3.set_title('Cleaned Waveform — Message Segments',
                  fontsize=10, fontweight='bold', pad=4)
    ax3.tick_params(labelsize=8)

    ax1.set_xlim(0, duration)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    fig.subplots_adjust(hspace=0.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()


# ── Discover recordings ──────────────────────────────────────────────────────
recordings = []
for wav_path in sorted(DATASET_ROOT.rglob('*.wav')):
    stem      = wav_path.stem
    parent    = wav_path.parent
    orig_path = next(
        (parent / (stem + ext) for ext in ['.mp3', '.amr', '.flac']
         if (parent / (stem + ext)).exists()),
        None
    )
    label_path = parent / (stem + '.txt')
    if orig_path and label_path.exists():
        recordings.append((orig_path, wav_path, label_path))

print(f'Found {len(recordings)} recordings\n')

ok, skipped, errors = 0, 0, []
for i, (orig_path, clean_path, label_path) in enumerate(recordings, 1):
    out_path = orig_path.parent / (orig_path.stem + '.pdf')
    label    = f'[{i:2d}/{len(recordings)}] {orig_path.stem[:65]}'
    if out_path.exists():
        print(f'  skip  {label}')
        skipped += 1
        continue
    print(f'  gen   {label}', end=' ... ', flush=True)
    try:
        make_figure(orig_path, clean_path, label_path, out_path)
        print('OK')
        ok += 1
    except Exception as e:
        print(f'ERROR: {e}')
        errors.append((orig_path.name, str(e)))

print(f'\nDone — generated {ok}, skipped {skipped}, errors {len(errors)}')
for name, msg in errors:
    print(f'  ERROR {name}: {msg}')
