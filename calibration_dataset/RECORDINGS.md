# TEMPO Calibration Dataset — Recording Analysis

**Purpose:** Provide robust real-world calibration of TEMPO encoding parameters ω (per-word speed
multiplier), r (dot/dash duration ratio), and σ (timing noise) across a wide range of operator
speeds and keying styles. Collected in response to reviewer feedback on Experiment 4.

**Parameter definitions:**
- **WPM** — estimated speed in words per minute from envelope analysis
- **σ_elem** — timing jitter of element durations, in units of T_u (dot period)
- **r** — dash/dot duration ratio (nominal = 3.0; bug keys → higher)
- **SNR** — carrier-to-noise ratio in dB from tone survey
- **Elems** — number of dot + dash elements extracted (proxy for usable data volume)

All σ_elem, r, WPM values are from the Phase 7 per-recording envelope analysis (`analysis/phase7_per_recording.csv`).

---

## Already Labeled

### Exp 4 — Italian QRS Net Operators (`labeled/exp4/`)

Six operators recorded on 2026-03-12 from the IZ8VKW-hosted QRS nets (7.030 MHz and 3.557 MHz).
These are the original Experiment 4 calibration recordings. MP3 files are the originals. WAV files
are cleaned versions produced in Audacity (see Audio Cleaning below). TXT files are Audacity label
files with manually marked message segment boundaries. AUP3 files are the Audacity project files.

| File | Date | Operator | Band | WPM est. | σ_elem | r | Notes |
|------|------|----------|------|----------|--------|---|-------|
| `20260312_it9etc iz8vkw_qrs_7030 khz` | 2026-03-12 | IT9ETC | 40m | ~15.6 | ~0.44 | ~3.0 | Sicilian op; vertical key |
| `20260312_iu2udw iz8vkw_qrs_3557 khz` | 2026-03-12 | IU2UDW | 80m | ~15.4 | ~0.44 | ~3.0 | Paddle |
| `20260312_iu7qci iz8vkw_qrs_3557 khz` | 2026-03-12 | IU7QCI | 80m | ~16.1 | ~0.44 | ~3.0 | Paddle |
| `20260312_iw5dua iz8vkw_qrs_3557 khz` | 2026-03-12 | IW5DUA | 80m | ~14.5 | ~0.44 | ~3.0 | Paddle |
| `20260312_iz0rga iz8vkw_qrs_7030 khz` | 2026-03-12 | IZ0RGA | 40m | ~18.7 | ~0.44 | ~3.0 | Paddle |
| `20260312_iz4dyx iz8vkw_qrs_3557 khz` | 2026-03-12 | IZ4DYX | 80m | ~16.5 | ~0.44 | ~3.0 | Paddle; identified as catastrophic failure case in Exp 47 (4*T_u ISI ambiguity) |

**Speed range covered:** ~14–19 WPM. All operators show σ_elem ≈ 0.44 T_u (pooled Exp 4/26 value).
**Note:** WB8MON slow net (5–12 WPM, `data/nets/slownet_20260401.wav`) referenced in the Exp 4
notebook does not exist on disk; it is not included here.

---

### QRQcw CFOnet Recordings (`labeled/qrqcw/`)

Eight recordings from the Chicken Fat Operators net (CFOnet), 2011–2012. Single operator (QRQcw)
sending at high speed. FLAC files are the originals; WAV files are cleaned versions produced in
Audacity (see Audio Cleaning below); TXT files are Audacity label files with manually marked segment
boundaries. AUP3 files are the Audacity project files. Analysis completed May 2026 (see
`ChickenFatOperators/ANALYSIS_REPORT.md`).

| File | Date | WPM est. | Notes |
|------|------|----------|-------|
| `QRQcw - CFOnet 12-20-2011` | 2011-12-20 | ~35.7 | Highest fldigi linguistic score (0.6147); high SNR |
| `QRQcw - CFOnet 2-28-2012` | 2012-02-28 | ~35.7 | Largest recording (68 segments) |
| `QRQcw - CFOnet 3-1-2011`  | 2011-03-01 | ~35.7 | Best fldigi score overall (0.6560); slow-pass useful (9%) |
| `QRQcw - CFOnet 3-29-2011` | 2011-03-29 | ~35.7 | Cleaning helps most (55% char win) |
| `QRQcw - CFOnet 4-24-2012` | 2012-04-24 | ~35.7 | Slow-pass useful (7.7%) |
| `QRQcw - CFOnet 8-23-2011` | 2011-08-23 | ~35.7 | Missing multimon slow-pass data |
| `QRQcw - CFOnet 8-30-2011` | 2011-08-30 | ~35.7 | Clean fldigi advantage |
| `QRQcw - CFOnet 9-6-2011`  | 2011-09-06 | ~35.7 | Only recording where multimon edges ahead of fldigi |

**Speed range covered:** ~35–37 WPM. SG-64 letter F1 improved from 10.4% (Exp 30A BiGRU) to
56.2% (Exp 50 BiGRU) on these recordings (Exp 51).

---

## Candidates for Labeling

The corpus floor is **10.2 WPM** — no recordings below that exist in `cw_qso_recordings/`.
Recordings are listed in priority order within each group.

---

### Group A — Core WPM Gap: 22–35 WPM (`candidates/group_a_wpm_gap/`)

**Priority: Highest.** Zero labeled recordings currently exist in this range. These eight recordings
fill the central gap between the slow Italian ops (~19 WPM) and QRQcw (~35 WPM).

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 1 | `20210613_pa-iz0ngh iz8vkw_paddle_14039 khz.mp3` | 2021-06-13 | 28.2 | **0.069** | 2.91 | 53 | 1627 | Exceptionally clean — lowest σ in any candidate; best-case noise floor |
| 2 | `20221103_ik6ihm iz8vkw_paddle_qrq_7027 khz.mp3` | 2022-11-03 | 32.0 | 0.088 | 2.84 | 46 | 1880 | Very clean paddle at 32 WPM; high element count |
| 3 | `20230101_ik6ihm iz8vkw_30 wpm_paddle_test keyer_3567 khz.mp3` | 2023-01-01 | 22.9 | 0.071 | 2.70 | 48 | 1487 | Electronic keyer test — pristine timing; 23 WPM speed point |
| 4 | `20190823_9a9cw iz8vkw_janus_7008 khz.mp3` | 2019-08-23 | 28.2 | 0.093 | 2.79 | 51 | 1650 | 9A9CW = Croatian op; stylistically distinct from Italian net operators |
| 5 | `20230210_iu0pjj i8qfk iz8vkw_25-40 wpm_7034 khz.mp3` | 2023-02-10 | 28.2 | 0.113 | 2.72 | **59** | 1789 | Best SNR in this group (59 dB); variable speed label in filename (25–40 WPM) |
| 6 | `20230527_i8qfk iz8vkw_paddle schurr - paddle ik1ojm prototipo7034 khz.mp3` | 2023-05-27 | 32.0 | 0.115 | 2.66 | 55 | **2074** | Most elements in this group; I8QFK Schurr paddle |
| 7 | `20210619_dk4lx iz8vkw_paddle_20900 khz.mp3` | 2021-06-19 | 25.3 | 0.200 | 2.53 | 54 | 1725 | 25 WPM speed point; DK4LX = German op (geographic diversity) |
| 8 | `20200820_pa-.iz0ngh iz8vkw_paddles_qrq.mp3` | 2020-08-20 | 32.0 | 0.337 | **2.43** | 52 | 2013 | Also provides extreme low-r data point (r=2.43); PA-/IZ0NGH dual paddle |

---

### Group B — Fast Tier Extension: 37–44 WPM (`candidates/group_b_fast/`)

**Priority: High.** QRQcw is labeled but represents a single operator style. These add operator
diversity at the fast end and extend coverage to 43.6 WPM (the only >40 WPM recording in Exp 31).

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 9  | `20230808_ik7ukf iz8vkw_paddle_7031 khz.mp3` | 2023-08-08 | 36.9 | 0.309 | 3.04 | **57** | **2166** | Best quality in fast tier; IK7UKF paddle |
| 10 | `20230707_ik1wjq iz8vkw i8gmg_30 wpm_10114 khz.mp3` | 2023-07-07 | 36.9 | 0.228 | 3.22 | 52 | 2037 | Clean fast op; IK1WJQ + I8GMG (different callsigns from usual net) |
| 11 | `20220623_ik6ihm iz8vkw_bug_paddle_qrq_7027 khz.mp3` | 2022-06-23 | **43.6** | 1.202 | **5.14** | 53 | 1683 | **Only recording >40 WPM in Exp 31.** IK6IHM bug key; very high σ and r — plan for difficult labeling |

**Note:** `20230117_ik6ihm iz8vkw_paddle_qrq_25-45_3555 khz.mp3` was considered but excluded —
the 25–45 WPM variable speed in the filename makes T_u estimation unreliable for calibration.

---

### Group C — High-σ Noisy Operators (`candidates/group_c_high_sigma/`)

**Priority: High.** The Italian ops cluster at σ ≈ 0.44 T_u. These two recordings extend coverage
to σ ≈ 0.79 and σ ≈ 1.0, which is critical for calibrating the upper tail of the noise distribution.

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 12 | `20231002_ik6ihm iz8vkw i1dmp iu1czf_qrq test IK4POF keyer_3554 khz.mp3` | 2023-10-02 | 32.0 | 0.788 | 2.77 | 50 | 2066 | Multi-op QRQ test session; high σ + high element count |
| 13 | `20190921-IZ4KBW-IK0AAE-MONARCH-7026.amr` | 2019-09-21 | 28.2 | 0.997 | 3.85 | 42 | 1552 | Vibroplex Monarch bug key; σ near 1.0 T_u — highest-noise clean candidate |

---

### Group D — Bug Key / Extreme r (`candidates/group_d_bug_key/`)

**Priority: Medium–High.** Bug key operators produce systematically higher r ratios (r > 4.0).
Important for calibrating the right tail of the r distribution, which is absent from the Exp 4 data.

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 14 | `20210902_ik7ukf_iz7fun_2xBug_qrs.mp3` | 2021-09-02 | 28.2 | 0.544 | 4.52 | **61** | 1275 | Both operators using bug keys (dual bug QSO); excellent SNR |
| 15 | `20210320_iz4kbw iz8vkw_bug_5356 khz.mp3` | 2021-03-20 | 32.0 | 0.562 | 4.25 | 54 | 1480 | IZ4KBW bug key at 32 WPM |

**Note:** `4_5958620614657837558.mp3` found in the same directory as #15 is a confirmed duplicate
(identical statistics and dot/dash counts) — do not label it separately.

---

### Group E — Slow Tier: 14–15 WPM (`candidates/group_e_slow/`)

**Priority: Medium.** Complements the Exp 4 Italian ops (14–19 WPM) with clean operators and
distinctive keying styles; also adds geographic diversity.

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 16 | `20241105_iv3ram iz8vkw_qrs_10118 khz.mp3` | 2024-11-05 | 14.5 | 0.097 | 2.95 | 56 | 798 | IV3RAM very clean; excellent σ contrast with Italian op cluster |
| 17 | `20220101_iz8qpa iz8vkw_qrs_28055 khz.mp3` | 2022-01-01 | 14.5 | 0.139 | 3.06 | **58** | 860 | IZ8QPA; high SNR; good for confirming low-σ slow-speed behavior |
| 18 | `20190628_hwk7_6998 khz.mp3` | 2019-06-28 | 11.2 | 0.250 | **2.48** | 54 | 775 | HWK7 (likely North American op); lowest WPM in this group; also low-r data point |

---

### Group E2 — Very Slow: 10–14 WPM (`candidates/group_e2_very_slow/`)

**Priority: High given rarity.** Only 29 recordings in the entire 3,247-file corpus fall below 14 WPM.
These are the recordings closest to the lower boundary of the TEMPO training range (13 WPM after Exp 50).
Each labeled recording in this range carries substantial weight.

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 19 | `it9gvt iz8vkw_2018 03 02_3547 kHz.mp3` | 2018-03-02 | 13.7 | 0.328 | 2.76 | 46 | **2441** | Most elements of any slow recording in the corpus; 2018 QRS net |
| 20 | `iu0hmb iz8vkw_2018 02 26_3544 kHz.mp3` | 2018-02-26 | 13.7 | 0.394 | 3.08 | 50 | 2038 | Second-highest element count in slow tier; same era net |
| 21 | `20240804_i1gzg iz8vkw_qrs_10118 khz.mp3` | 2024-08-04 | 12.3 | 0.434 | 2.87 | 55 | 1648 | Best quality (SNR + elements) in the 10–13 WPM range |
| 22 | `IU8GVN-IZ0KBW-06042019-7025-MONARCH.amr` | 2019-04-06 | 11.7 | 0.523 | 2.61 | 35 | 1331 | Monarch bug key at 11.7 WPM; provides both slow + high-σ data point. Low SNR (35 dB) — may require careful bandpass filtering before labeling |
| 23 | `20211009-IK5TTA-IZ4KBW-18WPM-COOTIE-7035.mp3` | 2021-10-09 | 10.7 | 0.619 | 2.52 | 40 | 1008 | Cootie/sideswiper key at 10.7 WPM; lowest WPM with decent element count |
| 24 | `20220505-I2PHD-IZ4KBW-10WPM-VERTICALE-7033.mp3` | 2022-05-05 | 10.7 | 0.452 | 3.27 | 50 | 602 | 10.7 WPM vertical key; different r (3.27) from #23 |
| 25 | `sq6mih iz0kbw_2018 03 08.amr` | 2018-03-08 | 10.2 | 0.282 | 2.90 | 48 | 730 | **Lowest WPM in the full corpus (10.2 WPM).** SQ6MIH = Polish op. Skip `SQ6MIH.amr` in same directory — confirmed duplicate |

---

### Group E3 — Slow with Diverse Parameters (`candidates/group_e3_slow_diverse/`)

**Priority: Medium.** These 13–15 WPM recordings were selected for keying style variety (bug,
sideswiper, Blueracer), geographic diversity, or distinctive r/σ values not well covered elsewhere.

| # | File | Date | WPM | σ_elem | r | SNR | Elems | Notes |
|---|------|------|-----|--------|---|-----|-------|-------|
| 26 | `20200701_e73kw iz8vkw_7026 khz.mp3` | 2020-07-01 | 13.0 | 0.432 | **2.54** | 56 | 1523 | E73KW = Bosnian op; low r (2.54); geographic diversity |
| 27 | `20201004-IZ4KBW-IU0LSQ-BLUERACER-7032.mp3` | 2020-10-04 | 13.7 | 0.411 | 2.94 | 50 | 1577 | Vibroplex Blueracer bug key; high element count for slow speed |
| 28 | `20210903_iz5ovp iz8vkw_qrs_3556 khz.mp3` | 2021-09-03 | 14.5 | 0.472 | **3.96** | 60 | 1080 | r near 4.0 at slow speed (bug-like ratio at QRS pace); excellent SNR (60 dB) |
| 29 | `20220507_ik6ihm iz8vkw_sideswiper_7027 khz.mp3` | 2022-05-07 | 14.5 | **0.175** | 2.58 | 57 | 1127 | Sideswiper key; very clean (σ=0.175) — low-σ counterpoint to noisy slow ops |
| 30 | `G4LNA - UR3QX - VK6AJ WITH IZ8VKW_30 10 2017.mp3` | 2017-10-30 | 14.5 | 0.524 | 2.94 | 54 | 1152 | Multi-continental DX QSO: G4LNA (UK), UR3QX (Ukraine), VK6AJ (Australia); stylistically the most distinct recording in the dataset |

---

## Parameter Space Coverage Summary

| Group | Count | WPM Range | σ Range | r Range | Primary Calibration Value |
|-------|-------|-----------|---------|---------|--------------------------|
| Exp 4 (labeled) | 6 | 14–19 | ~0.44 | ~3.0 | Baseline slow-speed cluster |
| QRQcw (labeled) | 8 | ~35–37 | unknown | unknown | Fast operator baseline |
| A — Core gap | 8 | 22–35 | 0.07–0.34 | 2.43–2.91 | Fills main unlabeled WPM range |
| B — Fast | 3 | 37–44 | 0.23–1.20 | 3.04–5.14 | Extends beyond QRQcw |
| C — High σ | 2 | 28–32 | 0.79–1.00 | 2.77–3.85 | Upper noise distribution tail |
| D — Bug key | 2 | 28–32 | 0.54–0.56 | 4.25–4.52 | r > 4.0 calibration |
| E — Slow | 3 | 11–15 | 0.10–0.25 | 2.48–3.06 | Clean low-speed ops |
| E2 — Very slow | 7 | 10–14 | 0.28–0.62 | 2.52–3.27 | Near lower model boundary |
| E3 — Slow diverse | 5 | 13–15 | 0.17–0.52 | 2.54–3.96 | Keying style and r diversity |
| **Total candidates** | **30** | **10–44** | **0.07–1.20** | **2.43–5.14** | |

---

## Known Duplicates in the Corpus — Do Not Label

These files in `cw_qso_recordings/` have identical statistics (WPM, σ, r, SNR, dot/dash counts)
and should be treated as single recordings:

| Keep | Skip |
|------|------|
| `IZ7EBY-IZ0KBW-17112018-7029-PADDLE-MARCONI-BRACER.amr` | `IZ7EBY-IZ0KBW-27112018-7029-PADDLE-MARCONI-BRACER.amr` |
| `20210320_iz4kbw iz8vkw_bug_5356 khz.mp3` | `4_5958620614657837558.mp3` (same directory) |
| `sq6mih iz0kbw_2018 03 08.amr` | `SQ6MIH.amr` |
| `20200818-IU4MRU-IZ4KBW-MARCONI213-5356.amr` | `20200818-IU4MRU-IZ4KBW-MARCONI213-5356(1).amr` |
| `20210720-IZ4KBW-DL4AC-15WPM-MARCONI213-7027.mp3` | `20210720-IZ4KBW-DL4AC-15WPM-MARCONI213-7027(1).mp3` |
| `WF1S NM1I.amr` (one copy) | `WF1S NM1I.amr` (second copy, same directory) |

---

## Audio Cleaning

All recordings have been cleaned in Audacity and exported as WAV files with the same stem name as
the original audio file. The cleaning chain applied to every recording:

| Stage | Type | Frequency | Slope / Q |
|-------|------|-----------|-----------|
| High-pass filter | Butterworth | 350 Hz | 48 dB/octave |
| Low-pass filter | Butterworth | 1200 Hz | 48 dB/octave |
| Notch filter(s) | Notch | As needed | Q = 0.75 |

The high-pass and low-pass filters define a 350–1200 Hz passband that captures the full CW tone
range present in the recordings while rejecting low-frequency hum, powerline interference, and
high-frequency noise. Notch filters were applied on a per-recording basis to remove specific
narrowband interference where present. The Audacity project file (`.aup3`) for each recording
preserves the exact filter settings and notch frequencies used.

**File inventory per recording:**

| Extension | Contents |
|-----------|---------|
| `.mp3` / `.flac` / `.amr` | Original unprocessed audio |
| `.wav` | Cleaned audio (350–1200 Hz bandpass + notch filters) |
| `.txt` | Audacity label file — segment boundary timestamps |
| `.aup3` | Audacity project file — preserves filter chain and notch settings |

---

## Audacity Labeling Procedure

Label files (`.txt`) use Audacity's tab-separated format:
```
start_time_s    end_time_s    label
```
Each row marks one message segment. The label text is the decoded content (or left blank if
unknown). For calibration purposes, accurate **boundary timestamps** are the critical output;
transcript accuracy is secondary.

---

*Analysis performed 2026-05-29. Labels and cleaned WAV files completed 2026-05-31. Parameter values from `analysis/phase7_per_recording.csv` (Phase 7 envelope analysis, Exp 31).*
