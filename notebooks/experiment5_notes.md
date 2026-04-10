# Experiment 5 — Remote Execution Notes

## Prerequisites

Install `papermill` on the remote server if not already present:

```bash
pip install papermill
```

Check available Jupyter kernels (to confirm kernel name if needed):

```bash
jupyter kernelspec list
```

---

## What This Experiment Does

Experiment 5 is an architecture ablation study (addressing reviewer concern M5). It trains **30 models** (3 architectures × 10 seeds) and evaluates them across both the temporal irreducibility conditions from Experiment 2 and the noise robustness sweep from Experiment 3.

- **Arch-A** (Benchmark): fast(42) + medium(43) + slow(43) — matches Experiments 2 & 3
- **Arch-B** (No slow group): fast(64) + medium(64) — ablates the ω-invariance claim
- **Arch-C** (Single-timescale): fast(128) — cross-architecture baseline

Training checkpoints are saved to `../checkpoints/exp5/` incrementally; completed models are **skipped on re-run**, so the job is safe to restart if interrupted.

The final output is `../figures/exp5_architecture_ablation.pdf`.

---

## Running the Notebook

### 1. SSH into the remote server and start a named screen session

```bash
screen -S exp5
```

### 2. Navigate to the notebooks directory

```bash
cd /path/to/tempo/notebooks
```

### 3. Ensure the output directories exist

```bash
mkdir -p ../checkpoints/exp5 ../figures
```

### 4. Launch the notebook with papermill

```bash
papermill experiment5_architecture_ablation.ipynb \
          experiment5_architecture_ablation_output.ipynb \
          --log-output \
          --log-level INFO \
          2>&1 | tee experiment5_run.log
```

- Cell outputs are saved to `experiment5_architecture_ablation_output.ipynb` incrementally after each cell completes — the original notebook is not modified.
- All output is streamed to `experiment5_run.log` in real-time.
- Because checkpoints are written per model, a mid-run failure can be resumed from where it left off — completed models will be skipped automatically.

### 5. Detach from the screen session (job keeps running after SSH disconnect)

```
Ctrl+A, then D
```

---

## Monitoring Progress

From any terminal (including after reconnecting via SSH):

```bash
tail -f experiment5_run.log
```

To see how many checkpoints have been saved so far (out of 30):

```bash
ls ../checkpoints/exp5/*.pt | wc -l
```

---

## Reconnecting to the Session

```bash
screen -r exp5
```

---

## Troubleshooting

**Kernel not found** — specify the kernel explicitly:

```bash
papermill experiment5_architecture_ablation.ipynb \
          experiment5_architecture_ablation_output.ipynb \
          --kernel <kernel_name> \
          --log-output \
          --log-level INFO \
          2>&1 | tee experiment5_run.log
```

**Relative path errors** — add `--cwd` to set the working directory:

```bash
papermill experiment5_architecture_ablation.ipynb \
          experiment5_architecture_ablation_output.ipynb \
          --cwd /path/to/tempo/notebooks \
          --log-output \
          --log-level INFO \
          2>&1 | tee experiment5_run.log
```

**Resuming after a failure** — simply re-run the same papermill command. Any checkpoint files already written to `../checkpoints/exp5/` will be detected and those models will be skipped.

**List all screen sessions:**

```bash
screen -ls
```

**Kill the session when done:**

```bash
screen -S exp5 -X quit
```
