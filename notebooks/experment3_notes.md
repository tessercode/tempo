# Experiment 3 — Remote Execution Notes

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

## Running the Notebook

### 1. SSH into the remote server and start a named screen session

```bash
screen -S exp3
```

### 2. Navigate to the notebooks directory

```bash
cd /path/to/tempo/notebooks
```

### 3. Launch the notebook with papermill

```bash
papermill experiment3_stochastic_robustness.ipynb \
          experiment3_stochastic_robustness_output.ipynb \
          --log-output \
          --log-level INFO \
          2>&1 | tee experiment3_run.log
```

- Cell outputs are saved to `experiment3_stochastic_robustness_output.ipynb` incrementally after each cell completes — the original notebook is not modified.
- All output is streamed to `experiment3_run.log` in real-time.
- If the job dies mid-run, all completed cells are preserved in the output notebook.

### 4. Detach from the screen session (job keeps running after SSH disconnect)

```
Ctrl+A, then D
```

---

## Monitoring Progress

From any terminal (including after reconnecting via SSH):

```bash
tail -f experiment3_run.log
```

---

## Reconnecting to the Session

```bash
screen -r exp3
```

---

## Troubleshooting

**Kernel not found** — specify the kernel explicitly:

```bash
papermill experiment3_stochastic_robustness.ipynb \
          experiment3_stochastic_robustness_output.ipynb \
          --kernel <kernel_name> \
          --log-output \
          --log-level INFO \
          2>&1 | tee experiment3_run.log
```

**Relative path errors** — add `--cwd` to set the working directory:

```bash
papermill experiment3_stochastic_robustness.ipynb \
          experiment3_stochastic_robustness_output.ipynb \
          --cwd /path/to/tempo/notebooks \
          --log-output \
          --log-level INFO \
          2>&1 | tee experiment3_run.log
```

**List all screen sessions:**

```bash
screen -ls
```

**Kill the session when done:**

```bash
screen -S exp3 -X quit
```
