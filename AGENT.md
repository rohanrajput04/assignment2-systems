# Agent handoff — CS336 Assignment 2 (Systems)

Read this at the start of a session to resume work. Last updated 2026-11-24.

## How to work with Rohan

- `CLAUDE.md` is the course's AI policy (teaching assistant, not solution
  generator). Rohan is **auditing** the course, not taking it for credit, and
  has said direct fixes are fine when he asks for them explicitly. Default to
  guiding; write or edit code/writeup only when he asks you to.
- He prefers **one step at a time**: give one small, concrete fix, wait for
  "next" / "done, check", then read the file and review.
- When he says "check", read the file fresh — he edits between messages.
- Keep explanations short and concrete; point at exact line numbers.

## Environment

- Local: macOS, no CUDA. `uv` manages deps (`uv sync`). Use `python3`, not `python`.
- GPU: **Modal**. Login once per machine with `uv run modal setup`.
  - App `cs336-assignment-2`, volume `cs336-data-am-2` (mounted at `/result`).
  - `modal_run.py` (repo root, deliberately outside the package):
    - `gpu_run` — H100 sweep; runs `cs336_systems.benchmark` as a subprocess per
      config, appends JSONL to `/result/result.jsonl`, commits after each row.
      Edit the `sizes` / `modes` / `warmups` lists in the file to change the sweep.
    - `toy_dtypes` — L4; runs the autocast dtype experiment.
    - Run with `uv run modal run modal_run.py::start` or `::toy`.
    - Fetch results: `uv run modal volume get cs336-data-am-2 result.jsonl <local-name> --force`
      (remote file is append-only and contains every run ever; filter in pandas).
- Writeup: `writeup.typ` (Typst). Compile with `typst compile writeup.typ`;
  Tinymist VS Code extension for live preview.
- Remotes: `origin` = Rohan's fork (push here), `upstream` = stanford-cs336 (fetch only).
  Working branch: `assignment_setup`.

## Code

- `cs336_systems/benchmark.py` — CLI benchmark. Flags: `--size {small,medium,large,xl,10B}`,
  `--mode {forward,forward_backward,full_step}`, `--warmup_steps`, `--num_steps` (>=2),
  `--no_grad` (forward only), `--dump_times`, plus individual model hyperparameters.
  Builds the model directly on device via `with torch.device(device)`. Syncs after every
  warm-up and measured step. stdout = one JSON row; stderr = diagnostics.
- `cs336_systems/bechmarking_mixed_precision.py` (typo in name, **this is the real one**) —
  ToyModel autocast dtype experiment, CUDA + FP16. `modal_run.py` references this name.
- `cs336_systems/benchmarking_mixed_precision.py` — stale copy (model only). Pending cleanup:
  delete it, rename the real one, update `modal_run.py`.
- `cs336_systems/mixed_precision_accumulation.py` — the four handout summation snippets.

## Progress

### Done
- **2.1.3 benchmarking_script (a)(b)(c)** — script, data, tables and prose in writeup.
- **2.1.5 mixed_precision_accumulation** — table + prose in writeup.
- **2.1.5 benchmarking_mixed_precision (a)** — data collected on L4 (CUDA, FP16):
  params fp32, fc1 out fp16, LayerNorm out fp32, logits fp16, loss fp32, grads fp32.
  ⚠️ `writeup.typ` currently says "Model parameters: fp16" — **wrong, measured fp32**.
  Bullets also still need a one-line reason each.

### Next
1. Fix (a) parameter dtype in writeup; add reasons.
2. **benchmarking_mixed_precision (b)** — written only. Which LayerNorm steps are FP16-sensitive
   (squaring -> overflow vs. summation -> rounding) and whether BF16 (8 exp / 7 mantissa bits)
   removes the need. Observed: CUDA autocast upcasts LayerNorm to fp32 even under BF16.
   Hint already given: `cs336-basics/cs336_basics/model.py:94-100` (RMSNorm upcast comment).
3. **benchmarking_mixed_precision (c)** — add `--autocast/--dtype` to benchmark.py using the same
   `nullcontext` pattern as `--no_grad`; rerun sizes with/without BF16 on H100.
4. **memory_profiling (d)** — arithmetic only (xl residual-stream activation, FP32, MiB).
5. **memory_profiling (a)(b)(c)(e)(f)** — `torch.cuda.memory._record_memory_history` around the
   measured region, dump pickle to the volume, view at pytorch.org/memory_viz. xl at ctx 128 and 2048.
   Use `--no_grad` for the "inference only" case.
6. **nsys_profile (a)–(e)** — last. Needs an NVIDIA CUDA base image with `nsight-systems-cli`
   (debian_slim won't have it), NVTX ranges (exclude warm-up), `.nsys-rep` pulled back via the volume.
7. Later sections: FlashAttention-2 Triton, DDP, optimizer sharding, FSDP.

## Key results and findings (for reference in later answers)

Clean H100 table (ctx 512, batch 4, FP32, 5 warm-up, 10 steps), ms:

| size | params | forward | fwd+bwd | full step |
|---|---|---|---|---|
| small | 128.6M | 20.13 | 61.37 | 69.44 |
| medium | 423.2M | 55.10 | 168.64 | 191.07 |
| large | 969.4M | 123.42 | 384.90 | 436.15 |
| xl | 3.407B | 335.57 | 1072.13 | 1237.97 |
| 10B | 12.83B | OOM | OOM | OOM |

All std < 0.2% of mean.

- **Warm-up sync bug (found and fixed):** an unsynchronized warm-up loop let the CPU run ahead, so the
  first measured step absorbed leftover warm-up work. small/forward was 22.8 ± 8.2 → 20.1 ± 0.04 ms.
  Larger effect on forward than full_step because `optim.step()`'s Python loop keeps the CPU busy.
- (fwd+bwd)/fwd = 3.05 → 3.20, i.e. backward ≈ 2.05–2.20× forward.
- Optimizer step ≈ 49–63 ns/param, decreasing with size (per-tensor Python overhead amortizes).
- Warm-up (large): 0 steps inflates mean 13–21%; forward and fwd+bwd converge after 1 step;
  full_step needs 2 because AdamW allocates m/v lazily on the first `step()` (~7.8 GB for large).
- 10B: 51.3 GB FP32 weights load fine; forward OOMs at the attention einsum because autograd keeps
  ~1 GB of activations per layer × 50 layers. Full training would need ~205 GB.
- Mixed-precision accumulation: fp16 accumulator 9.9531 (−0.47%, swamping); fp32 accumulator with fp16
  addends 10.0021 = 1000 × fp16(0.01), i.e. pure representation error.
- CPU autocast does **not** upcast LayerNorm (gives bf16); CUDA autocast does. Use CUDA for the answers.

## Result files

- `result_v3.jsonl` — every run. Filter clean main-table rows with
  `no_grad == False & warmup_steps == 5 & num_steps == 10`; warm-up study rows are `size == "large"`
  with `warmup_steps` in {0, 1, 2}. Rows without a `no_grad` key are pre-fix (contaminated) runs.
- Failure rows have only `size`, `mode`, `status`, `stderr` (no device_name) — known gap.
