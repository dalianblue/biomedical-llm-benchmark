# Local LLM Benchmark for Biomedical Research

A reproducible benchmark to evaluate **local LLMs on real biomedical tasks**, so
we can pick the best hardware + engine combo for our group's biology / medicine
work.

Three machines in this comparison (all running the same DeepSeek V4 Flash 0731 weights):
- **mac-m3max**: Apple M3 Max / 128 GB, `ds4-server` — the older Mac, baseline
- **mac-m5max**: Apple M5 Max / 128 GB, `ds4-server` — throughput winner
- **dgx-spark**: NVIDIA DGX Spark (GB10) / 128 GB, `vLLM` — ran twice, before & after a vLLM recipe upgrade (see [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark))

The benchmark measures **performance** (TTFT, throughput) AND **quality**
(accuracy against ground truth) across 9 tasks, then produces a single
side-by-side chart for cross-machine comparison.

---

## Results at a glance (4 runs)

Full write-up: `results/biomedical_benchmark_report_v2.html`. Overall score = `quality×50% + throughput×30% + ttft×10% + stability×10%`.

| Run | Machine | Engine | runs/task | Overall | Quality | Throughput | TTFT |
|-----|---------|--------|-----------|---------|---------|------------|------|
| 1 | M3 Max | ds4-server | 5 | **54.9** | 70.6 | 24.6 | 22.3 |
| 2 | M5 Max | ds4-server | 5 | **72.9** | 71.7 | 70.4 | 59.7 |
| 3 | DGX Spark (before upgrade) | vLLM | 2 | **67.1** | 75.8 | 45.3 | 55.5 |
| 4 | DGX Spark (after upgrade) | vLLM | 5 | **73.5** | 76.9 | 61.8 | 65.7 |

### Key takeaways

- **Model sets the quality ceiling, hardware only sets how fast you reach it.** All four quality scores cluster in 70.6–76.9 (≤6 spread), but overall scores span 54.9–73.5 (≈19 spread). The entire gap is speed. Want better answers? Change the model, not the machine.
- **Engine×chip tuning can matter more than the chip generation itself.** M3 Max → M5 Max (same engine, one chip gen newer): throughput ×3.9. DGX before → after (same machine, different vLLM config): `pubmedqa` throughput ×14, TTFT ×28.
- **Long context ≠ fast long context.** 256K context fits, doesn't mean it runs fast. M3 Max TTFT on an 8K-token input balloons to 52s; DGX (pre-upgrade) to 21s. Always check a real long-context benchmark, not the advertised ctx size.
- **The DGX before→after jump came from software, not hardware.** Mia's upgraded recipe wired up the GB10-specific paths: Anemll's GB10 vLLM 0.25 port, DSpark speculative decoding (`MTP_NUM_TOKENS=5`), NVFP4 DS-MLA KV cache (this is what fixed long prefill — `pubmedqa` TTFT 21.59s → 0.76s), and 2-node TP=2. The run-2 score is the trustworthy DGX number (5 runs vs 2, proper warmup).
- **Two tasks were honest design failures.** `mutation_call` (P=0 R=0 on all machines — LLMs can't do character-level alignment, use BLAST/BWA) and `expression_matrix` (1–2/4 — LLMs can't do exact arithmetic, have them write pandas instead). Five other tasks scored full marks everywhere.

---

## Quick start (single machine, ~5 min)

```bash
cd Local_LLM_test

# 1. Create venv (uv is fastest; pip works too)
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install requests matplotlib
# or: pip install requests matplotlib

# 3. Make sure your local LLM server is running and reachable, then:
LLM_API_URL=http://127.0.0.1:8000/v1  \
LLM_MODEL=deepseek-v4-flash           \
LLM_LABEL=mac-m5max-ds4-0731          \
LLM_CONTEXT=256000                    \
RUNS=3                                \
python test_dgx.py
```

Output:
```
results/bench_mac-m5max-ds4-0731_YYYYMMDD_HHMMSS.json   # full numbers + env snapshot
results/bench_mac-m5max-ds4-0731_YYYYMMDD_HHMMSS.png    # one-glance summary chart
```

---

## Three-machine comparison workflow

1. **On each machine**: run `test_dgx.py` once (Quick start above). Pick a unique
   `LLM_LABEL` per machine (`mac-m5max`, `dgx-spark`, `linux-a100`, ...).
2. **Collect** all three `bench_*.json` files into one `results/` folder.
3. **Generate the comparison chart**:

   ```bash
   python compare_bench.py results/bench_mac*.json results/bench_dgx*.json results/bench_linux*.json
   ```

   Output: `compare_YYYYMMDD_HHMMSS.png` — a single 2x2 panel:
   - TTFT per task (lower is better)
   - tokens/s per task (higher is better)
   - Quality score per task (0-1, vs ground truth)
   - Environment comparison table (chip / engine / model / ctx for each machine)

   Options:
   ```bash
   python compare_bench.py results/*.json --metric ttft       # only TTFT panel
   python compare_bench.py results/*.json --metric quality    # only quality panel
   python compare_bench.py results/*.json -o my_compare.png   # custom output name
   ```

---

## Directory layout

```
Local_LLM_test/
├── README.md                          # this file
├── test_dgx.py                        # ⚠ main HTTP benchmark, works with ANY
│                                      #   OpenAI-compatible server (ds4-server,
│                                      #   vLLM, llama.cpp, Ollama, …) — NOT
│                                      #   DGX-specific despite the name
├── test_mac.py                        # CLI-only variant, calls ds4 binary via
│                                      #   subprocess (no HTTP), only 1 task
├── compare_bench.py                   # multi-machine comparison chart generator
├── test_data/                         # all benchmark inputs (frozen, ~3.4 MB)
│   ├── README.md                      # data provenance and sources
│   ├── mutation/                      # BRCA1 ref + 20-mutation sequence
│   ├── expression/                    # GSE100866 single-cell top500 genes
│   ├── protein/                       # BRCA1 p53-binding domain + UniProt features
│   └── benchmark/                     # PubMedQA + MedMCQA question banks (40 Qs)
└── results/                           # generated after running (json + png)
                                      #   + biomedical_benchmark_report_v2.html
                                      #     (the 4-run write-up; open in a browser)
```

No external downloads needed at runtime — all task inputs are bundled in
`test_data/`. The full package is < 5 MB.

---

## Environment requirements

- **Python**: 3.10+ (developed on 3.11; 3.14 works but `uv venv` will pick 3.11
  automatically if available)
- **Dependencies**: `requests`, `matplotlib` (only for chart generation)
- **Reachable LLM endpoint**: any OpenAI-compatible `/v1/chat/completions` server
  - `ds4-server`, `vLLM`, `llama.cpp server`, `Ollama`, LM Studio, etc.
  - Must support `stream: true` and `stream_options.include_usage` (all of the
    above do)
- **Optional**: `uv` for fast venv setup (`brew install uv` on macOS)

No GPU/CUDA/PyTorch needed on the benchmark client — it just sends HTTP.

---

## The 9 tasks

Each task targets a different capability dimension. Tasks with **Quality** in the
rightmost column have automated evaluators with ground truth.

| # | Task | Input size | Tests | Quality evaluator |
|---|------|------------|-------|-------------------|
| 1 | `mutation_call` | ~15 KB (BRCA1 7088 nt x2) | Long-context + nucleotide-level reasoning | Position P/R vs 20 known sites |
| 2 | `expression_genes` | ~1.7 KB (top30 genes) | Single-cell domain knowledge | — |
| 3 | `expression_matrix` | ~1.6 KB (20x10 sub-matrix) | Numerical reasoning on tables | 4 sub-checks vs computed truth |
| 4 | `expression_code` | ~1.4 KB (dataset description) | Scanpy code generation | 12 code key-point checks |
| 5 | `protein_function` | ~3.3 KB (p53 domain + SS) | Structural biology knowledge | 4 sub-question keyword matches |
| 6 | `pubmedqa` | ~33 KB (20 abstracts) | Literature comprehension + yes/no reasoning | Accuracy vs labeled answers |
| 7 | `medmcqa` | ~4 KB (20 MCQs) | Medical domain knowledge (4-choice) | Accuracy vs labeled answers |
| 8 | `json_output` | ~625 B (5 known variants) | Structured output reliability | 6 schema/correctness checks |
| 9 | `long_generation` | ~1 KB (review prompt) | **Sustained decode throughput** — write a 600-800 word review on PARP/BRCA | Length + 4 sections + topic coverage |

**Note on task 1**: this is the only task shared between `test_dgx.py` (HTTP)
and `test_mac.py` (CLI). The prompt is **byte-identical** (verified by script),
so the HTTP path and the CLI path can be compared directly on this task —
HTTP overhead included.

---

## Configuration (environment variables)

All in `test_dgx.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_API_URL` | `http://localhost:8000/v1` | OpenAI-compatible API root |
| `LLM_MODEL` | `deepseek-v4-flash` | Model id sent in the request |
| `LLM_LABEL` | `unknown` | Run label, goes into result filename |
| `LLM_CONTEXT` | `unknown` | Your engine's configured ctx size (for env snapshot) |
| `RUNS` | `5` | Repetitions per task (first run is warmup, not counted) |
| `OUTPUT_DIR` | `./results` | Where to write JSON + PNG |
| `PLOT` | `1` | Set to `0` to skip chart generation (headless / CI) |

---

## Overall score (0-100)

Both `test_dgx.py` (single machine) and `compare_bench.py` (multi-machine)
compute a single **overall score from 0 to 100** that captures "fitness for
biomedical research" in one number. It is computed from four sub-scores:

| Sub-score | Weight | What it measures | Anchors |
|-----------|--------|------------------|---------|
| **quality** | 50% | Average of all task quality evaluators (accuracy, JSON correctness, etc.) | 0-100 (already 0-1, scaled) |
| **throughput** | 30% | Median tokens/s across all tasks, log-normalized | 5 tok/s=30, 30 tok/s=70, 60 tok/s=90, 100 tok/s=100 |
| **ttft** | 10% | Median TTFT across all tasks, log-normalized (lower is better) | 0.1s=95, 1s=80, 5s=50, 15s=20 |
| **stability** | 10% | CV (coefficient of variation) of `long_generation` runtime across runs | CV<1%=100, CV=10%=0 |

Each sub-score is computed **independently for each machine** (no cross-machine
normalization), so a single machine's score is meaningful on its own. Missing
data (e.g. only 1 run so no stability) counts as 0 for that sub-score — this
intentionally penalizes incomplete runs.

**Quality dominates** because in biomedical research a wrong answer is much
worse than a slow one. **Throughput is second** because it gates real
research throughput. **TTFT and stability** are smaller weights because they
affect UX but not correctness.

The score shows up:
- **Terminal**: printed at the end of `test_dgx.py`, and as a ranking table in `compare_bench.py`
- **Single-machine PNG**: top of the chart, big colored number + 4 sub-scores
- **Compare PNG**: top of the chart, format `[#1] label-A 78  [#2] label-B 65  [#3] label-C 52`
- **JSON**: under `overall_score` key

Color coding: **green ≥70 (good)**, **orange 40-69 (fair)**, **red <40 (weak)**.

---

## Output files

### `bench_<label>_<timestamp>.json`

Full results, including:
- `environment`: hardware + engine snapshot (chip, memory, GPU, engine type
  auto-detected from `/v1/models`, ctx, etc.)
- `tasks.<name>`: per-task stats — TTFT/total/tok-per-s mean/min/max across
  runs, plus full sample-by-sample raw data
- `tasks.<name>.quality`: evaluator output (varies by task)

### `bench_<label>_<timestamp>.png` (single-machine summary)

A 2x2 panel:
- TTFT per task (lower better)
- tokens/s per task (higher better)
- Quality per task (0-1, with random/majority baselines as reference lines)
- Environment info table

### `compare_<timestamp>.png` (multi-machine, from `compare_bench.py`)

A 2x2 panel with grouped bar charts (one bar per machine per task) and an
environment comparison table. Suitable for sharing in chat / slides.

---

## Script naming: `test_dgx.py` vs `test_mac.py`

**The names are misleading** — they were chosen by development order, not by
purpose. The real distinction is **HTTP vs CLI**, not **DGX vs Mac**:

| Script | Real role | When to use |
|--------|-----------|-------------|
| `test_dgx.py` | Universal HTTP benchmark, all 9 tasks | **Always, if you have any HTTP server running** (ds4-server, vLLM, llama.cpp, Ollama, LM Studio, SGLang, TGI, …) — works on Mac, Linux, DGX, anything |
| `test_mac.py` | CLI subprocess wrapper, only task 1 (`mutation_call`) | Only when you have the `ds4` binary but no server, and want to measure raw CLI speed without HTTP overhead |

`test_mac.py` imports all task/evaluator/score logic from `test_dgx.py` —
it's a thin wrapper that only swaps the LLM calling method. **For any full
benchmark run, use `test_dgx.py`**, regardless of hardware.

The names are legacy and should ideally become `bench_http.py` /
`bench_ds4_cli.py`, but renaming would break existing results references.

---

## `test_mac.py` — when to use it

`test_dgx.py` covers everything via HTTP. `test_mac.py` exists for **one
specific purpose**: measuring the `ds4` **command-line binary** directly
(no HTTP layer) on the `mutation_call` task. Use it when:

- You want to isolate raw inference speed from server overhead
- The machine doesn't have `ds4-server` running, only the `ds4` binary
- You want to compare CLI invocation overhead between machines

**Caveats**:
- `test_mac.py` only runs task 1 (`mutation_call`). For all 9 tasks use
  `test_dgx.py`.
- `test_mac.py` has hard-coded paths in its config section that need editing
  before running: `DS4_PATH`, `MODEL_PATH`. Set these to your `ds4` binary and
  GGUF model file.
- The prompt is byte-identical to `test_dgx.py`'s `mutation_call` task, so the
  two are directly comparable on this task.
- `test_mac.py` uses `subprocess.run` (not streaming), so it reports only total
  time — no TTFT, no quality evaluation. Use it for raw timing only.

---

## Data sources and provenance

All under `test_data/`, with full provenance in `test_data/README.md`.

| Dataset | Source | License / citation |
|---------|--------|--------------------|
| BRCA1 NM_007294.4 + 20 mutations | NCBI Nucleotide + synthetic simulation (substitution-only, seed=42) | NCBI public |
| GSE100866 CD8 CITE-seq | GEO (Stoeckius et al. 2017) | GEO public, CC BY 4.0 |
| BRCA1 P38398 + p53 domain | UniProt + Zhang et al. 1998 (PMID 9582019) | UniProt CC BY 4.0 |
| PubMedQA (20 Qs sampled) | `qiaojin/PubMedQA` `pqa_labeled` split, seed=42 | Jin et al. 2019 |
| MedMCQA (20 Qs sampled) | `openlifescienceai/medmcqa` `validation` split, seed=42 | Pal et al. 2012 |

**Sampling is deterministic** (seed=42) — every machine gets the same 40
questions. The sampled subsets are checked into `test_data/benchmark/*.json`
so **no internet access is needed at benchmark time**.

---

## Troubleshooting

**`Connection refused` / `404 unknown endpoint`**
- Verify your LLM server is running: `curl http://127.0.0.1:8000/v1/models`
- The API URL should be the root, no trailing `/v1/chat/completions` — the
  script appends the path itself
- If running on a remote machine, make sure the port is reachable:
  `curl http://<host>:8000/v1/models`

**`stream error` / empty content with `finish_reason=length`**
- Your server is spending all `max_tokens` on reasoning. The script sends
  `reasoning_effort: none` to disable this — verify your server honors it.
  ds4-server, vLLM, and llama.cpp do. If yours doesn't, increase `max_tokens`
  in the relevant `task_*` builder function.

**First run is much slower than later runs**
- Normal. First call triggers GPU kernel JIT compilation (especially
  Metal/CUDA). The script does 1 warmup run before the counted `RUNS` runs to
  keep stats clean.

**Chart shows Chinese squares / missing glyphs**
- The chart uses English labels only, so this shouldn't happen. If it does,
  install a fallback font: `matplotlib` usually picks one up automatically.

**`compare_bench.py` shows `?` for some machines**
- That JSON was produced by an older script version without the environment
  snapshot. Re-run `test_dgx.py` on that machine with the current script.

**Python 3.14 venv missing packages**
- `uv venv` defaults to a stable Python (3.11). If you only have 3.14,
  `torch`-style deps may not have wheels yet — but this benchmark only needs
  `requests` + `matplotlib`, both of which work fine on 3.14.

---

## Known limitations

- **Temperature is fixed at 0** (deterministic). Stability across temperatures
  is not measured. Add a loop over `temperature in [0, 0.3, 0.7]` if you care.
- **Single-turn only**. Real biomedical workflows are multi-turn iterative
  conversations; this benchmark tests single-shot quality.
- **PubMedQA / MedMCQA subsets are 20 questions each**. Good for ranking
  machines by rough accuracy, too small for a publishable comparison. Scale up
  by editing `test_data/benchmark/*.json` if you need narrower error bars.
- **No tool-calling / agent loop test**. JSON output is tested, but not
  function-calling semantics.
- **Cost / energy not normalized**. `tok/s` is raw throughput; to compare
  `tok/s/$` you need to fill in hardware cost yourself in the env snapshot.

---

## Reproducing the comparison

To make results comparable across machines:

1. Use the **same** `test_data/` directory (this exact folder).
2. Use the **same** `RUNS` (default 3 is fine; 1 is OK for a quick check).
3. Run on a **quiet machine** — close browsers, no other GPU jobs.
4. Let the warmup finish (don't kill the first run thinking it's hung — first
   prompt can take 30-60 s on a cold GPU due to kernel compilation).
5. Share the **JSON** (not just the PNG) — it has the full per-run samples,
   so colleagues can re-plot or do their own analysis.
