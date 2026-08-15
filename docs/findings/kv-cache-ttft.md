# Finding: TTFT measurements are confounded by KV prefix caching

> Status: **fixed in v2 protocol** (warmup archived, cold/hot TTFT split, `prefill_nonce` task added).
> Also covers two `expression_code` evaluator false-negative fixes found during the same investigation.

## Summary

While re-running the benchmark on the same machine (M5 Max + ds4-server, same model ds4f-q2-0731) two days apart, `pubmedqa` TTFT jumped from **2.57s → 11.8s** (4.6×) with everything else unchanged. Root cause: engine-level **KV prefix caching** (ds4-server's disk/live KV cache; vLLM has the equivalent automatic prefix caching). The 0813 run hit a warm server (3+ prior bench rounds → cache hot); the 0815 run started from a cold server. The 2.57s was cache-assisted; **11.8s is the true prefill speed** (~625 tok/s on M5 Max q2, matching ds4's own benchmarks).

This made TTFT non-reproducible across runs and non-comparable across engines under the old protocol, and it silently affected the overall score (TTFT has 10% weight — enough to flip rankings that are decided by <1 point).

## Evidence

**Same machine, same script, same prompt, 5 runs each:**

| Run | Server state | pubmedqa TTFT ×5 | Interpretation |
|---|---|---|---|
| 0813 | PID started 0812, 3+ bench rounds already run | `[2.57, 2.55, 2.57, 2.56, 2.59]` | disk KV cache hot |
| 0815 | cold start that morning | `[11.81, 11.83, 34.09, 11.81, 11.85]` | full prefill every time |
| manual probe after 0815 bench | warm from bench | `[11.7, 2.5, 2.5]` | 1st miss, then hit |

**Existing repo data also shows the signature** (previously unexplained):

- `dgx-spark (pre-upgrade)`: pubmedqa runs `[1.19, 41.99]` — 35× spread between two identical requests. The 1.19s is almost certainly a prefix-cache hit from the (unrecorded) warmup request; the mean of 21.59s mixes one hit with one full prefill.
- `dgx-spark (post-upgrade)`: `[0.76] × 5` — perfectly stable, but 7372 tok / 0.76s ≈ 9700 tok/s prefill. Is that real NVFP4 compute or vLLM prefix caching? **Cannot tell from the JSON**, because warmup metrics were never saved.

## Why this matters for fairness

1. The protocol (1 warmup + 5 runs) is formally symmetric but does not control cache state. Which engine hits its own cache on repeat requests, and whether the server had prior history, is accidental.
2. TTFT conflates two different, both-legitimate measurements:
   - **cold TTFT** — first time you feed new data (batch analysis of new papers)
   - **hot TTFT** — iterative refinement (agents re-sending near-identical context)
   
   Averaging them produces a number that is neither.
3. README's own key takeaway says TTFT deserves *more* weight for interactive research use — but TTFT is precisely the most cache-polluted metric.
4. Ranking impact: dgx-post (73.5) vs m5-fixed (74.0) differ by 0.5. Cache state alone shifts the TTFT sub-score by ~4 points (≈0.4 overall) — same order as the winning margin.

Caching itself is not cheating — hot TTFT is a real benefit for iterative workloads. The problem is the benchmark doesn't treat cold/hot as a controlled variable.

## Protocol changes (v2, implemented)

**First v2-protocol run (M5 Max, `bench_m5max-ds4-0731-v2_20260815_130800.json`) caught the effect red-handed:**

| task | cold | hot | cache_speedup |
|------|------|-----|---------------|
| expression_matrix | 1.75s | 0.05s | **×32.9** |
| protein_function | 2.46s | 0.05s | **×45.6** |
| pubmedqa | 2.64s | 2.50s | ×1.1 (server already warm) |
| prefill_nonce (5 runs) | 11.6–12.0s, CV<2% | n/a | pure prefill, no cache |

The changes:

1. **Record warmup metrics in the JSON** (`tasks.<name>.warmup`) — it is the only natural cold sample and was previously printed then discarded.
2. **Split TTFT reporting**: `cold_ttft_s` (warmup) and `hot_ttft_s` (one extra probe immediately after the 5 runs, cache guaranteed hot), plus `cache_speedup_ttft = cold/hot` as a hit-rate signal (flagged in the summary table when ≥2×).
3. **New `prefill_nonce` task** (10th task): reuses the pubmedqa 20-abstract body (~33K chars) with a **unique deterministic nonce per run** prepended (`Random(9582019 + run_idx)` → cross-machine identical nonces). Every request is a guaranteed cache miss, so its TTFT is pure prefill compute. `max_tokens=32` (decode kept minimal). Excluded from throughput/TTFT medians in scoring; reported as `prefill_tok_per_s = prompt_tokens / ttft` — the fair cross-engine prefill number.
4. Score unchanged (quality 50 / tput 30 / ttft 10 / stab 10) — the mixed-load TTFT keeps its meaning ("iterative UX"), and the clean number lives alongside it.

## Also fixed along the way (evaluator false negatives on expression_code)

Two regex bugs in `evaluate_expression_code` penalized correct code:

1. `read_csv\s*\(.{0,100}compression` — `.` doesn't match newlines, so any model that writes `read_csv(` with parameters on following lines (with `compression="gzip"` correctly specified!) fails the check. Fixed with `re.DOTALL` + window 200.
2. PCA check only accepted `sc.tl.pca` (legacy API); `sc.pp.pca` is the current recommended scanpy API and equally correct. Now both accepted.

**Impact on existing results**: M3 Max and M5 Max runs were under-scored on this task (both models wrote correct code); DGX runs happened to pass (different quantization → different formatting style → single-line calls + legacy API, which matched the regexes). Replayed with the fix: both M5 runs are 12/12 on all samples; M3 Max 9/12 → 11/12 (`hv_genes` remains a genuine miss). This is itself an interesting datapoint: **evaluators that are sensitive to formatting style can systematically favor one engine's sampling quirks over another's.**

## Reproduction

```bash
# cold server (restart ds4-server) → run bench → pubmedqa TTFT ≈ 11.8s, cache_speedup ≈ 1
# repeat bench immediately          → pubmedqa TTFT ≈ 2.5s,  cache_speedup ≥ 4
LLM_API_URL=http://127.0.0.1:8000/v1 LLM_MODEL=deepseek-v4-flash \
LLM_LABEL=m5max-ds4-0731-v2 LLM_CONTEXT=256000 RUNS=5 python test_dgx.py
```

Suggested follow-ups for other engines:
- DGX side: export vLLM prefix-cache hit metrics per request to confirm whether 0.76s is compute or cache.
- Document server-state requirements in README before cross-machine comparison (cold restart vs pre-warmed).
