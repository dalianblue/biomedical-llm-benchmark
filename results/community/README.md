# Community benchmark submissions

社区贡献的 `bench_*.json` 放这里。详见仓库根目录 README 的「贡献指南 / Contributing」。

Community-contributed `bench_*.json` results live here. See the
"Contributing" section in the repo root README.

## 命名约定 / Naming

`bench_<your-label>_<YYYYMMDD_HHMMSS>.json`

`<your-label>` 要唯一、说清硬件+引擎（+量化），例如：

- `amd-7900xtx-vllm-fp8`
- `intel-a770-ollama-q4_km`
- `rtx4090-vllm-fp8`
- `a100-80gb-sglang`

## 提交清单 / Checklist

- [ ] 用仓库**原版** `test_data/`（seed=42，不要改题）
- [ ] `RUNS=5`，机器空闲，warmup 跑完
- [ ] `LLM_LABEL` 唯一，且和文件名里的 label 一致
- [ ] （建议）同目录放一份 `<your-label>.md`：芯片/显存、引擎+版本、量化、模型+版本、ctx、驱动/CUDA/ROCm 版本、跑分日期、是否有非标准配置

## 规矩 / Rules

不要为刷分改 prompt / `max_tokens` / task 顺序——脚本怎么来就怎么跑。非标准配置（降 RUNS、跳 warmup、机器有其它负载）可以提交，但请在附带说明里写清楚。详见根 README「贡献指南」。

Don't tune scores (no prompt / `max_tokens` / task-order changes). Run the
script as-is. Non-standard configs (lower RUNS, skipped warmup, shared
machine) are still accepted — just flag them in the accompanying note.
