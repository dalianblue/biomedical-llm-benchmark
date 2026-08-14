# Local LLM Benchmark for Biomedical Research / 本地 LLM 生物医学基准测试

> **Languages / 语言:** [中文](#中文版) · [English](#english-version)

A reproducible benchmark to evaluate **local LLMs on real biomedical tasks**, so
we can pick the best hardware + engine combo for biology / medicine work.
Three machines, two chip architectures, two inference engines — all running the
**same model (DeepSeek V4 Flash 0731)**.

一套可复现的基准测试，在**真实生物医学任务**上评估本地 LLM，帮我们为科研工作挑出最合适的
"硬件 + 引擎"组合。三台机器、两种芯片架构、两套推理引擎——跑的是**同一个模型（DeepSeek V4 Flash 0731）**。

- 📄 **Full report / 完整报告:** [`results/biomedical_benchmark_report_v3.html`](results/biomedical_benchmark_report_v3.html)
  (open locally, or [preview on HTMLPreview](https://htmlpreview.github.io/?https://github.com/dalianblue/biomedical-llm-benchmark/main/results/biomedical_benchmark_report_v3.html))
- 🧪 9 biomedical tasks / 9 个生物医学任务 · 4 result runs / 4 份跑分结果
- 🔒 Frozen inputs (`test_data/`, seed=42) — no network at runtime / 输入冻结，运行时无需联网

---

## 中文版

三台参与对比的机器（都跑同一份 DeepSeek V4 Flash 0731 权重）：

- **mac-m3max**：Apple M3 Max / 128 GB，`ds4-server` —— 老 Mac，做基线
- **mac-m5max**：Apple M5 Max / 128 GB，`ds4-server` —— 吞吐王
- **dgx-spark**：NVIDIA DGX Spark (GB10) / 128 GB，`vLLM` —— 跑了两次，升级 vLLM recipe 前后各一次（见 [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)）

本测试同时衡量**性能**（TTFT、吞吐）和**质量**（对照 ground truth 的准确率），覆盖 9 个任务，
最终生成一张横向对比图。

### 结果速览（4 次跑分）

完整报告见 [`results/biomedical_benchmark_report_v3.html`](results/biomedical_benchmark_report_v3.html)。
综合分 = `质量×50% + 吞吐×30% + TTFT×10% + 稳定性×10%`。

| 跑次 | 机器 | 引擎 | runs/task | 综合分 | 质量 | 吞吐 | TTFT |
|------|------|------|-----------|--------|------|------|------|
| 1 | M3 Max | ds4-server | 5 | **54.9** | 70.6 | 24.6 | 22.3 |
| 2 | M5 Max | ds4-server | 5 | **72.9** | 71.7 | 70.4 | 59.7 |
| 3 | DGX Spark（升级前） | vLLM | 2 | **67.1** | 75.8 | 45.3 | 55.5 |
| 4 | DGX Spark（升级后） | vLLM | 5 | **73.5** | 76.9 | 61.8 | 65.7 |

### 核心结论

- **模型决定质量上限，硬件只决定你多快够到这个上限。** 四份质量分挤在 70.6–76.9（极差 ≤6），综合分却从 54.9 冲到 73.5（差 ≈19）。差距全在速度。想提升答案质量？换模型，别换机器。
- **引擎×芯片的调优，可能比芯片代际本身还重要。** M3 Max → M5 Max（同引擎、升一代）：吞吐 ×3.9。DGX 升级前后（同机器、换 vLLM 配置）：`pubmedqa` 吞吐 ×14、TTFT ×28。
- **长上下文 ≠ 长上下文跑得快。** 能装下 256K，不等于跑得动。M3 Max 在 8K 输入上 TTFT 飙到 52s；DGX（升级前）21s。买之前看真实长上下文 benchmark，别只看宣传的 ctx size。
- **DGX 升级前后的跃升来自软件，不是硬件。** Mia 升级后的 recipe 把 GB10 专用路径全接上了：Anemll 的 GB10 vLLM 0.25 移植版、DSpark 投机解码（`MTP_NUM_TOKENS=5`）、NVFP4 DS-MLA KV cache（修好长 prefill 的就是它——`pubmedqa` TTFT 21.59s → 0.76s）、外加 2 节点 TP=2。第 2 次跑分才是 DGX 的可信数字（5 runs 对 2 runs，warmup 更充分）。
- **有两个任务是诚实的设计翻车。** `mutation_call`（所有机器 P=0 R=0——LLM 做不了字符级对齐，用 BLAST/BWA）和 `expression_matrix`（1–2/4——LLM 做不了精确算术，让它写 pandas）。其余 5 个任务全场满分。

### 快速开始（单机，约 15–30 分钟）

```bash
cd Local_LLM_test

# 1. 建虚拟环境（uv 最快，pip 也行）
uv venv .venv
source .venv/bin/activate

# 2. 装依赖
uv pip install requests matplotlib
# 或：pip install requests matplotlib

# 3. 确保本地 LLM 服务已起且可达，然后：
# LLM_LABEL 每台机器起一个唯一标签，如 mac-m5max-ds4-0731 / dgx-spark-vllm / linux-a100
# LLM_CONTEXT 填你引擎实际配置的上下文上限，如 M3/M5=256000，DGX(vLLM)=1048576
LLM_API_URL=http://127.0.0.1:8000/v1  \
LLM_MODEL=deepseek-v4-flash           \
LLM_LABEL=<机器-引擎-模型>             \
LLM_CONTEXT=<上下文上限>               \
RUNS=5                                \
python test_dgx.py
```

> `RUNS=5` 是脚本默认值，也是本仓库 4 份跑分用的设置（首次额外 1 次预热不计入）。
> 只想冒烟测试可设 `RUNS=1`，约 5 分钟跑完。
> `LLM_LABEL` 会写进结果文件名，所以每台机器要用不同的值——横评时就是靠它区分谁的跑分。

输出（文件名里的 `<机器-引擎-模型>` 就是上面设的 `LLM_LABEL`）：
```
results/bench_<机器-引擎-模型>_YYYYMMDD_HHMMSS.json   # 完整数据 + 环境快照
results/bench_<机器-引擎-模型>_YYYYMMDD_HHMMSS.png    # 一眼速览图
```

### 三机对比流程

1. **每台机器上**：各跑一次 `test_dgx.py`（见上文）。给每台机器起唯一的 `LLM_LABEL`（`mac-m5max`、`dgx-spark`、`linux-a100` …）。
2. **收集**三份 `bench_*.json` 到同一个 `results/` 目录。
3. **生成对比图**：

   ```bash
   python compare_bench.py results/bench_mac*.json results/bench_dgx*.json results/bench_linux*.json
   ```

   输出 `compare_YYYYMMDD_HHMMSS.png`，一张 2×2 面板：每任务 TTFT（越低越好）、每任务 tokens/s（越高越好）、每任务质量分（0–1，对照 ground truth）、环境对比表。

   可选参数：
   ```bash
   python compare_bench.py results/*.json --metric ttft       # 只画 TTFT
   python compare_bench.py results/*.json --metric quality    # 只画质量
   python compare_bench.py results/*.json -o my_compare.png   # 自定义输出名
   ```

### 目录结构

```
Local_LLM_test/
├── README.md                          # 本文件（中英双语）
├── test_dgx.py                        # ⚠ 主 HTTP 基准脚本，兼容任何
│                                      #   OpenAI 服务（ds4-server / vLLM /
│                                      #   llama.cpp / Ollama …）——名字虽叫
│                                      #   dgx，但并非 DGX 专用
├── test_mac.py                        # CLI 变体，subprocess 调 ds4 二进制，
│                                      #   无 HTTP，只跑 1 个任务
├── compare_bench.py                   # 多机对比图生成器
├── test_data/                         # 所有测试输入（冻结，~3.4 MB）
│   ├── README.md                      # 数据来源与出处
│   ├── mutation/                      # BRCA1 参考 + 20 突变序列
│   ├── expression/                    # GSE100866 单细胞 top500 基因
│   ├── protein/                       # BRCA1 p53 结合域 + UniProt 特征
│   └── benchmark/                     # PubMedQA + MedMCQA 题库（40 题）
└── results/                           # 运行后生成（json + png）
                                      #   + biomedical_benchmark_report_v3.html
                                      #     （4 次跑分的完整报告，浏览器打开）
```

运行时无需任何外部下载——所有任务输入都在 `test_data/` 里，整个包 < 5 MB。

### 环境要求

- **Python**：3.10+（开发用 3.11；3.14 也能跑，但 `uv venv` 会自动选 3.11）
- **依赖**：`requests`、`matplotlib`（仅生成图用）
- **可达的 LLM 端点**：任何 OpenAI 兼容的 `/v1/chat/completions` 服务
  - `ds4-server`、`vLLM`、`llama.cpp server`、`Ollama`、LM Studio 等
  - 需支持 `stream: true` 和 `stream_options.include_usage`（以上都支持）
- **可选**：`uv` 快速建虚拟环境（macOS 上 `brew install uv`）

基准客户端不需要 GPU/CUDA/PyTorch——它只发 HTTP 请求。

### 9 个任务

每个任务瞄准一种能力维度。最右列带**质量评估**的有 ground truth 自动评估器。

| # | 任务 | 输入大小 | 测什么 | 质量评估 |
|---|------|----------|--------|----------|
| 1 | `mutation_call` | ~15 KB（BRCA1 7088 nt ×2） | 长上下文 + 核苷酸级推理 | 位置 P/R vs 20 个已知位点 |
| 2 | `expression_genes` | ~1.7 KB（top30 基因） | 单细胞领域知识 | — |
| 3 | `expression_matrix` | ~1.6 KB（20×10 子矩阵） | 表格数值推理 | 4 项子检查 vs 计算真值 |
| 4 | `expression_code` | ~1.4 KB（数据集描述） | Scanpy 代码生成 | 12 个代码关键点检查 |
| 5 | `protein_function` | ~3.3 KB（p53 结构域 + 二级结构） | 结构生物学知识 | 4 个子问题关键词匹配 |
| 6 | `pubmedqa` | ~33 KB（20 篇摘要） | 文献理解 + yes/no 推理 | 准确率 vs 标注答案 |
| 7 | `medmcqa` | ~4 KB（20 道选择题） | 医学领域知识（4 选 1） | 准确率 vs 标注答案 |
| 8 | `json_output` | ~625 B（5 个已知变异） | 结构化输出可靠性 | 6 项 schema/正确性检查 |
| 9 | `long_generation` | ~1 KB（综述 prompt） | **持续 decode 吞吐**——就 PARP/BRCA 写 600–800 字综述 | 长度 + 4 章节 + 主题覆盖 |

**关于任务 1**：这是 `test_dgx.py`（HTTP）和 `test_mac.py`（CLI）唯一共享的任务，prompt **逐字节相同**（脚本验证过），所以 HTTP 路径和 CLI 路径在该任务上可直接对比（含 HTTP 开销）。

### 配置（环境变量）

都在 `test_dgx.py` 里：

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `LLM_API_URL` | `http://localhost:8000/v1` | OpenAI 兼容 API 根地址 |
| `LLM_MODEL` | `deepseek-v4-flash` | 请求里带的模型 id |
| `LLM_LABEL` | `unknown` | 跑分标签，写进结果文件名 |
| `LLM_CONTEXT` | `unknown` | 引擎配置的上下文大小（用于环境快照） |
| `RUNS` | `5` | 每任务重复次数（首次为 warmup，不计入） |
| `OUTPUT_DIR` | `./results` | JSON + PNG 写到哪 |
| `PLOT` | `1` | 设 `0` 跳过出图（无头/CI 环境） |

### 综合分（0–100）

`test_dgx.py`（单机）和 `compare_bench.py`（多机）都会算一个 **0–100 的综合分**，
用四个子分加权：

| 子分 | 权重 | 衡量什么 | 锚点 |
|------|------|----------|------|
| **质量** | 50% | 所有任务质量评估器的均值（准确率、JSON 正确性等） | 0–100（原始 0–1，等比放大） |
| **吞吐** | 30% | 各任务 tokens/s 中位数，对数归一化 | 5 tok/s=30，30=70，60=90，100=100 |
| **TTFT** | 10% | 各任务 TTFT 中位数，对数归一化（越低越好） | 0.1s=95，1s=80，5s=50，15s=20 |
| **稳定性** | 10% | `long_generation` 跨 run 运行时间的 CV（变异系数） | CV<1%=100，CV=10%=0 |

每个子分**按机器独立计算**（不做跨机归一化），所以单机分数本身就有意义。缺失数据（比如只有 1 run 算不出稳定性）按 0 计——这会故意惩罚不完整的跑分。

**质量占比最大**，因为生物医学里答错比答慢代价高得多；**吞吐其次**，因为它卡的是真实科研节奏；**TTFT 和稳定性**权重小，因为它们影响体验、不影响正确性。

分数出现在：终端（`test_dgx.py` 结尾打印、`compare_bench.py` 排名表）、单机 PNG 顶部、对比 PNG 顶部、JSON 的 `overall_score` 字段。配色：**绿 ≥70（好）/ 橙 40–69（一般）/ 红 <40（弱）**。

### 脚本命名：`test_dgx.py` vs `test_mac.py`

**名字有误导**——按开发顺序起的，真正区别是 **HTTP 对 CLI**，不是 **DGX 对 Mac**：

| 脚本 | 真实角色 | 何时用 |
|------|----------|--------|
| `test_dgx.py` | 通用 HTTP 基准，9 个任务全跑 | **只要有任何 HTTP 服务在跑就用它**（ds4-server、vLLM、llama.cpp、Ollama、LM Studio、SGLang、TGI……）——Mac/Linux/DGX 通吃 |
| `test_mac.py` | CLI subprocess 包装，只跑任务 1（`mutation_call`） | 只有 `ds4` 二进制、没起服务，想测裸 CLI 速度（无 HTTP 开销）时 |

`test_mac.py` 从 `test_dgx.py` 导入全部任务/评估器/计分逻辑，只换调用方式。**任何完整跑分都用 `test_dgx.py`**，不分硬件。名字是历史遗留，理想情况应改名 `bench_http.py` / `bench_ds4_cli.py`，但改名会破坏已有结果引用。

`test_mac.py` 的注意事项：只跑任务 1；运行前需改硬编码路径 `DS4_PATH`、`MODEL_PATH`；prompt 与 `test_dgx.py` 逐字节相同，可直接对比；用 `subprocess.run`（非流式），只报总时间——无 TTFT、无质量评估。

### 数据来源

全部在 `test_data/` 下，出处详见 `test_data/README.md`。

| 数据集 | 来源 | 协议/引用 |
|--------|------|-----------|
| BRCA1 NM_007294.4 + 20 突变 | NCBI Nucleotide + 合成模拟（仅替换，seed=42） | NCBI public |
| GSE100866 CD8 CITE-seq | GEO（Stoeckius et al. 2017） | GEO public, CC BY 4.0 |
| BRCA1 P38398 + p53 结构域 | UniProt + Zhang et al. 1998（PMID 9582019） | UniProt CC BY 4.0 |
| PubMedQA（抽 20 题） | `qiaojin/PubMedQA` `pqa_labeled`，seed=42 | Jin et al. 2019 |
| MedMCQA（抽 20 题） | `openlifescienceai/medmcqa` `validation`，seed=42 | Pal et al. 2012 |

**采样是确定性的**（seed=42）——每台机器拿到同样的 40 题。子集已 check-in 到 `test_data/benchmark/*.json`，**跑分时无需联网**。

### 已知限制

- **温度固定为 0**（确定性），不测跨温度稳定性。需要的话自己加 `temperature in [0, 0.3, 0.7]` 循环。
- **仅单轮**。真实科研是多轮迭代对话，本测试只测单轮质量。
- **PubMedQA / MedMCQA 各 20 题**。够给机器排序，不足以做可发表对比。需要更窄误差棒就改 `test_data/benchmark/*.json` 扩容。
- **不测工具调用 / agent loop**。测了 JSON 输出，但没测函数调用语义。
- **成本/能耗未归一化**。`tok/s` 是裸吞吐；要比 `tok/s/$` 自己在环境快照里填硬件成本。
- **盲区一：量化精度（Q2 / FP4 / FP8）分离不出来。** 每台机器跑各自引擎的默认量化（DGX=NVFP4，Mac=GGUF 低位量化），差异被硬件差异裹住，没法单独归因；而且任务不够"挑剔精度"——失败的是能力问题（FP16 也救不回），通过的答案不变（精度损失埋在噪声地板下），加上 20 题样本太小，量化差统计上验不出。想真测：同机同引擎只换量化 + perplexity/精确 token 召回类任务 + 样本 ≥500。详见报告 §07。
- **盲区二：长上下文容量优势没机会出场。** 最长输入才 ~8K tokens（pubmedqa），用到 256K 的 3%、1M 的 0.8%——DGX 的 1M 上下文在这套单轮题里完全显不出来。容量是"装得下"不是"跑得快"，1M 优势只在喂 >256K 时才兑现（加一道 >256K 的任务——整基因组 / 大单细胞矩阵 / 论文全文 in-context——M3/M5 会 OOM，差距立刻变断崖）。
- **盲区二的另一半：1M 真正发光的地方是多轮长 agent。** 这套测试全是单轮单题，测不到；但在 agent 式长链路里（多轮工具调用、检索累加、草稿在上下文里反复改），KV cache 逐轮增长，256K 几十轮就撞墙，1M 给 4 倍"不打理内存也能一直跑"的余量。注意这是<strong>"省心余量"</strong>，不是"能力解锁"——M5 Max 同样能跑，只是要手动管理上下文。
- **顺手破个误会："256K 写不了万字综述"。** 万字综述是**单次 max_tokens 输出上限**问题，不归上下文窗口管，<strong>两机一样卡</strong>（DeepSeek V4 Flash 一般 ~8K 输出 ≈ 5000–6000 中文字）。256K 的输入/累积空间对一篇万字综述的素材绰绰有余。M5 Max 写法：先大纲后分节、map-reduce、多轮迭代 + 草稿落盘、续写模式——分段生成 + 上下文累积，所有机器通用。详见报告 §07。

### 复现对比

要让结果可比：用**同一个** `test_data/`；用**同一个** `RUNS`（脚本默认 5，也是本仓库跑分用的值；只做冒烟测试可用 1）；在**安静的机器**上跑（关浏览器、无其它 GPU 任务）；让 warmup 跑完（别把第一次当卡死杀掉——冷启动 GPU 上首条 prompt 可能要 30–60s 做内核编译）；分享 **JSON**（不只是 PNG），它有完整每轮样本，同事能重画或自分析。

---

## English version

Three machines in this comparison (all running the same DeepSeek V4 Flash 0731 weights):

- **mac-m3max**: Apple M3 Max / 128 GB, `ds4-server` — the older Mac, baseline
- **mac-m5max**: Apple M5 Max / 128 GB, `ds4-server` — throughput winner
- **dgx-spark**: NVIDIA DGX Spark (GB10) / 128 GB, `vLLM` — ran twice, before & after a vLLM recipe upgrade (see [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark))

The benchmark measures **performance** (TTFT, throughput) AND **quality**
(accuracy against ground truth) across 9 tasks, then produces a single
side-by-side chart for cross-machine comparison.

### Results at a glance (4 runs)

Full write-up: [`results/biomedical_benchmark_report_v3.html`](results/biomedical_benchmark_report_v3.html).
Overall score = `quality×50% + throughput×30% + ttft×10% + stability×10%`.

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

### Quick start (single machine, ~15–30 min)

```bash
cd Local_LLM_test

# 1. Create venv (uv is fastest; pip works too)
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install requests matplotlib
# or: pip install requests matplotlib

# 3. Make sure your local LLM server is running and reachable, then:
# LLM_LABEL: one unique label per machine, e.g. mac-m5max-ds4-0731 / dgx-spark-vllm / linux-a100
# LLM_CONTEXT: your engine's actual context limit, e.g. M3/M5=256000, DGX (vLLM)=1048576
LLM_API_URL=http://127.0.0.1:8000/v1  \
LLM_MODEL=deepseek-v4-flash           \
LLM_LABEL=<machine-engine-model>       \
LLM_CONTEXT=<context-limit>            \
RUNS=5                                \
python test_dgx.py
```

> `RUNS=5` is the script default and what the 4 runs in this repo used (plus 1
> warmup, not counted). For a smoke test set `RUNS=1` — finishes in ~5 min.
> `LLM_LABEL` goes into the result filename, so give each machine a different
> value — that's how runs are told apart in a cross-machine comparison.

Output (the `<machine-engine-model>` in the filename is the `LLM_LABEL` you set):
```
results/bench_<machine-engine-model>_YYYYMMDD_HHMMSS.json   # full numbers + env snapshot
results/bench_<machine-engine-model>_YYYYMMDD_HHMMSS.png    # one-glance summary chart
```

### Three-machine comparison workflow

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

### Directory layout

```
Local_LLM_test/
├── README.md                          # this file (bilingual CN/EN)
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
                                      #   + biomedical_benchmark_report_v3.html
                                      #     (the 4-run write-up; open in a browser)
```

No external downloads needed at runtime — all task inputs are bundled in
`test_data/`. The full package is < 5 MB.

### Environment requirements

- **Python**: 3.10+ (developed on 3.11; 3.14 works but `uv venv` will pick 3.11
  automatically if available)
- **Dependencies**: `requests`, `matplotlib` (only for chart generation)
- **Reachable LLM endpoint**: any OpenAI-compatible `/v1/chat/completions` server
  - `ds4-server`, `vLLM`, `llama.cpp server`, `Ollama`, LM Studio, etc.
  - Must support `stream: true` and `stream_options.include_usage` (all of the
    above do)
- **Optional**: `uv` for fast venv setup (`brew install uv` on macOS)

No GPU/CUDA/PyTorch needed on the benchmark client — it just sends HTTP.

### The 9 tasks

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

### Configuration (environment variables)

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

### Overall score (0-100)

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

### Script naming: `test_dgx.py` vs `test_mac.py`

**The names are misleading** — they were chosen by development order, not by
purpose. The real distinction is **HTTP vs CLI**, not **DGX vs Mac**:

| Script | Real role | When to use |
|--------|-----------|-------------|
| `test_dgx.py` | Universal HTTP benchmark, all 9 tasks | **Always, if you have any HTTP server running** (ds4-server, vLLM, llama.cpp, Ollama, LM Studio, SGLang, TGI, …) — works on Mac, Linux, DGX, anything |
| `test_mac.py` | CLI subprocess wrapper, only task 1 (`mutation_call`) | Only when you have the `ds4` binary but no server, and want to measure raw CLI speed without HTTP overhead |

`test_mac.py` imports all task/evaluator/score logic from `test_dgx.py` —
it's a thin wrapper that only swaps the LLM calling method. **For any full
benchmark run, use `test_dgx.py`**, regardless of hardware.

`test_mac.py` caveats: runs only task 1; edit hard-coded `DS4_PATH` / `MODEL_PATH` first; prompt byte-identical to `test_dgx.py` (directly comparable); uses `subprocess.run` (not streaming), so only total time — no TTFT, no quality eval.

### Data sources and provenance

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

### Known limitations

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
- **Blind spot 1: quantization precision (Q2 / FP4 / FP8) is not isolated.** Each
  machine runs its engine's default quantization (DGX=NVFP4, Mac=low-bit GGUF),
  so precision differences are confounded with hardware and can't be attributed
  alone. The tasks also aren't precision-discriminating: failures are capability
  failures (they'd fail at FP16 too), passes are unchanged across quants (loss
  is below the noise floor), and 20-question samples are too small to detect
  single-digit-% differences. To really measure it: same machine + engine, swap
  only quantization, use precision-sensitive tasks (perplexity, exact token
  recall), ≥500 samples. See report §07.
- **Blind spot 2: long-context capacity advantage never shows.** The longest
  input is ~8K tokens (pubmedqa) — 3% of 256K, 0.8% of 1M — so DGX's 1M context
  is invisible in this single-turn suite. Capacity is "fits" not "fast"; the 1M
  edge only materializes when you feed >256K (add a >256K task — whole genome /
  large single-cell matrix / full papers in-context — and M3/M5 OOM, gap turns
  into a cliff).
- **The other half of blind spot 2: 1M truly shines in long multi-turn agents.**
  Every task here is single-turn, so it's untested; but in long agent loops
  (multi-turn tool calls, retrieval accumulation, in-context draft edits), KV
  cache grows each turn — 256K hits the wall after a few dozen, 1M gives 4× the
  "run without babysitting memory" headroom. Note this is a **convenience
  headroom**, not a capability unlock — M5 Max can do the same work, just with
  manual context management.
- **Busting a myth: "256K can't write a 10k-word review."** A long review is a
  **per-generation max_tokens output** problem, not a context-window problem,
  and the cap is **identical on both machines** (DeepSeek V4 Flash ~8K output ≈
  5000–6000 Chinese chars). 256K of input/accumulation is plenty for the source
  material. M5 Max recipes: outline-then-section, map-reduce, multi-turn with
  draft on disk, continuation mode — segmented generation + in-context
  accumulation works on any machine. See report §07.

### Reproducing the comparison

To make results comparable across machines:

1. Use the **same** `test_data/` directory (this exact folder).
2. Use the **same** `RUNS` (the script default is 5, and what this repo's runs used; use 1 for a smoke test).
3. Run on a **quiet machine** — close browsers, no other GPU jobs.
4. Let the warmup finish (don't kill the first run thinking it's hung — first
   prompt can take 30-60 s on a cold GPU due to kernel compilation).
5. Share the **JSON** (not just the PNG) — it has the full per-run samples,
   so colleagues can re-plot or do their own analysis.

---

## License / 许可

Code released under the MIT License. Benchmark data under each upstream source's
respective license (see `test_data/README.md`).

代码遵循 MIT 协议。测试数据遵循各上游数据源的原协议（见 `test_data/README.md`）。
