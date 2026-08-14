#!/usr/bin/env python3
"""
本地 LLM 性能基准测试 —— 9 个生物医学任务，覆盖不同输入规模和能力维度。
默认每个任务跑 5 次（首次额外 1 次预热），单机总耗时约 5-8 分钟。

3 台本地 LLM 用法：每台机器各跑一次，通过环境变量切换 endpoint：
    LLM_API_URL=http://127.0.0.1:8000/v1  \
    LLM_MODEL=deepseek-v4-flash           \
    LLM_LABEL=mac-m5max-ds4               \
    RUNS=5 python test_dgx.py

可选环境变量：
    LLM_API_URL   OpenAI 兼容 API 根（默认 http://localhost:8000/v1）
    LLM_MODEL     模型 id（默认 deepseek-v4-flash）
    LLM_LABEL     本次运行的标签，写入结果文件名（默认 unknown）
    RUNS          每个 task 重复次数，默认 5（首次额外 1 次预热）
    OUTPUT_DIR    结果输出目录（默认 ./results）

9 个任务：
    1. mutation_call      BRCA1 ref/mut 全序列 → 找 20 个变异（有 ground truth）
    2. expression_genes   top500 基因名 + 统计 → CD8 T 细胞单细胞数据生物学解读
    3. expression_matrix  top20 基因 × 10 cells 子矩阵 → 表格数值推理
    4. expression_code    数据集描述 → 写 Scanpy 分析代码
    5. protein_function   p53 结合域 + 二级结构 → 功能推断
    6. pubmedqa           20 篇摘要 → yes/no 文献推理（长 prefill 照妖镜）
    7. medmcqa            20 道医学选择题 → 领域知识广度
    8. json_output        5 个变异 → 指定 schema 的结构化输出
    9. long_generation    PARP/BRCA 综述 → 压测持续 decode 吞吐
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests

# ============ 配置 ============
API_URL = os.environ.get("LLM_API_URL", "http://localhost:8000/v1").rstrip("/")
# ponytail: 只防尾斜杠（…/v1/ → …/v1//chat/completions 双斜杠多数服务容忍但脆）。
# 不自动补 /v1 —— 那是用户该填的，README 已说明。
MODEL_NAME = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
LLM_LABEL = os.environ.get("LLM_LABEL", "unknown")
RUNS = int(os.environ.get("RUNS", "5"))  # 3 太少看不出短任务差异；5 是统计/耗时的折中
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./results"))
# 跑完是否自动生成结果图（默认开）。设 0 关闭，便于纯 CI/无头环境。
PLOT = os.environ.get("PLOT", "1") not in ("0", "false", "no", "")

DATA_DIR = Path(__file__).parent / "test_data"

# ds4-server 默认开 thinking，会把 max_tokens 都消耗在 reasoning 上。
# 关掉思考保证 max_tokens 全部留给正式答案。vLLM/其他 OpenAI 兼容服务会忽略该字段。
COMMON_PAYLOAD = {"reasoning_effort": "none", "temperature": 0.0, "seed": 42}


# ============ 数据加载（一次性）============
def load_mutation():
    # .strip() 去掉 FASTA 末尾的 \n\n，保证 prompt 干净。
    ref = (DATA_DIR / "mutation" / "BRCA1_NM_007294.4.fasta").read_text().strip()
    mut = (DATA_DIR / "mutation" / "BRCA1_NM_007294.4_20mut.fasta").read_text().strip()
    events_log = (DATA_DIR / "mutation" / "mutation_events.log").read_text()
    # 真实 ground truth 位置
    truth_positions = [int(m.group(1)) for m in re.finditer(r"SUB\s+(\d+):", events_log)]
    return ref, mut, truth_positions


def load_expression():
    genes_tsv = (DATA_DIR / "expression" / "top500_genes.tsv").read_text()
    summary_tsv = (DATA_DIR / "expression" / "top500_summary.tsv").read_text()
    matrix_tsv = (DATA_DIR / "expression" / "top500_expression_matrix.tsv").read_text()
    return genes_tsv, summary_tsv, matrix_tsv


def load_protein():
    p53_fasta = (DATA_DIR / "protein" / "BRCA1_P38398_p53_binding_domain.fasta").read_text()
    ss_tsv = (DATA_DIR / "protein" / "BRCA1_P38398_secondary_structure.tsv").read_text()
    summary_tsv = (DATA_DIR / "protein" / "BRCA1_P38398_summary.tsv").read_text()
    return p53_fasta, ss_tsv, summary_tsv


def load_benchmark():
    """加载 PubMedQA + MedMCQA 题库（各 20 题，离线 JSON）"""
    bench_dir = DATA_DIR / "benchmark"
    pubmedqa = json.loads((bench_dir / "pubmedqa_20.json").read_text())
    medmcqa = json.loads((bench_dir / "medmcqa_20.json").read_text())
    return pubmedqa, medmcqa


# ============ 硬件/环境快照 ============
def snapshot_environment():
    """采集本机硬件 + LLM 引擎信息，写入结果 JSON 方便 3 台机器对比。
    所有字段都用 try/except 包起来，任何一项采集失败都不能让脚本挂掉。
    """
    import platform as _p
    import subprocess as _sp
    import socket as _sock

    snap = {
        "hostname":    _sock.gethostname(),
        "platform":    f"{_p.system()} {_p.release()} ({_p.machine()})",
        "python":      _p.python_version(),
        "llm_label":   LLM_LABEL,
        "llm_api_url": API_URL,
        "llm_model":   MODEL_NAME,
    }

    # CPU
    try:
        snap["cpu_cores"] = _os_cpu_count()
    except Exception:
        pass

    # macOS 专用：芯片型号、内存、GPU
    if _p.system() == "Darwin":
        try:
            chip = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=3).stdout.strip()
            if chip:
                snap["chip"] = chip
        except Exception:
            pass
        try:
            mem_bytes = int(_sp.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True, timeout=3).stdout.strip())
            snap["memory_gb"] = round(mem_bytes / (1024 ** 3), 1)
        except Exception:
            pass
        try:
            # 硬件概览，能拿到芯片型号 + 统一内存
            hw = _sp.run(["system_profiler", "SPHardwareDataType"],
                         capture_output=True, text=True, timeout=5).stdout
            for line in hw.splitlines():
                line = line.strip()
                if line.startswith("Chip:"):
                    snap["chip"] = line.split(":", 1)[1].strip()
                elif line.startswith("Total Number of Cores:"):
                    snap["cpu_cores"] = line.split(":", 1)[1].strip()
                elif line.startswith("Memory:"):
                    snap["memory_gb_str"] = line.split(":", 1)[1].strip()
        except Exception:
            pass
        snap["gpu"] = snap.get("chip", "Apple Silicon (unified)")

    # Linux 专用：CPU 型号、内存、NVIDIA GPU
    elif _p.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        snap["chip"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        snap["memory_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                        break
        except Exception:
            pass
        # NVIDIA GPU
        try:
            nvidia = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
            if nvidia:
                snap["gpu"] = nvidia.replace("\n", "; ")
        except Exception:
            pass

    # 引擎版本探测（ds4 / vLLM）：尝试 /v1/models 接口看返回字段
    try:
        r = requests.get(f"{API_URL}/models", timeout=10)
        if r.status_code == 200:
            snap["llm_endpoint_models"] = r.json().get("data", [])
            # ds4-server 返回 owned_by=ds4.c
            if snap["llm_endpoint_models"]:
                owner = snap["llm_endpoint_models"][0].get("owned_by", "")
                if owner == "ds4.c":
                    snap["llm_engine"] = "ds4-server"
                elif "vllm" in str(r.headers).lower():
                    snap["llm_engine"] = "vllm"
                else:
                    snap["llm_engine"] = f"unknown ({owner})"
    except Exception:
        pass

    # 同时记录 LLM_CONTEXT（用户可设环境变量说明这台机器的 ctx 配置）
    snap["llm_context"] = os.environ.get("LLM_CONTEXT", "unknown")

    return snap


def _os_cpu_count():
    import os as _o
    return _o.cpu_count()


# ============ 9 个任务定义 ============
def task_mutation_call(data):
    """长上下文 + 推理 + 有 ground truth。"""
    ref, mut, _ = data["mutation"]
    prompt = f"""You are a professional genomic variant annotator. Analyze the following full-length BRCA1 cDNA reference sequence and a simulated tumor sample mutated sequence. Complete the following tasks:

1. Align the mutated sequence to the reference and identify all variants (SNPs and Indels), reporting their exact genomic positions.
2. For each variant located in the coding region, deduce the corresponding change in the protein sequence (based on the BRCA1 ORF).
3. For each coding variant, predict its potential functional impact (e.g., synonymous, missense, frameshift, or nonsense). If missense, provide a SIFT/PolyPhen-like functional prediction (deleterious/tolerated) and your reasoning.
4. Format your final output as a valid VCF (Variant Call Format) file, with the following INFO fields: TYPE, PROTEIN_CHANGE, FUNCTIONAL_PREDICTION.

Reference sequence:
{ref}

Mutated sequence:
{mut}"""
    return prompt, 1500


def task_expression_genes(data):
    """短输入 + 领域知识"""
    genes_tsv, summary_tsv, _ = data["expression"]
    # 只取 top30 基因，避免输入过长
    top_lines = "\n".join(genes_tsv.splitlines()[:31])  # header + top30
    prompt = f"""Below is a summary of a single-cell RNA-seq dataset and its top-30 most highly expressed genes (ranked by total UMI count).

Dataset summary:
{summary_tsv}

Top-30 genes (gene\\ttotal_umi\\trank):
{top_lines}

As a single-cell genomics expert, briefly answer (≤400 words total):

1. **Sample identity**: What cell type does this dataset most likely represent? Name at least 3 marker genes from the list that support your call.
2. **Ribosomal / mitochondrial content**: Quantify (rough %) how much of the top-30 UMI mass comes from ribosomal proteins (RPL/RPS) vs mitochondrial genes (MT-). Is this typical for healthy CD8+ T cells?
3. **MT-RNR2 anomaly**: MT-RNR2 (mitochondrial 16S rRNA) dominates the ranking. Is high MT-RNR2 a sign of cell stress, or an artifact of the library prep? Briefly explain.
4. **One quality-control concern** with this dataset that the cell-type call alone would miss.
"""
    return prompt, 700


def task_expression_matrix(data):
    """表格数值推理"""
    _, _, matrix_tsv = data["expression"]
    # 取 top20 基因 × 前 10 cells 的子矩阵
    lines = matrix_tsv.splitlines()
    header_cells = lines[0].split("\t")
    sub_header = "\t".join(header_cells[:11])  # gene + 10 cells
    sub_rows = ["\t".join(line.split("\t")[:11]) for line in lines[1:21]]
    sub_matrix = "\n".join([sub_header] + sub_rows)

    prompt = f"""Below is a UMI count sub-matrix (top-20 genes × first 10 cells) from a CITE-seq dataset (GSE100866, CD8+ T cells). Each value is the raw UMI count for that gene in that cell.

Sub-matrix (TSV, gene\\tcell1\\tcell2\\t...\\tcell10):
{sub_matrix}

Compute and report (no Python, do it directly):

1. For each of the top-5 genes by row-sum (within this sub-matrix), report: gene, total UMI across these 10 cells, mean per cell, and max single-cell value.
2. Identify the 3 genes with the highest variance across these 10 cells, and report their variance.
3. Across all 200 (gene × cell) values, how many are zero? Report the sparsity as a fraction.
4. Which cell (column) has the highest total UMI across these 20 genes?

Show your calculations briefly, then give the final answers in a clear list.
"""
    return prompt, 800


def task_expression_code(data):
    """代码生成"""
    _, summary_tsv, _ = data["expression"]
    prompt = f"""Write a complete, runnable Python analysis script using `scanpy` and `pandas` for the following dataset.

Dataset summary:
{summary_tsv}

The full data is a CSV file at `GSE100866_CD8_merged-RNA_umi.csv.gz` with shape 11,757 genes × 1,774 cells (genes in rows, cells in columns, values are UMI counts).

Your script must:
1. Load the matrix with pandas (handle gzip), transpose so cells are rows, and build an `AnnData` object.
2. Apply standard QC: filter cells with < 500 detected genes; filter genes detected in < 3 cells; compute `pct_counts_mt` for genes prefixed `MT-`; flag cells with >20% mitochondrial counts.
3. Normalize to 10,000 counts per cell, log-transform, and identify the top 2,000 highly variable genes.
4. Run PCA → neighbors → UMAP → Leiden clustering.
5. Rank marker genes per cluster with `sc.tl.rank_genes_groups`, and save the top 10 per cluster to a TSV.
6. Save the annotated AnnData to disk.

Use clear comments. Do not include any external dataset beyond what is described. Do not call `sc.pl.*` (no plotting).
"""
    return prompt, 1500


def task_protein_function(data):
    """短输入 + 领域知识"""
    p53_fasta, ss_tsv, summary_tsv = data["protein"]
    prompt = f"""Below is information about the human BRCA1 protein (UniProt P38398) and its p53-binding domain.

Protein summary:
{summary_tsv}

p53-binding domain sequence (aa 224-500, 277 aa):
{p53_fasta}

Secondary structure features (UniProt HELIX/STRAND/TURN annotations):
{ss_tsv}

As a structural biology expert, briefly answer (≤400 words total):

1. **Paradox**: The p53-binding domain (aa 224-500) is documented as an intrinsically disordered region (IDP) with no HELIX/STRAND/TURN annotation in UniProt. How can an IDP mediate a specific, biologically critical protein-protein interaction with p53?
2. **Mechanism**: What is the most likely molecular mechanism by which BRCA1 residues 224-500 bind the C-terminal domain of p53? Name the relevant concept (e.g., coupled folding-and-binding, molecular recognition feature / MoRF, etc.) and explain in 2-3 sentences.
3. **Functional implication**: How does BRCA1 binding affect p53's function? Mention at least one experimental readout that would distinguish "BRCA1 activates p53 transcription" from "BRCA1 stabilizes p53 protein".
4. **Experimental design**: If you wanted to map the minimal BRCA1 residues required for p53 binding, which technique would you use and why? (NMR, HDX-MS, alanine scanning, cryo-EM — pick one and justify.)
"""
    return prompt, 700


# ============ 公开医学 benchmark 任务 ============
def task_pubmedqa(data):
    """PubMedQA：读摘要 → 回答 yes/no/maybe。20 题，有 ground truth。"""
    items = data["benchmark"][0]
    blocks = []
    for i, q in enumerate(items, 1):
        blocks.append(f"""## Question {i}/{len(items)}
{q['context']}

Question: {q['question']}

Answer with exactly one of: yes / no / maybe
""")
    prompt = f"""You are a biomedical research assistant. For each of the {len(items)} questions below, decide whether the answer is **yes**, **no**, or **maybe** based ONLY on the provided abstract context.

Reply with EXACTLY one line per question in this format (no other text):
```
1. <yes|no|maybe>
2. <yes|no|maybe>
...
```

""" + "\n".join(blocks) + "\nAnswer now (one line per question):"
    return prompt, 200


def task_medmcqa(data):
    """MedMCQA：医学专业多选题。20 题，有 ground truth (A/B/C/D)。"""
    items = data["benchmark"][1]
    blocks = []
    for i, q in enumerate(items, 1):
        opts = "\n".join(f"   {letter}. {text}" for letter, text in q["options"].items())
        blocks.append(f"""## Q{i}/{len(items)}
{q['question']}
{opts}
""")
    prompt = f"""You are a medical expert. For each multiple-choice question below, pick the single best answer.

Reply with EXACTLY one line per question in this format (no other text):
```
1. <A|B|C|D>
2. <A|B|C|D>
...
```

""" + "\n".join(blocks) + "\nAnswer now (one line per question):"
    return prompt, 200


# ============ JSON 结构化输出任务 ============
def task_json_output(data):
    """测 JSON 输出可靠性：把 5 个变异（已知）输出成 JSON。
    生物信息流程强依赖结构化输出（wisp-science、自动化分析等）。"""
    events = """205 T>C
245 C>G
713 A>T
913 C>G
5544 C>T"""
    prompt = f"""You are a structured data extractor. Below is a list of 5 BRCA1 variants (position, REF>ALT).

Variants:
{events}

Output a SINGLE valid JSON object (no markdown, no commentary) with this exact schema:

{{
  "gene": "BRCA1",
  "variants": [
    {{"position": <int>, "ref": "<A|C|G|T>", "alt": "<A|C|G|T>", "type": "<SNP|INDEL>"}}
  ],
  "count": <int>
}}

Rules:
- `position` is an integer (no quotes).
- `ref` and `alt` are single characters A/C/G/T.
- `type` is "SNP" (all 5 here are SNPs).
- `count` equals the number of variants in the array (5).
- Output JSON only. No prose, no code fences.
"""
    return prompt, 800


def task_long_generation(data):
    """纯 decode 速度测试：短 prompt + 长输出（~1000 tokens）。

    这个任务最贴近真实生物医学写作负载（写综述 / 病例报告 / 方法部分），
    也最能稳定测出机器的纯生成吞吐 —— prompt 短，prefill 占比小，
    总时间几乎全在 decode，方差来源最少。是横向对比硬件 decode 能力的关键指标。
    """
    prompt = """Write a focused research review (600-800 words) on the following topic for a clinical oncology audience.

Topic: **PARP inhibitor synthetic lethality in BRCA1/2-deficient cancers**

Your review must include:

1. **Mechanism** (≥150 words): Explain the molecular basis of synthetic lethality between PARP inhibition and homologous recombination repair deficiency. Mention PARP trapping, single-strand break repair, and the role of BRCA1/2 in double-strand break repair via HR.

2. **Clinical evidence** (≥150 words): Discuss at least two PARP inhibitors approved for BRCA-deficient cancers (e.g., olaparib, rucaparib, niraparib, talazoparib) and the pivotal trial(s) behind them. Cite specific outcome metrics where possible (e.g., PFS hazard ratio).

3. **Resistance mechanisms** (≥100 words): Describe at least two known mechanisms of acquired resistance (e.g., BRCA1/2 reversion mutations, replication fork protection, drug efflux).

4. **Open question** (≥100 words): Pose one current open research question and briefly outline why it matters for patient care.

Use precise biomedical terminology throughout. Structure with the four headers above. Do not include a separate summary or conclusion paragraph.
"""
    return prompt, 1500


# ============ 推理（流式，测 TTFT + 吞吐）============
def call_llm_stream(prompt, max_tokens):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **COMMON_PAYLOAD,
    }

    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    chunks = []
    finish_reason = None

    with requests.post(f"{API_URL}/chat/completions", json=payload,
                       stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.removeprefix("data: ").strip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "error" or "error" in obj:
                raise RuntimeError(f"stream error: {obj}")
            if (usage := obj.get("usage")) and usage is not None:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}) or {}
            if "content" in delta and delta["content"]:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(delta["content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    t_total = time.perf_counter() - t0
    if ttft is None:
        # 极端情况：没输出任何 content
        ttft = t_total

    output = "".join(chunks)
    return {
        "output": output,
        "ttft_s": round(ttft, 3),
        "total_s": round(t_total, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "tok_per_s": round(completion_tokens / t_total, 2) if t_total > 0 else 0,
    }


# 所有 task 函数已定义，注册到任务表
TASKS = [
    ("mutation_call",     task_mutation_call,     1500),
    ("expression_genes",  task_expression_genes,  700),
    ("expression_matrix", task_expression_matrix, 800),
    ("expression_code",   task_expression_code,   1500),
    ("protein_function",  task_protein_function,  700),
    ("pubmedqa",          task_pubmedqa,          200),
    ("medmcqa",           task_medmcqa,           200),
    ("json_output",       task_json_output,       800),
    ("long_generation",   task_long_generation,   1500),
]


# ============ 质量评估（仅 mutation_call 有 ground truth）============
def evaluate_mutation_call(output, data):
    """检查输出里出现的 1-based 位置与 ground truth 的重合度。

    Prompt 要求输出 VCF 格式（POS 在第 2 列），但 LLM 不一定严格遵循，
    所以同时识别以下几种模式：
      1. 标准 VCF 数据行：`CHROM\tPOS\tID\tREF\tALT...`（第 2 列为数字）
      2. 显式文本：`pos=123` / `position 123` / `position: 123`
      3. Markdown 表格行：`| chr | 123 | ... |`（第 2 列为数字）
    所有抓到的位置先按 1..7088 过滤，避免行号/版本号等噪音。
    """
    _, _, truth = data["mutation"]
    truth_set = set(truth)
    found = set()

    def _add(p):
        if 1 <= p <= 7088:
            found.add(p)

    # 1. 显式文本模式（最稳）
    for m in re.finditer(r"pos(?:ition)?\s*[:=]?\s*(\d{1,4})", output, re.IGNORECASE):
        _add(int(m.group(1)))

    # 2. VCF 数据行 / markdown 表格行：按 tab、|、或多空格 切分，看第 2 列是不是纯数字
    for line in output.splitlines():
        # 跳过注释/表头
        s = line.lstrip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        # 切分（tab 优先；其次 |；最后多空格——LLM 经常用空格分难示意性 VCF）
        if "\t" in line:
            parts = line.split("\t")
        elif line.count("|") >= 3:
            parts = [p.strip() for p in line.split("|") if p.strip() != ""]
        else:
            # 兜底：连续空格切分，要求至少 5 个字段才像 VCF/表格（避免误抓普通文本）
            parts = line.split()
            if len(parts) < 5:
                continue
            # 第 1 列应该像 chromosome 名（chr17 / 17 / 1 等纯字母数字）
            if not re.match(r"^[A-Za-z0-9_.+-]+$", parts[0]):
                continue
        if len(parts) < 5:
            continue
        # 第 1 列是 CHROM（如 chr17 / 17 / 1），第 2 列是 POS
        pos_str = parts[1]
        # 排除表头文字（POS / Position）
        if pos_str.isdigit():
            _add(int(pos_str))

    hit = found & truth_set
    return {
        "truth_count": len(truth_set),
        "found_count": len(found),
        "hit_count": len(hit),
        "recall": round(len(hit) / len(truth_set), 3) if truth_set else 0,
        "precision": round(len(hit) / len(found), 3) if found else 0,
        # 完整记录 hit/miss 方便人工查看
        "hits": sorted(hit),
        "missed": sorted(truth_set - hit),
        "wrong": sorted(found - truth_set),
    }


def _parse_matrix_answer(output):
    """从 expression_matrix 任务的输出里提取关键数字，容忍格式差异。
    返回 dict，字段缺失时为 None。"""
    result = {
        "top5_genes": [],      # 提取到的 top5 基因名
        "sparsity": None,      # 提取到的稀疏度数字
        "top_cell": None,      # 提取到的最高 UMI cell 名
    }
    # 抓稀疏度：匹配 0.xx 或百分比 或 "3/200"
    for pat in [r"sparsity[^0-9-]*(0\.\d+)", r"fraction[^0-9-]*(0\.\d+)",
                r"(\d+)\s*/\s*200\s*"]:
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            if "/" in m.group(0):
                # 3/200 格式
                num, den = m.group(0).split("/")
                result["sparsity"] = round(int(num.strip()) / int(den.strip()), 3)
            else:
                result["sparsity"] = float(m.group(1))
            break

    # 抓 cell barcode（CD8_off_XXXXXX 这种格式）
    cells = re.findall(r"CD8_off_[ACGT]+", output)
    if cells:
        result["top_cell"] = cells[0]

    # 抓 top5 基因名（从已知候选集里找哪些被提到了，按出现顺序）
    return result


def evaluate_expression_matrix(output, data):
    """评分 expression_matrix 任务：3 个有客观答案的子项。

    Ground truth（从 test_data/expression/top500_expression_matrix.tsv 的
    top20 × 前 10 cells 子矩阵计算得到，确定性）：
      - top5 by row-sum: MT-RNR2, MALAT1, RPS6, RPS8, RPL13A
      - top3 by variance: MT-RNR2, MALAT1, RPS6
      - sparsity: 3/200 = 0.015
      - highest-UMI cell: CD8_off_GACCCCCCTTCT
    """
    truth_top5 = ["MT-RNR2", "MALAT1", "RPS6", "RPS8", "RPL13A"]
    truth_top3_var = ["MT-RNR2", "MALAT1", "RPS6"]
    truth_sparsity = 0.015
    truth_cell = "CD8_off_GACCCCCCTTCT"

    score = {"max": 4, "scored": 0, "checks": {}}

    # 1. top5 命中数
    hits_top5 = [g for g in truth_top5 if re.search(rf"\b{re.escape(g)}\b", output)]
    score["checks"]["top5_genes"] = {
        "expected": truth_top5,
        "hit": hits_top5,
        "hit_count": len(hits_top5),
        "passed": len(hits_top5) >= 4,  # 容忍漏 1 个
    }
    if len(hits_top5) >= 4:
        score["scored"] += 1

    # 2. top3 方差命中
    hits_top3 = [g for g in truth_top3_var if re.search(rf"\b{re.escape(g)}\b", output)]
    score["checks"]["top3_variance"] = {
        "expected": truth_top3_var,
        "hit": hits_top3,
        "hit_count": len(hits_top3),
        "passed": len(hits_top3) >= 2,
    }
    if len(hits_top3) >= 2:
        score["scored"] += 1

    # 3. 稀疏度（容差 ±0.02）
    parsed = _parse_matrix_answer(output)
    if parsed["sparsity"] is not None and abs(parsed["sparsity"] - truth_sparsity) <= 0.02:
        score["checks"]["sparsity"] = {
            "expected": truth_sparsity,
            "got": parsed["sparsity"],
            "passed": True,
        }
        score["scored"] += 1
    else:
        score["checks"]["sparsity"] = {
            "expected": truth_sparsity,
            "got": parsed["sparsity"],
            "passed": False,
        }

    # 4. top cell
    cell_passed = parsed["top_cell"] == truth_cell
    score["checks"]["top_cell"] = {
        "expected": truth_cell,
        "got": parsed["top_cell"],
        "passed": cell_passed,
    }
    if cell_passed:
        score["scored"] += 1

    return score


def evaluate_expression_code(output, data):
    """评分 expression_code 任务：检查 Python 代码的关键点，不强求能跑。

    评分点（每对一个 +1）：
      1. 包含 scanpy / anndata / AnnData import
      2. 用 pandas.read_csv 读 csv 且指定 compression
      3. 转置矩阵（.T 或 transpose）
      4. QC 过滤（cells < 500 detected genes；genes < 3 cells）
      5. mito% 计算 + 阈值过滤
      6. normalize_total(1e4) + log1p
      7. highly_variable_genes（n_top_genes=2000）
      8. PCA → neighbors → umap → leiden 完整流程
      9. rank_genes_groups 调用
      10. AnnData 写盘（write / save）
    """
    text = output
    score = {"max": 10, "scored": 0, "checks": {}}

    def check(name, pattern, flags=re.IGNORECASE):
        ok = re.search(pattern, text, flags) is not None
        score["checks"][name] = {"passed": ok}
        if ok:
            score["scored"] += 1

    check("import_scanpy",       r"\bscanpy\b|\bsc\b\s*=|\bimport\s+scanpy")
    check("import_anndata",      r"\bAnnData\b|\banndata\b")
    check("read_csv",            r"read_csv\s*\(.{0,100}compression")
    check("transpose",           r"\.T\b|transpose\s*\(\s*\)")
    check("qc_filter_cells",     r"(n_genes|detected_genes)\s*<\s*500|min_genes\s*=\s*500|filter_cells.{0,50}500")
    check("qc_filter_genes",     r"(n_cells|detected_cells)\s*<\s*3|min_cells\s*=\s*3|filter_genes.{0,50}3")
    check("mito_pct",            r"(pct_counts_mt|percent_mito|mt_pct|mt\.|MT-)\s*.{0,80}(0\.2|20)")
    check("normalize_log",       r"normalize_total\s*\(.{0,80}1e4|1\s*0\s*0\s*0\s*0.{0,80}log1p|log1p")
    check("hv_genes",            r"highly_variable_genes.{0,80}(2\s*0\s*0\s*0|n_top_genes)")
    check("pca_neighbors_umap_leiden",
          r"(sc\.tl\.pca|tl\.pca).{0,500}(sc\.pp\.neighbors|pp\.neighbors).{0,500}(sc\.tl\.umap|tl\.umap).{0,500}(sc\.tl\.leiden|tl\.leiden)",
          re.IGNORECASE | re.DOTALL)
    check("rank_genes",          r"rank_genes_groups")
    check("save_anndata",        r"\.write\s*\(|save\s*\(|to_h5ad|h5ad")

    # 上面打了 11 个 check，重新校准 max
    score["max"] = len(score["checks"])
    return score


def evaluate_protein_function(output, data):
    """评分 protein_function 任务：基于关键概念词的清单匹配。

    每个子问题有关键词清单，匹配命中越多分越高。容忍同义词。
    """
    text_lower = output.lower()
    score = {"max": 4, "scored": 0, "checks": {}}

    # 子问题 1：IDP 如何介导特异相互作用
    q1_keywords = [["coupled folding", "folding upon binding", "folding and binding"],
                   ["morf", "molecular recognition feature", "intrinsically disordered"],
                   ["induced fit", "conformational selection"]]
    q1_hits = sum(1 for syns in q1_keywords if any(k in text_lower for k in syns))
    q1_passed = q1_hits >= 1
    score["checks"]["q1_idp_mechanism"] = {
        "hit_count": q1_hits,
        "passed": q1_passed,
        "keywords_matched": [k for syns in q1_keywords for k in syns if k in text_lower],
    }
    if q1_passed:
        score["scored"] += 1

    # 子问题 2：具体机制名
    q2_keywords = ["morf", "molecular recognition feature",
                   "coupled folding", "folding upon binding",
                   "short linear motif", "slim",
                   "intrinsically disordered"]
    q2_hits = [k for k in q2_keywords if k in text_lower]
    q2_passed = len(q2_hits) >= 1
    score["checks"]["q2_mechanism_name"] = {
        "hit_count": len(q2_hits),
        "keywords_matched": q2_hits,
        "passed": q2_passed,
    }
    if q2_passed:
        score["scored"] += 1

    # 子问题 3：BRCA1-p53 功能关系
    q3_keywords = ["transcription", "transactivation", "stabiliz",
                   "apoptosis", "cell cycle", "dna repair",
                   "tumor suppressor", "oncogene"]
    q3_hits = [k for k in q3_keywords if k in text_lower]
    q3_passed = len(q3_hits) >= 2
    score["checks"]["q3_function"] = {
        "hit_count": len(q3_hits),
        "keywords_matched": q3_hits,
        "passed": q3_passed,
    }
    if q3_passed:
        score["scored"] += 1

    # 子问题 4：实验方法
    q4_keywords = {
        "nmr": ["nmr", "nuclear magnetic"],
        "hdx-ms": ["hdx", "hydrogen/deuterium", "hydrogen exchange", "deuterium exchange"],
        "alanine": ["alanine scanning", "ala scan", "alanine mutagenesis"],
        "cryo-em": ["cryo-em", "cryoem", "cryo em", "electron microscopy"],
        "peptide_array": ["peptide array", "peptide tiling"],
    }
    q4_hits = [name for name, syns in q4_keywords.items() if any(k in text_lower for k in syns)]
    q4_passed = len(q4_hits) >= 1
    score["checks"]["q4_experiment"] = {
        "methods_matched": q4_hits,
        "passed": q4_passed,
    }
    if q4_passed:
        score["scored"] += 1

    return score


# ============ benchmark 评估器（有 ground truth）============
def _extract_indexed_answers(output, n_expected, valid_set):
    """从 'N. answer' 格式的输出里抽取 N 个答案。
    valid_set 是合法答案集合（如 {'yes','no','maybe'} 或 {'A','B','C','D'}）。
    """
    found = {}
    # 匹配 "1. yes" / "2) no" / "Q3: A" 等
    for m in re.finditer(r"(?:^|\n)\s*(?:Q?|\#)?\s*(\d{1,3})\s*[\.\)\:\-]\s*([A-Za-z]+)", output):
        idx = int(m.group(1))
        ans = m.group(2).strip().lower()
        if 1 <= idx <= n_expected and ans in valid_set and idx not in found:
            found[idx] = ans
    return found


def evaluate_pubmedqa(output, data):
    items = data["benchmark"][0]
    n = len(items)
    truth = {i + 1: items[i]["answer"].lower() for i in range(n)}  # 1-based
    found = _extract_indexed_answers(output, n, {"yes", "no", "maybe"})
    correct = sum(1 for i, a in found.items() if a == truth.get(i))
    # 三类基线准确率
    majority_class = "yes"
    baseline = sum(1 for i in range(1, n + 1) if truth[i] == majority_class) / n
    return {
        "max": n,
        "scored": correct,
        "answered": len(found),
        "accuracy": round(correct / n, 3) if n else 0,
        "majority_baseline_yes": round(baseline, 3),
        "missing": sorted(set(range(1, n + 1)) - set(found)),
        # 逐题对错
        "per_item": [{"i": i, "truth": truth[i], "got": found.get(i), "correct": found.get(i) == truth[i]} for i in range(1, n + 1)],
    }


def evaluate_medmcqa(output, data):
    items = data["benchmark"][1]
    n = len(items)
    truth = {i + 1: items[i]["answer"].lower() for i in range(n)}  # 1-based, a/b/c/d
    found = _extract_indexed_answers(output, n, {"a", "b", "c", "d"})
    correct = sum(1 for i, a in found.items() if a == truth.get(i))
    # 4 选 1 随机基线 25%
    from collections import Counter
    counter = Counter(truth.values())
    majority_class = counter.most_common(1)[0][0]
    baseline = counter[majority_class] / n
    return {
        "max": n,
        "scored": correct,
        "answered": len(found),
        "accuracy": round(correct / n, 3) if n else 0,
        "random_baseline": 0.25,
        "majority_baseline": round(baseline, 3),
        "missing": sorted(set(range(1, n + 1)) - set(found)),
        "per_item": [{"i": i, "truth": truth[i], "got": found.get(i), "correct": found.get(i) == truth[i]} for i in range(1, n + 1)],
    }


def evaluate_json_output(output, data):
    """JSON 任务：解析输出、检查 schema、对比 5 个变异。
    评分维度（max=6）：
      1. 输出能被 json.loads 解析（最严苛，错一个字符就 0 分）
      2. 顶层是 dict 且有 gene / variants / count 字段
      3. count == 5
      4. variants 长度 == 5
      5. 5 个位置全部正确 (205/245/713/913/5544)
      6. 5 个 ref/alt 全部正确
    """
    score = {"max": 6, "scored": 0, "checks": {}}

    # 去 markdown 围栏
    text = output.strip()
    if text.startswith("```"):
        # 去掉首尾 ``` 行
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 找第一个 { 到最后一个 }（容忍前后有解释文字）
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        score["checks"]["parse"] = {"passed": False, "error": "no JSON object found"}
        return score
    try:
        obj = json.loads(text[start:end + 1])
        score["checks"]["parse"] = {"passed": True}
        score["scored"] += 1
    except json.JSONDecodeError as e:
        score["checks"]["parse"] = {"passed": False, "error": str(e)[:100]}
        return score  # 后续检查无意义

    # 2. 顶层结构
    struct_ok = (isinstance(obj, dict)
                 and "gene" in obj
                 and "variants" in obj
                 and "count" in obj)
    score["checks"]["schema"] = {"passed": struct_ok}
    if struct_ok:
        score["scored"] += 1
    else:
        return score

    # 3. count == 5
    count_ok = obj["count"] == 5
    score["checks"]["count_eq_5"] = {"passed": count_ok, "got": obj["count"]}
    if count_ok:
        score["scored"] += 1

    # 4. variants 长度 == 5
    variants = obj.get("variants") or []
    len_ok = isinstance(variants, list) and len(variants) == 5
    score["checks"]["variants_len_5"] = {"passed": len_ok, "got": len(variants) if isinstance(variants, list) else None}
    if len_ok:
        score["scored"] += 1
    if not isinstance(variants, list):
        return score

    # 5. 位置
    truth_pos = {205, 245, 713, 913, 5544}
    got_pos = set()
    for v in variants:
        try:
            got_pos.add(int(v.get("position")))
        except (TypeError, ValueError):
            pass
    pos_ok = got_pos == truth_pos
    score["checks"]["positions"] = {
        "passed": pos_ok,
        "expected": sorted(truth_pos),
        "got": sorted(got_pos),
    }
    if pos_ok:
        score["scored"] += 1

    # 6. ref/alt
    truth_ra = {
        205: ("T", "C"), 245: ("C", "G"), 713: ("A", "T"),
        913: ("C", "G"), 5544: ("C", "T"),
    }
    correct_ra = 0
    for v in variants:
        try:
            pos = int(v.get("position"))
            ref = str(v.get("ref", "")).upper()
            alt = str(v.get("alt", "")).upper()
            if truth_ra.get(pos) == (ref, alt):
                correct_ra += 1
        except (TypeError, ValueError):
            pass
    ra_ok = correct_ra == 5
    score["checks"]["ref_alt"] = {"passed": ra_ok, "correct_count": correct_ra}
    if ra_ok:
        score["scored"] += 1

    return score


def evaluate_long_generation(output, data):
    """评分 long_generation 任务：长度达标 + 4 个章节齐全 + 关键主题词覆盖。

    评分维度（max=6）：
      1. 字数达标（500-1200 词）
      2. 4 个章节标题都出现（Mechanism / Clinical / Resistance / Open question）
      3-6. 4 个主题词簇分别命中至少 2 个（机制 / 药名 / 抗性 / open question）
    """
    score = {"max": 6, "scored": 0, "checks": {}}
    text_lower = output.lower()
    word_count = len(output.split())

    # 1. 字数
    len_ok = 500 <= word_count <= 1200
    score["checks"]["word_count"] = {
        "passed": len_ok,
        "word_count": word_count,
        "expected": "500-1200",
    }
    if len_ok:
        score["scored"] += 1

    # 2. 4 个章节标题（容忍 markdown / 全大写）
    section_keywords = {
        "mechanism":      ["mechanism", "molecular basis"],
        "clinical":       ["clinical evidence", "clinical", "pivotal trial", "approval"],
        "resistance":     ["resistance mechanism", "resistance", "acquired resistance"],
        "open_question":  ["open question", "open research", "future direction"],
    }
    sections_found = sum(
        1 for kws in section_keywords.values()
        if any(kw in text_lower for kw in kws)
    )
    sections_ok = sections_found >= 4
    score["checks"]["sections"] = {
        "passed": sections_ok,
        "found": sections_found,
        "expected": 4,
    }
    if sections_ok:
        score["scored"] += 1

    # 3-6. 主题词簇（4 组，每组命中 ≥2 个）
    topic_clusters = {
        "hr_mechanism":    ["homologous recombination", "hr repair", "dsb", "double-strand break",
                            "parp trapping", "single-strand break", "ssb", "brca1", "brca2"],
        "drug_names":      ["olaparib", "rucaparib", "niraparib", "talazoparib",
                            "parp inhibitor", "parpi"],
        "resistance_term": ["reversion mutation", "reversion", "fork protection",
                            "replication fork", "drug efflux", "abc transporter",
                            "loss of 53bp1", "shieldin"],
        "outcome_metric":  ["pfs", "progression-free survival", "hazard ratio",
                            "overall survival", "os", "response rate", "orr"],
    }
    for cluster_name, kws in topic_clusters.items():
        hits = [k for k in kws if k in text_lower]
        ok = len(hits) >= 2
        score["checks"][f"topic_{cluster_name}"] = {
            "passed": ok,
            "hit_count": len(hits),
            "keywords_matched": hits,
        }
        if ok:
            score["scored"] += 1

    return score


EVALUATORS = {
    "mutation_call":       evaluate_mutation_call,
    "expression_matrix":   evaluate_expression_matrix,
    "expression_code":     evaluate_expression_code,
    "protein_function":    evaluate_protein_function,
    "pubmedqa":            evaluate_pubmedqa,
    "medmcqa":             evaluate_medmcqa,
    "json_output":         evaluate_json_output,
    "long_generation":     evaluate_long_generation,
}


# ============ 结果可视化 ============
def _quality_to_01(task_name, quality):
    """把不同任务的 quality 评分归一化到 [0, 1]，便于在同一张图上展示。
    没有评估器的任务（如 expression_genes）返回 None。
    """
    if quality is None:
        return None
    # mutation_call: 用 recall（也能反映精确度，因为模型 rarely 误报真位置）
    if "recall" in quality:
        return quality["recall"]
    # accuracy 类（pubmedqa / medmcqa）
    if "accuracy" in quality:
        return quality["accuracy"]
    # scored/max 类（expression_matrix / expression_code / protein_function / json_output）
    if "scored" in quality and "max" in quality:
        m = quality["max"] or 1
        return quality["scored"] / m
    return None


# ============ 综合评分 (0-100) ============
# 设计目标：
#   - 单机就能算（不依赖其他机器），方便每台机器自己看分
#   - 体现"生物医学研究适用度"：质量 > 吞吐 > TTFT > 稳定性
#   - 各子项分别归一化到 0-100 再加权，避免某一维的量纲污染总分
#
# 权重分配（合计 100%）：
#   质量分   50%  答错比慢更严重，生物医学研究的核心
#   吞吐分   30%  实际工作效率
#   TTFT分   10%  交互响应感
#   稳定性   10%  长生成任务的运行间一致（CV 越小越好）
SCORE_WEIGHTS = {"quality": 0.5, "throughput": 0.3, "ttft": 0.1, "stability": 0.1}


def _tps_to_score(tps):
    """吞吐 → 0-100。锚点：5 tok/s=30 分，30 tok/s=70 分，60 tok/s=90 分，100+ tok/s≈100 分。
    用 log 归一化，避免高端 GPU 把分数拉爆。"""
    if tps is None or tps <= 0:
        return 0
    import math
    # log10(5)=0.7, log10(100)=2 → 锚点映射
    return max(0.0, min(100.0, (math.log10(tps) - 0.7) / (2.0 - 0.7) * 100))


def _ttft_to_score(ttft_s):
    """TTFT → 0-100（越低越好）。锚点：0.1s=95, 1s=80, 5s=50, 15s+≤20。"""
    if ttft_s is None or ttft_s < 0:
        return 0
    import math
    if ttft_s <= 0.1:
        return 100.0
    # log10(0.1)=-1, log10(20)≈1.3
    return max(0.0, min(100.0, (1.3 - math.log10(ttft_s)) / (1.3 - (-1.0)) * 100))


def _stability_to_score(cv_pct):
    """长任务运行间 CV → 0-100。CV<1%=100 分，CV=10%=0 分（线性）。"""
    if cv_pct is None or cv_pct < 0:
        return 0
    return max(0.0, min(100.0, 100.0 * (10.0 - cv_pct) / (10.0 - 1.0)))


def compute_overall_score(all_results):
    """计算单机综合评分 (0-100)。返回 dict 含总分 + 各子项。
    None 表示该项数据不足（比如某个任务全失败），按权重 0 计入。
    """
    tasks = all_results.get("tasks", {})

    # ---- 1. 质量：所有有 evaluator 任务的归一化分平均 ----
    quality_scores = []
    for tname, _, _ in TASKS:
        td = tasks.get(tname, {})
        if "quality" not in td:
            continue
        q01 = _quality_to_01(tname, td["quality"])
        if q01 is not None:
            quality_scores.append(q01)
    quality_score = (sum(quality_scores) / len(quality_scores) * 100) if quality_scores else None

    # ---- 2. 吞吐：所有任务的 tok/s 平均后再打分 ----
    tps_values = []
    for tname, _, _ in TASKS:
        td = tasks.get(tname, {})
        if "tok_per_s" in td and isinstance(td["tok_per_s"], dict):
            v = td["tok_per_s"]["mean"]
            if v is not None:
                tps_values.append(v)
    # 用中位数更鲁棒（不受极端值影响）
    if tps_values:
        tps_sorted = sorted(tps_values)
        tps_median = tps_sorted[len(tps_sorted) // 2]
        throughput_score = _tps_to_score(tps_median)
    else:
        throughput_score = None

    # ---- 3. TTFT：所有任务 TTFT 中位数 ----
    ttft_values = []
    for tname, _, _ in TASKS:
        td = tasks.get(tname, {})
        if "ttft_s" in td and isinstance(td["ttft_s"], dict):
            v = td["ttft_s"]["mean"]
            if v is not None:
                ttft_values.append(v)
    if ttft_values:
        ttft_sorted = sorted(ttft_values)
        ttft_median = ttft_sorted[len(ttft_sorted) // 2]
        ttft_score = _ttft_to_score(ttft_median)
    else:
        ttft_score = None

    # ---- 4. 稳定性：long_generation 任务的 CV（运行间方差）----
    stability_score = None
    lg = tasks.get("long_generation", {})
    samples = lg.get("samples") or []
    totals = [s.get("total_s") for s in samples if "total_s" in s]
    if len(totals) >= 2:
        mean_t = sum(totals) / len(totals)
        if mean_t > 0:
            var = sum((t - mean_t) ** 2 for t in totals) / len(totals)
            cv = (var ** 0.5) / mean_t * 100
            stability_score = _stability_to_score(cv)

    # ---- 加权求和：缺项按 0 计入（拖低总分，体现"数据不全不算好机器"）----
    sub_scores = {
        "quality":    quality_score,
        "throughput": throughput_score,
        "ttft":       ttft_score,
        "stability":  stability_score,
    }
    overall = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        s = sub_scores[key]
        overall += weight * (s if s is not None else 0.0)

    return {
        "overall":        round(overall, 1),
        "sub_scores":     {k: (round(v, 1) if v is not None else None) for k, v in sub_scores.items()},
        "weights":        SCORE_WEIGHTS,
        # 也带上原始指标方便调试
        "quality_count":  len(quality_scores),
        "tps_median":     round(tps_median, 2) if tps_values else None,
        "ttft_median":    round(ttft_median, 2) if ttft_values else None,
    }


def plot_single_run(all_results, out_path):
    """画单机结果图：4 个子图（TTFT / tok/s / 质量 / 硬件信息表）。
    out_path 是 PNG 路径。"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # 用 Agg 后端，避免 macOS/Linux 无头时报错
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

    env = all_results.get("environment", {})
    label = all_results.get("label", "unknown")
    model = all_results.get("model", "?")
    engine = env.get("llm_engine", "?")
    chip = env.get("chip", "?")
    mem = env.get("memory_gb_str") or f"{env.get('memory_gb', '?')}GB"
    gpu = env.get("gpu", "?")
    ctx = env.get("llm_context", "?")

    tasks = all_results["tasks"]
    # 只要"有可绘图数据"的任务：tok_per_s 是必须的（用于吞吐图）；TTFT 缺失也算（CLI 模式）
    task_names = [t for t, _, _ in TASKS
                  if t in tasks
                  and isinstance(tasks[t].get("tok_per_s"), dict)
                  and tasks[t]["tok_per_s"]["mean"] is not None]
    failed = [t for t, _, _ in TASKS if t not in task_names]

    # 三个核心指标（TTFT 可能为 None：CLI 模式没有流式）
    ttfts_raw = [tasks[t].get("ttft_s", {}).get("mean") for t in task_names]
    has_ttft = any(v is not None for v in ttfts_raw)
    # 给 None 填 0 用于画图（后面会标注），None 单独记一份
    ttfts = [v if v is not None else 0 for v in ttfts_raw]
    tps = [tasks[t]["tok_per_s"]["mean"] for t in task_names]
    quals = [_quality_to_01(t, tasks[t].get("quality")) for t in task_names]

    fig = plt.figure(figsize=(15, 9), dpi=110)
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25,
                  left=0.07, right=0.97, top=0.88, bottom=0.08)

    color_blue = "#3b82f6"
    color_green = "#10b981"
    color_amber = "#f59e0b"
    color_gray = "#94a3b8"

    # ---- 子图 1: TTFT (越低越好) ----
    ax1 = fig.add_subplot(gs[0, 0])
    # TTFT 为 None 的任务（CLI 模式）：用浅灰柱 + 标 "n/a"
    bar_colors = [color_blue if v is not None and v > 0 else "#cbd5e1" for v in ttfts_raw]
    bars = ax1.bar(task_names, ttfts, color=bar_colors, alpha=0.85, edgecolor="#1e40af")
    ax1.set_ylabel("TTFT (s)  ↓ better", fontsize=10)
    ax1.set_title("Time to First Token", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for b, v_raw in zip(bars, ttfts_raw):
        label = "n/a" if v_raw is None else f"{v_raw:.2f}"
        # 标在柱顶（None 时柱高为 0，标在底部稍微上方）
        y = (v_raw or 0) if v_raw is not None else 0.05
        ax1.text(b.get_x() + b.get_width() / 2, y, label,
                 ha="center", va="bottom", fontsize=8,
                 color="#94a3b8" if v_raw is None else "black")

    # ---- 子图 2: tok/s (越高越好) ----
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(task_names, tps, color=color_green, alpha=0.85, edgecolor="#065f46")
    ax2.set_ylabel("tokens/s  ↑ better", fontsize=10)
    ax2.set_title("Generation Throughput", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for b, v in zip(bars, tps):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)

    # ---- 子图 3: 质量分（0-1，越高越好）----
    ax3 = fig.add_subplot(gs[1, 0])
    plot_names = [t for t, q in zip(task_names, quals) if q is not None]
    plot_vals = [q for q in quals if q is not None]
    no_eval = [t for t, q in zip(task_names, quals) if q is None]
    bars = ax3.bar(plot_names, plot_vals, color=color_amber, alpha=0.85, edgecolor="#92400e")
    ax3.set_ylabel("Quality score (0-1)  ↑ better", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Output Quality (vs ground truth)", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="x", rotation=30, labelsize=8)
    ax3.axhline(y=0.25, color=color_gray, linestyle="--", linewidth=0.8,
                label="random (4-choice)")
    ax3.axhline(y=0.5, color=color_gray, linestyle=":", linewidth=0.8,
                label="majority baseline")
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.9)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    for b, v in zip(bars, plot_vals):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    if no_eval:
        ax3.text(0.5, -0.32, f"(no evaluator: {', '.join(no_eval)})",
                 transform=ax3.transAxes, ha="center", fontsize=7,
                 style="italic", color="#6b7280")

    # ---- 子图 4: 硬件信息表 ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_title("Environment", fontsize=12, fontweight="bold", loc="left")
    info_rows = [
        ("Label",       label),
        ("Chip",        chip),
        ("Memory",      str(mem)),
        ("Cores",       str(env.get("cpu_cores", "?"))),
        ("GPU",         gpu),
        ("Engine",      engine),
        ("Model",       model),
        ("Context",     str(ctx)),
        ("API URL",     all_results.get("api_url", "?")),
        ("Runs / task", f"{all_results.get('runs', '?')}"),
        ("Timestamp",   all_results.get("timestamp", "?")[:19]),
    ]
    if failed:
        info_rows.append(("Failed tasks", ", ".join(failed)))
    tbl = ax4.table(
        cellText=[[k, str(v)] for k, v in info_rows],
        colWidths=[0.28, 0.72],
        loc="upper left",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    # 给两列不同颜色
    n_rows = len(info_rows)
    for r in range(n_rows):
        for c in range(2):
            cell = tbl[r, c]
            cell.set_edgecolor("#e5e7eb")
            if c == 0:
                cell.set_facecolor("#f3f4f6")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#ffffff")
    # 顶部标题（含综合评分）
    score = compute_overall_score(all_results)
    overall = score["overall"]
    # 0-100 颜色：红(<40) / 橙(40-70) / 绿(>70)
    if overall >= 70:
        score_color, verdict = "#16a34a", "good"
    elif overall >= 40:
        score_color, verdict = "#f59e0b", "fair"
    else:
        score_color, verdict = "#dc2626", "weak"

    title = f"Local LLM Benchmark — {label}"
    # 用分数+子项做副标题，让 0-100 分一眼可见
    sub = score["sub_scores"]
    sub_str = (f"Q={sub['quality'] if sub['quality'] is not None else '-'}  "
               f"Tput={sub['throughput'] if sub['throughput'] is not None else '-'}  "
               f"TTFT={sub['ttft'] if sub['ttft'] is not None else '-'}  "
               f"Stab={sub['stability'] if sub['stability'] is not None else '-'}")
    subtitle = (f"{chip}  ·  {mem}  ·  {engine}  ·  {model}  ·  ctx={ctx}")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10, color="#4b5563")
    # 分数行：大号 0-100 + 子项 + 文字 verdict
    fig.text(0.5, 0.905,
             f"Overall score: ",
             ha="center", fontsize=11, color="#374151")
    # 用 annotate 画大号彩色数字（fig.text 不支持局部颜色）
    fig.text(0.565, 0.905,
             f"{overall:.0f}/100",
             ha="center", fontsize=14, fontweight="bold", color=score_color)
    fig.text(0.5, 0.885,
             f"({sub_str})  ·  {verdict}",
             ha="center", fontsize=9, color="#6b7280")

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============ 主流程 ============
def main():
    print("=" * 70)
    print("🧬 本地 LLM 性能基准 — 9 个生物医学任务")
    print(f"   label:  {LLM_LABEL}")
    print(f"   API:    {API_URL}")
    print(f"   Model:  {MODEL_NAME}")
    print(f"   Runs:   {RUNS} per task (+1 warmup)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n📦 加载数据...")
    data = {
        "mutation":   load_mutation(),
        "expression": load_expression(),
        "protein":    load_protein(),
        "benchmark":  load_benchmark(),
    }
    print(f"   mutation: ref={len(data['mutation'][0])}B mut={len(data['mutation'][1])}B "
          f"truth={len(data['mutation'][2])} positions")
    print(f"   protein:  p53_fasta={len(data['protein'][0])}B ss={len(data['protein'][1])}B")
    print(f"   benchmark: pubmedqa={len(data['benchmark'][0])} medmcqa={len(data['benchmark'][1])}")

    # 健康检查
    print("\n🔌 健康检查：发送一个短请求...")
    probe = call_llm_stream("Reply with exactly: OK", 16)
    print(f"   probe ok: ttft={probe['ttft_s']}s total={probe['total_s']}s "
          f"finish={probe['finish_reason']}")

    all_results = {
        "label": LLM_LABEL,
        "api_url": API_URL,
        "model": MODEL_NAME,
        "runs": RUNS,
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    # 硬件/环境快照（在 all_results 创建之后赋值）
    hw = snapshot_environment()
    print(f"   host:     {hw.get('hostname')} / {hw.get('platform', '')}")
    print(f"   chip:     {hw.get('chip', 'N/A')}  cores={hw.get('cpu_cores', 'N/A')}  "
          f"mem={hw.get('memory_gb', 'N/A')}GB")
    print(f"   gpu:      {hw.get('gpu', 'N/A')}  ctx={hw.get('llm_context', 'N/A')}")
    all_results["environment"] = hw

    for task_name, builder, max_tok in TASKS:
        print(f"\n{'─' * 70}")
        print(f"📋 Task: {task_name}  (max_tokens={max_tok})")
        print(f"{'─' * 70}")
        prompt, _ = builder(data)
        prompt_chars = len(prompt)
        print(f"   prompt: {prompt_chars} chars")

        # 预热 1 次
        print(f"   🔥 warmup...", end=" ", flush=True)
        try:
            warm = call_llm_stream(prompt, max_tok)
            print(f"done (ttft={warm['ttft_s']}s, total={warm['total_s']}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            all_results["tasks"][task_name] = {"error": str(e)}
            continue

        runs = []
        for i in range(RUNS):
            try:
                r = call_llm_stream(prompt, max_tok)
                runs.append(r)
                print(f"   Run {i+1}/{RUNS}: ttft={r['ttft_s']:.3f}s  "
                      f"total={r['total_s']:.3f}s  "
                      f"out_tok={r['completion_tokens']}  "
                      f"tok/s={r['tok_per_s']}  "
                      f"finish={r['finish_reason']}")
            except Exception as e:
                print(f"   Run {i+1}/{RUNS}: FAILED: {e}")
                runs.append({"error": str(e)})

        # 聚合统计
        ok = [r for r in runs if "error" not in r]
        summary = {"runs": len(runs), "ok": len(ok), "samples": runs}
        if ok:
            def stat(key):
                xs = [r[key] for r in ok]
                return {
                    "mean": round(sum(xs) / len(xs), 3),
                    "min": round(min(xs), 3),
                    "max": round(max(xs), 3),
                }
            summary["ttft_s"] = stat("ttft_s")
            summary["total_s"] = stat("total_s")
            summary["tok_per_s"] = stat("tok_per_s")
            summary["prompt_tokens"] = ok[0]["prompt_tokens"]
            summary["completion_tokens"] = stat("completion_tokens")

        # 质量评估（如有）
        if task_name in EVALUATORS and ok:
            last = ok[-1]
            eval_result = EVALUATORS[task_name](last["output"], data)
            summary["quality"] = eval_result
            print(f"   📊 quality: {eval_result}")

        all_results["tasks"][task_name] = summary

    # ============ 汇总表 ============
    print(f"\n{'=' * 70}")
    print(f"📊 汇总（{LLM_LABEL} / {MODEL_NAME}）")
    print(f"{'=' * 70}")
    print(f"{'task':<22}{'ttft(s)':<14}{'total(s)':<14}{'tok/s':<14}{'quality':<24}")
    print("-" * 88)
    for task_name, _, _ in TASKS:
        t = all_results["tasks"].get(task_name, {})
        if "ttft_s" not in t:
            print(f"{task_name:<22}{'FAILED':<14}{'-':<14}{'-':<14}{'-':<24}")
            continue
        ttft = t["ttft_s"]["mean"]
        total = t["total_s"]["mean"]
        tps = t["tok_per_s"]["mean"]
        q = "-"
        if "quality" in t:
            qv = t["quality"]
            # mutation_call 用 P/R，其他用 scored/max
            if "precision" in qv:
                q = f"P={qv.get('precision',0)} R={qv.get('recall',0)}"
            elif "scored" in qv:
                q = f"{qv['scored']}/{qv['max']}"
        print(f"{task_name:<22}{ttft:<14}{total:<14}{tps:<14}{q:<24}")

    # ============ 综合评分 ============
    score = compute_overall_score(all_results)
    all_results["overall_score"] = score
    print("\n" + "-" * 88)
    overall = score["overall"]
    sub = score["sub_scores"]
    verdict = "good" if overall >= 70 else ("fair" if overall >= 40 else "weak")
    print(f"⭐ Overall: {overall:.1f}/100  ({verdict})")
    print(f"   quality={sub['quality'] if sub['quality'] is not None else '-'}  "
          f"throughput={sub['throughput'] if sub['throughput'] is not None else '-'}  "
          f"ttft={sub['ttft'] if sub['ttft'] is not None else '-'}  "
          f"stability={sub['stability'] if sub['stability'] is not None else '-'}")
    print(f"   weights: {SCORE_WEIGHTS}")

    # ============ 保存 JSON ============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"bench_{LLM_LABEL}_{timestamp}.json"
    out_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n💾 详细结果: {out_file}")

    # ============ 生成可视化图 ============
    if PLOT:
        try:
            png_file = OUTPUT_DIR / f"bench_{LLM_LABEL}_{timestamp}.png"
            plot_single_run(all_results, png_file)
            print(f"📊 可视化图: {png_file}")
        except Exception as e:
            print(f"⚠️  绘图失败（不影响结果）: {type(e).__name__}: {e}")
            print(f"   设 PLOT=0 可跳过绘图。")


if __name__ == "__main__":
    main()
