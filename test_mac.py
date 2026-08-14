#!/usr/bin/env python3
"""
Mac 测试脚本 - 用 ds4 命令行（CLI 直调，绕开 HTTP）跑完整 9 个生物医学任务。

与 test_dgx.py 的关系：
  - 9 个 task builder / TASKS / evaluator / compute_overall_score / plot_single_run
    全部从 test_dgx.py 复用（import），保证两边 prompt 和评估完全一致。
  - 唯一区别是 LLM 调用方式：本脚本用 subprocess 调 ds4 二进制，
    test_dgx.py 用 requests 调 ds4-server HTTP API。
  - 因此没有 TTFT（CLI 不流式），其他指标都一致。

跑完输出 3 个文件到 OUTPUT_DIR：
  - mac_results_<ts>.json      原生格式（保留每条原始 output 文本）
  - bench_<label>_<ts>.json    与 test_dgx.py 同格式，可直接喂给 compare_bench.py
  - bench_<label>_<ts>.png     单机可视化图（含 0-100 综合评分）

运行方式（在 venv 中）：
    python test_mac.py
可选环境变量：
    DS4_PATH / MODEL_PATH / DATA_DIR / OUTPUT_DIR
    LLM_LABEL  (写入结果文件名，默认 mac-m5max-ds4-CLI)
    RUNS       (每个任务的正式测试次数，默认 5)
    WARMUP     (每个任务的预热次数，默认 1)
    PLOT=0     (跳过绘图)
"""

import json
import os
import statistics
import subprocess
import tempfile
import time
from datetime import datetime

# 从 test_dgx 复用所有共享逻辑：数据加载、9 个 task、9 个 evaluator、综合评分、绘图
import test_dgx as T
from test_dgx import (
    TASKS, EVALUATORS,
    load_mutation, load_expression, load_protein, load_benchmark,
    compute_overall_score, plot_single_run,
)

# ============ 配置区 ============
# 用环境变量覆盖更方便；默认值给本机当前安装位置。
DS4_PATH = os.environ.get("DS4_PATH", os.path.expanduser("~/ds4/ds4"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/ds4/ds4flash.gguf"))
DATA_DIR = os.environ.get("DATA_DIR", "./test_data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results")
LLM_LABEL = os.environ.get("LLM_LABEL", "mac-m5max-ds4-CLI")
RUNS = int(os.environ.get("RUNS", "5"))       # 与 test_dgx.py 默认值一致
WARMUP = int(os.environ.get("WARMUP", "1"))   # CLI 启动慢，预热 1 次即可
PLOT = os.environ.get("PLOT", "1") not in ("0", "false", "no", "")


# ============ 调用 ds4 CLI ============
def call_ds4(prompt, max_tokens):
    """通过 ds4 命令行调用本地推理。Prompt 写临时文件，--prompt-file 传入。

    注意：
    - ds4-server 在跑时会持有 /tmp/ds4.lock，直接调 ds4 二进制会被拒。
      用 DS4_LOCK_FILE 环境变量指定独立的锁文件避开。
    - 参数名：--prompt-file (不是 -f)、--temp (不是 -t)、-n/--tokens、--seed。
    - --nothink 关掉思考，否则 max_tokens 全被 reasoning 吃掉。
    - CLI 不流式，所以测不到 TTFT；只测总耗时。

    返回 dict 字段与 test_dgx.call_llm_stream 对齐，便于复用 main 逻辑；
    ttft_s 始终为 None（CLI 无法测首 token 时间）。
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        start = time.perf_counter()
        # ds4 二进制依赖 cwd = ~/ds4 才能找到 metal/ 源文件
        ds4_dir = os.path.dirname(os.path.abspath(DS4_PATH))
        result = subprocess.run(
            [DS4_PATH, "-m", MODEL_PATH,
             "--prompt-file", prompt_file,
             "-n", str(max_tokens),
             "--temp", "0.0",
             "--seed", "42",
             "--nothink"],
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd=ds4_dir,
            env={**os.environ, "DS4_LOCK_FILE": "/tmp/ds4_cli_test.lock"},
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"ds4 rc={result.returncode}: {result.stderr[:400]}")

        # CLI 没有流式 token 计数；用 max_tokens 估算 tok/s（实际输出可能更少，会略高估）
        out_text = result.stdout
        # 简单 token 估算：按英文 ~4 字符/token
        est_tokens = max(1, len(out_text) // 4)
        return {
            "output": out_text,
            "ttft_s": None,           # CLI 无流式
            "total_s": round(elapsed, 3),
            "prompt_tokens": None,    # CLI 不返回
            "completion_tokens": est_tokens,
            "tok_per_s": round(est_tokens / elapsed, 2) if elapsed > 0 else 0,
            "finish_reason": "stop",  # CLI 跑完即 stop，没有 finish_reason 字段
        }
    finally:
        os.unlink(prompt_file)


# ============ 硬件快照（与 test_dgx.snapshot_environment 等价，但 engine 标 CLI）============
def snapshot_environment():
    import platform as _p
    import socket as _sock
    import subprocess as _sp
    snap = {
        "hostname": _sock.gethostname(),
        "platform": f"{_p.system()} {_p.release()} ({_p.machine()})",
        "python":   _p.python_version(),
        "llm_label": LLM_LABEL,
        "llm_api_url": "local CLI (subprocess)",
        "llm_model": os.path.basename(MODEL_PATH),
        "llm_engine": "ds4 binary (CLI)",
        "llm_context": os.environ.get("LLM_CONTEXT", "256000"),
    }
    if _p.system() == "Darwin":
        try:
            hw = _sp.run(["system_profiler", "SPHardwareDataType"],
                         capture_output=True, text=True, timeout=5).stdout
            for line in hw.splitlines():
                line = line.strip()
                if line.startswith("Chip:"):
                    snap["chip"] = line.split(":", 1)[1].strip()
                    snap["gpu"] = snap["chip"]
                elif line.startswith("Total Number of Cores:"):
                    snap["cpu_cores"] = line.split(":", 1)[1].strip()
                elif line.startswith("Memory:"):
                    snap["memory_gb_str"] = line.split(":", 1)[1].strip()
            try:
                mem_bytes = int(_sp.run(["sysctl", "-n", "hw.memsize"],
                                        capture_output=True, text=True, timeout=3).stdout.strip())
                snap["memory_gb"] = round(mem_bytes / (1024 ** 3), 1)
            except Exception:
                pass
        except Exception:
            pass
    return snap


# ============ 主流程 ============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"🧬 Mac CLI 测试 — {len(TASKS)} 个生物医学任务（ds4 binary 直调）")
    print(f"   DS4_PATH:  {DS4_PATH}")
    print(f"   MODEL:     {MODEL_PATH}")
    print(f"   Label:     {LLM_LABEL}")
    print(f"   Warmup:    {WARMUP}/task   Runs: {RUNS}/task")
    print("=" * 70)

    # 加载所有数据（复用 test_dgx 的 loader）
    print("\n📦 加载数据...")
    data = {
        "mutation":   load_mutation(),
        "expression": load_expression(),
        "protein":    load_protein(),
        "benchmark":  load_benchmark(),
    }
    print(f"   mutation: ref={len(data['mutation'][0])}B mut={len(data['mutation'][1])}B")
    print(f"   benchmark: pubmedqa={len(data['benchmark'][0])} medmcqa={len(data['benchmark'][1])}")

    all_results = {
        "label": LLM_LABEL,
        "api_url": "local CLI (subprocess)",
        "model": os.path.basename(MODEL_PATH),
        "runs": RUNS,
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    # 硬件快照
    hw = snapshot_environment()
    print(f"   host:     {hw.get('hostname')} / {hw.get('platform', '')}")
    print(f"   chip:     {hw.get('chip', 'N/A')}  mem={hw.get('memory_gb', 'N/A')}GB")
    all_results["environment"] = hw

    # 遍历 9 个任务
    for task_name, builder, max_tok in TASKS:
        print(f"\n{'─' * 70}")
        print(f"📋 Task: {task_name}  (max_tokens={max_tok})")
        print(f"{'─' * 70}")
        prompt, _ = builder(data)
        print(f"   prompt: {len(prompt)} chars")

        # 预热
        print(f"   🔥 warmup...", end=" ", flush=True)
        try:
            w = call_ds4(prompt, max_tok)
            print(f"done (total={w['total_s']}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            all_results["tasks"][task_name] = {"error": str(e)}
            continue

        # 正式测试
        runs = []
        for i in range(RUNS):
            try:
                r = call_ds4(prompt, max_tok)
                runs.append(r)
                # CLI 没有 ttft，所以不打印
                print(f"   Run {i+1}/{RUNS}: total={r['total_s']:.3f}s  "
                      f"tok/s={r['tok_per_s']}  out_chars={len(r['output'])}")
            except Exception as e:
                print(f"   Run {i+1}/{RUNS}: FAILED: {e}")
                runs.append({"error": str(e)})

        # 聚合统计（与 test_dgx.py 同结构）
        ok = [r for r in runs if "error" not in r]
        summary = {"runs": len(runs), "ok": len(ok), "samples": runs}
        if ok:
            def stat(key):
                xs = [r[key] for r in ok if r.get(key) is not None]
                if not xs:
                    return {"mean": None, "min": None, "max": None}
                return {
                    "mean": round(sum(xs) / len(xs), 3),
                    "min": round(min(xs), 3),
                    "max": round(max(xs), 3),
                }
            summary["ttft_s"] = {"mean": None, "min": None, "max": None}  # CLI 没有
            summary["total_s"] = stat("total_s")
            summary["tok_per_s"] = stat("tok_per_s")
            summary["prompt_tokens"] = None
            summary["completion_tokens"] = stat("completion_tokens")

        # 质量评估（复用 test_dgx 的 evaluator）
        if task_name in EVALUATORS and ok:
            eval_result = EVALUATORS[task_name](ok[-1]["output"], data)
            summary["quality"] = eval_result
            print(f"   📊 quality: {_short_quality(eval_result)}")

        all_results["tasks"][task_name] = summary

    # ============ 汇总表 ============
    print(f"\n{'=' * 70}")
    print(f"📊 汇总（{LLM_LABEL} / {os.path.basename(MODEL_PATH)}）")
    print(f"{'=' * 70}")
    print(f"{'task':<22}{'total(s)':<14}{'tok/s':<14}{'quality':<24}")
    print("-" * 88)
    for task_name, _, _ in TASKS:
        t = all_results["tasks"].get(task_name, {})
        if "total_s" not in t:
            print(f"{task_name:<22}{'FAILED':<14}{'-':<14}{'-':<24}")
            continue
        total = t["total_s"]["mean"]
        tps = t["tok_per_s"]["mean"]
        q = "-"
        if "quality" in t:
            qv = t["quality"]
            if "precision" in qv:
                q = f"P={qv.get('precision', 0)} R={qv.get('recall', 0)}"
            elif "scored" in qv:
                q = f"{qv['scored']}/{qv['max']}"
        print(f"{task_name:<22}{total if total else 0:<14}{tps if tps else 0:<14}{q:<24}")

    # ============ 综合评分 ============
    score = compute_overall_score(all_results)
    all_results["overall_score"] = score
    overall = score["overall"]
    sub = score["sub_scores"]
    verdict = "good" if overall >= 70 else ("fair" if overall >= 40 else "weak")
    print("\n" + "-" * 88)
    print(f"⭐ Overall: {overall:.1f}/100  ({verdict})")
    print(f"   quality={sub['quality']}  throughput={sub['throughput']}  "
          f"ttft={sub['ttft']}  stability={sub['stability']}")
    print(f"   weights: {T.SCORE_WEIGHTS}")

    # ============ 保存 JSON ============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_file = f"{OUTPUT_DIR}/bench_{LLM_LABEL}_{timestamp}.json"
    with open(bench_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 bench 结果: {bench_file}")

    # ============ 绘图（复用 test_dgx.plot_single_run）============
    if PLOT:
        try:
            png_file = f"{OUTPUT_DIR}/bench_{LLM_LABEL}_{timestamp}.png"
            plot_single_run(all_results, png_file)
            print(f"📊 可视化图: {png_file}")
        except Exception as e:
            print(f"⚠️  绘图失败: {type(e).__name__}: {e}")


def _short_quality(q):
    """单行打印 quality dict"""
    if "recall" in q:
        return f"P={q['precision']} R={q['recall']} ({q['hit_count']}/{q['truth_count']})"
    if "accuracy" in q:
        return f"acc={q['accuracy']} ({q['scored']}/{q['max']})"
    if "scored" in q:
        return f"{q['scored']}/{q['max']}"
    return str(q)[:80]


if __name__ == "__main__":
    main()
