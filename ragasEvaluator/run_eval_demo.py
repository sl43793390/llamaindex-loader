"""
run_eval_demo.py — RAGAS 端到端评测 Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
提供命令行入口，一键跑通 RAGAS 评测全流程。

使用方式
--------
::

    # 1) 使用内置样本文档（零外部依赖，不需要准备文件）
    python -m ragasEvaluator.run_eval_demo

    # 2) 指定自己的文档
    python -m ragasEvaluator.run_eval_demo --file data/sample.pdf

    # 3) 选择切分方法（sentence / agentic / semantic）
    python -m ragasEvaluator.run_eval_demo --splitter agentic

    # 4) 控制每 chunk 生成几个问答对、RAG 检索 top_k
    python -m ragasEvaluator.run_eval_demo --n-qa 2 --top-k 5

    # 5) 使用 chat engine（有 memory，适合多轮对话评测）
    python -m ragasEvaluator.run_eval_demo --use-chat-engine

    # 6) 将评测结果保存为 JSON 文件
    python -m ragasEvaluator.run_eval_demo --output result.json

完整参数说明
------------
    --file             文档路径（.pdf/.docx/.md/.txt 等），不指定则用内置样本
    --splitter         切分方法: sentence(默认) / agentic / semantic
    --n-qa             每 chunk 生成几个问答对（默认 1，推荐 1~3）
    --top-k            RAG 检索 top_k（默认 3）
    --use-chat-engine  使用 chat engine（有 memory）；默认用 query engine（无 memory）
    --output           将评测结果保存为 JSON 文件路径（可选）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import Document

from ragasEvaluator import (
    RAGASEvaluator,
    run_ragas_eval,
)


# ---------------------------------------------------------------------------
# 内置样本文档（无需外部文件即可跑通 demo）
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    "Milvus 是一个开源的向量数据库，支持亿级向量的毫秒级检索。它广泛应用于推荐系统、图像检索、自然语言处理等领域。",
    "BM25 是经典的信息检索算法，基于词频和文档长度归一化，常被用作稀疏检索的 baseline。",
    "LlamaIndex 是一个用于构建 RAG（检索增强生成）应用的框架，支持多种数据加载、索引、检索方式。",
    "FAISS 是 Facebook 开发的稠密向量 ANN 检索库，适合单机大规模场景。",
    "Hybrid Search 通常结合稠密向量检索和稀疏检索，通过 RRF 或加权融合提升召回质量。",
    "Milvus 2.x 支持多向量字段，可以同时存稠密向量和稀疏向量做混合检索。",
    "RRF（Reciprocal Rank Fusion）是一种无参数的检索结果融合算法，被 Cormack 等人在 2009 年提出。",
    "Agentic Chunking 使用 LLM 提取命题，然后用 Embedding 聚类成主题块，适合复杂文档切分。",
]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _load_docs(file_path: str | None) -> list[Document]:
    """
    加载文档：有 --file 则走 dataLoader.auto_load，否则用内置 sample。
    """
    if not file_path:
        print("[加载] 使用内置样本文档（8 条）")
        return [Document(text=t) for t in SAMPLE_DOCS]

    path = Path(file_path)
    if not path.exists():
        print(f"[错误] 文件不存在: {file_path}")
        sys.exit(1)

    from dataLoader import auto_load
    docs = auto_load(file_path)
    print(f"[加载] 从 {file_path} 加载了 {len(docs)} 个文档")
    return docs


def _get_splitter(name: str):
    """根据 --splitter 参数选择切分函数。"""
    splitters = {
        "sentence": ("spliter", "split_by_sentence"),
        "agentic":  ("advancedSplitter", "split_agentic"),
        "semantic": ("advancedSplitter", "split_semantic"),
    }
    if name not in splitters:
        raise ValueError(f"未知切分方法: {name}，可选: {list(splitters.keys())}")

    module_name, func_name = splitters[name]
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def _print_separator(title: str = "") -> None:
    """打印分隔线，提升可读性。"""
    width = 60
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print("-" * width)


def _print_results(result: dict) -> None:
    """格式化打印评测结果。"""
    n = result["n"]
    scores = result["scores"]
    eval_data = result["eval_data"]

    _print_separator("评测结果")

    if n == 0:
        print("  无有效样本 — 请检查 LLM / Embedding / Milvus 是否配置正确")
        return

    # 指标分数
    print(f"\n  有效样本数: {n}")
    print(f"\n  {'指标':<24} {'分数':>8}")
    print(f"  {'─' * 24} {'─' * 8}")
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<24} {v:>8.4f}")
        else:
            print(f"  {k:<24} {v}")

    # 前 3 条样本细节
    show_count = min(3, n)
    _print_separator(f"前 {show_count} 条样本详情")
    for i in range(show_count):
        q = eval_data["question"][i]
        gt = eval_data["ground_truth"][i]
        a = eval_data["answer"][i]
        ctxs = eval_data["contexts"][i]

        print(f"\n  [{i + 1}]")
        print(f"  问题:     {q}")
        print(f"  标准答案: {gt[:120]}{'...' if len(gt) > 120 else ''}")
        print(f"  RAG 回答: {a[:120]}{'...' if len(a) > 120 else ''}")
        if ctxs:
            print(f"  检索上下文: {len(ctxs)} 条, 首条 = {ctxs[0][:80]}...")
        else:
            print(f"  检索上下文: (空)")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS 端到端评测 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m ragasEvaluator.run_eval_demo                          # 内置样本
  python -m ragasEvaluator.run_eval_demo --file data/sample.pdf   # 自定义文档
  python -m ragasEvaluator.run_eval_demo --n-qa 2 --top-k 5      # 调参
  python -m ragasEvaluator.run_eval_demo --output result.json     # 保存结果
        """,
    )
    parser.add_argument("--file", type=str, default=None,
                        help="文档路径（.pdf/.docx/.md/.txt 等）")
    parser.add_argument("--splitter", type=str, default="sentence",
                        choices=["sentence", "agentic", "semantic"],
                        help="切分方法（默认: sentence）")
    parser.add_argument("--n-qa", type=int, default=1,
                        help="每 chunk 生成几个问答对（默认: 1）")
    parser.add_argument("--top-k", type=int, default=3,
                        help="RAG 检索 top_k（默认: 3）")
    parser.add_argument("--use-chat-engine", action="store_true",
                        help="使用 chat engine（有 memory）；默认 query engine（无 memory，推荐评测）")
    parser.add_argument("--output", type=str, default=None,
                        help="将评测结果保存为 JSON 文件路径")
    args = parser.parse_args()

    # UTF-8 输出
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    _print_separator("RAGAS 端到端评测")

    # ---- Step 1: 加载文档 ----
    print("\n[Step 1/4] 加载文档")
    docs = _load_docs(args.file)

    # ---- Step 2: 切分 ----
    print(f"\n[Step 2/4] 切分文档（方法: {args.splitter}）")
    split_fn = _get_splitter(args.splitter)
    nodes = split_fn(docs)
    print(f"  切分为 {len(nodes)} 个节点")

    # ---- Step 3: 执行评测 ----
    print(f"\n[Step 3/4] 执行评测（n_qa={args.n_qa}, top_k={args.top_k}, "
          f"engine={'chat' if args.use_chat_engine else 'query'}）")

    t0 = time.time()
    result = run_ragas_eval(
        docs=docs,
        nodes=nodes,
        split_fn=None,  # nodes 已切分好，跳过
        n_questions_per_chunk=args.n_qa,
        similarity_top_k=args.top_k,
        use_chat_engine=args.use_chat_engine,
        evaluator=RAGASEvaluator(),
    )
    elapsed = time.time() - t0
    print(f"  评测耗时: {elapsed:.1f}s")

    # ---- Step 4: 展示结果 ----
    _print_results(result)

    # ---- 可选: 保存结果到 JSON ----
    if args.output:
        # eval_data 中的 contexts 是 List[List[str]]，可直接序列化
        output_data = {
            "n": result["n"],
            "scores": result["scores"],
            "eval_data": result["eval_data"],
            "elapsed_seconds": round(elapsed, 2),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[保存] 结果已写入: {args.output}")

    print()


if __name__ == "__main__":
    main()
