"""跑 4 组检索消融实验,输出 Markdown 对比表。

四组实验(对应论文里的"两阶段检索优化"叙事):
  1. vector         - 纯向量(语义检索)
  2. bm25           - 纯 BM25(关键词检索)
  3. hybrid         - BM25 + 向量 RRF 融合
  4. hybrid_rerank  - 混合召回 top-20 → cross-encoder 精排 top-k(完整方案)

期望趋势:vector / bm25 ≈ baseline < hybrid < hybrid_rerank

用法:
    python eval/run_eval.py --dataset eval/dataset.json
    python eval/run_eval.py --dataset eval/dataset_zh.json --out eval/results_zh.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.reranker import rerank
from app.retriever import _get_bm25, hybrid_search
from app.vectorstore import get_store
from eval.metrics import evaluate


METHODS = ["vector", "bm25", "hybrid", "hybrid_rerank"]
METHOD_NAMES = {
    "vector": "纯向量",
    "bm25": "纯 BM25",
    "hybrid": "混合 RRF",
    "hybrid_rerank": "**混合+精排(完整方案)**",
}


def get_topk_sources(method: str, query: str, k: int) -> list[str]:
    """对单条 query 跑某个方法,返回 top-k 个 source(按相关度降序)。

    对 hybrid_rerank:先召回 20 个候选(保 recall),再 cross-encoder 精排取 top-k(保 precision)。
    其他方法直接拿 top-k。最后统一截断到 k(EnsembleRetriever 可能返回 > k 个)。
    """
    if method == "vector":
        retriever = get_store().as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
    elif method == "bm25":
        bm25 = _get_bm25(k=k)
        docs = bm25.invoke(query) if bm25 else []
    elif method == "hybrid":
        docs = hybrid_search(query, k=k)
    elif method == "hybrid_rerank":
        candidates = hybrid_search(query, k=20)
        docs = rerank(query, candidates, top_k=k) if candidates else []
    else:
        raise ValueError(f"未知方法: {method}")
    sources = [d.metadata.get("source", "") for d in docs if d.metadata.get("source")]
    return sources[:k]


def run_method(method: str, eval_set: list, ks: tuple) -> dict:
    max_k = max(ks)
    predictions = []
    t0 = time.time()
    for i, item in enumerate(eval_set, 1):
        preds = get_topk_sources(method, item["question"], k=max_k)
        predictions.append(preds)
        if i % 5 == 0 or i == len(eval_set):
            print(f"    [{method}] {i}/{len(eval_set)}", flush=True)
    elapsed = time.time() - t0
    return {
        "predictions": predictions,
        "metrics": evaluate(eval_set, predictions, ks=ks),
        "time_s": round(elapsed, 1),
    }


def format_md_table(results: dict, ks: tuple) -> str:
    headers = ["方案"] + [f"Recall@{k}" for k in ks] + ["MRR", "单题耗时(ms)"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "---|" * len(headers))
    for method in METHODS:
        m = results[method]["metrics"]
        avg_ms = round(results[method]["time_s"] / m["n"] * 1000, 1)
        cells = [METHOD_NAMES[method]]
        cells += [f"{m[f'recall@{k}']:.3f}" for k in ks]
        cells.append(f"{m['mrr']:.3f}")
        cells.append(f"{avg_ms}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="评测集 JSON 路径")
    parser.add_argument("--out", default="eval/results.md", help="Markdown 报告输出路径")
    parser.add_argument("--ks", default="4,20", help="Recall@k 的 k 值,逗号分隔(如 4,20)")
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    eval_set = json.load(open(args.dataset, encoding="utf-8"))
    print(f"[1/{len(METHODS) + 2}] 加载评测集: {len(eval_set)} 题 from {args.dataset}")

    results = {}
    for i, method in enumerate(METHODS, 2):
        print(f"\n[{i}/{len(METHODS) + 2}] 跑 {method}...")
        results[method] = run_method(method, eval_set, ks)
        m = results[method]["metrics"]
        ks_str = "  ".join(f"R@{k}={m[f'recall@{k}']:.3f}" for k in ks)
        print(f"  → {ks_str}  MRR={m['mrr']:.3f}  耗时 {results[method]['time_s']}s")

    print(f"\n[{len(METHODS) + 2}/{len(METHODS) + 2}] 写报告 → {args.out}")
    table = format_md_table(results, ks)

    dataset_name = Path(args.dataset).name
    report = f"""# RAG 检索消融实验结果

- **评测集**: `{dataset_name}` ({len(eval_set)} 题)
- **库规模**: 30 篇 arXiv 论文 / 3321 chunks / bge-m3 (1024 维)
- **指标说明**:
  - **Recall@k**:top-k 命中 `expected_sources` 任意一个即算命中(宽松匹配)
  - **MRR**:命中位置倒数的均值
  - **单题耗时**:平均每题端到端耗时(ms)

## 结果对比

{table}

## 解读

- **纯向量 / 纯 BM25**:作为单一检索基线,各有所长(向量擅长语义、BM25 擅长关键词)
- **混合 RRF**:取两路之长,通过 RRF 排名融合,理论上优于单路
- **混合+精排(完整方案)**:在召回 20 个候选基础上,用 cross-encoder 重新打分,
  把真正相关的文档顶到 top-4,这是项目实际部署的方案

期望趋势:`纯向量 / 纯 BM25 < 混合 RRF < 混合+精排`
"""
    Path(args.out).write_text(report, encoding="utf-8")

    # 同时保存原始 metrics(便于排查 / 后续作图)
    raw_out = args.out.replace(".md", "_raw.json")
    raw = {m: {"metrics": r["metrics"], "time_s": r["time_s"]} for m, r in results.items()}
    Path(raw_out).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 最终结果 ===")
    print(table)
    print(f"\n[OK] 报告 → {args.out}")
    print(f"     原始数据 → {raw_out}")


if __name__ == "__main__":
    main()
