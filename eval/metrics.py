"""评测指标实现:Recall@k 与 MRR。

为什么这两个?
- 评测的是"检索系统能不能把正确论文排到前几位",不是"答案文字对不对"。
- Recall@k 直接告诉你"top-k 里有没有正确答案",是检索任务的"硬命中"指标。
- MRR 进一步看"正确答案排第几",考察排序质量。

约定:
- expected_sources 是 list(允许多篇论文都对),只要 predicted 里 top-k 命中
  其中任意一个即算命中(宽松匹配)。这与评测集构造时"q005 有 RAGAS+ARES
  两篇都正确"的设计对齐。
"""
from typing import List


def hit_at_k(predicted: List[str], expected: List[str], k: int) -> int:
    """top-k 个预测里命中 expected 任意一个 → 1,否则 0。"""
    expected_set = set(expected)
    return int(any(p in expected_set for p in predicted[:k]))


def reciprocal_rank(predicted: List[str], expected: List[str]) -> float:
    """第一个命中 expected 的位置 r → 1/r;没命中 → 0。"""
    expected_set = set(expected)
    for i, p in enumerate(predicted, start=1):
        if p in expected_set:
            return 1.0 / i
    return 0.0


def evaluate(
    eval_set: List[dict],
    predictions: List[List[str]],
    ks: tuple = (4, 20),
) -> dict:
    """对整个评测集做聚合,返回各指标的均值。

    eval_set:    [{"id", "question", "expected_sources", ...}, ...]
    predictions: 每条 query 的预测 source 列表(已按相关度降序)
    ks:          要计算 Recall@k 的 k 值列表
    返回:        {"recall@4": 0.85, "recall@20": 0.92, "mrr": 0.72, "n": 20}
    """
    n = len(eval_set)
    if n == 0:
        return {"n": 0}
    assert len(predictions) == n, "预测数量和评测集不匹配"

    out = {"n": n}
    for k in ks:
        hits = [hit_at_k(predictions[i], eval_set[i]["expected_sources"], k) for i in range(n)]
        out[f"recall@{k}"] = sum(hits) / n
    rrs = [reciprocal_rank(predictions[i], eval_set[i]["expected_sources"]) for i in range(n)]
    out["mrr"] = sum(rrs) / n
    return out
