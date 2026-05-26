"""Cross-encoder 精排:对召回结果做二次打分,选出最相关的 top_k。

为什么需要这个文件?
- retriever.py 用的是 bi-encoder 架构(query 和 doc 各自独立编码成向量),
  扛大规模快,但 query 和 doc 之间没有 token 级交互,精度有上限。
- Cross-encoder 把 [query, doc] 拼成一个序列喂进同一个 BERT,让 self-attention
  在两者之间充分交互,输出一个标量相关性分数。慢,但精度高得多。
- 两阶段流水线:retriever 召回 top-20(保 recall) -> reranker 精排 top-4(保 precision)。
"""
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# HF 镜像必须在 import sentence_transformers 之前生效
load_dotenv()
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """懒加载 cross-encoder 单例。首次调用会下载模型(BGE-reranker-base ~280MB)。

    用 sentence-transformers 的 CrossEncoder 封装:
    - 加载:CrossEncoder("BAAI/bge-reranker-base")
    - 打分:model.predict([(query, doc1), (query, doc2), ...]) -> [score1, score2, ...]
    """
    global _reranker
    if _reranker is None:
        model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank_with_scores(
    question: str, docs: List[Document], top_k: int = 4
) -> List[Tuple[float, Document]]:
    """对召回结果做精排,返回 [(score, doc), ...] 形式的 top_k(按分数降序)。

    与 rerank() 的区别:本函数把分数透传给上游,用于 UI 可视化(展示精排前后差异)。
    LLM 生成阶段用不到分数,但调试/演示时分数是很有说服力的"模型在干活"的证据。

    边界:
        - docs 为空,返回 []
        - docs 数量 <= top_k,跳过模型调用,用 0.0 占位分数(保证返回结构一致)
    """
    if not docs:
        return []
    if len(docs) <= top_k:
        return [(0.0, d) for d in docs]

    model = get_reranker()
    pairs = [[question, d.page_content] for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    return [(float(s), d) for s, d in ranked[:top_k]]


def rerank(question: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """对召回结果做精排,返回 top_k 个最相关的文档(不带分数,给 LLM 用)。"""
    return [d for _, d in rerank_with_scores(question, docs, top_k=top_k)]
