"""混合检索:BM25(关键词)+ 向量(语义)融合。

为什么需要这个文件?
- app/vectorstore.py 只做了纯向量检索,对"专有名词、缩写、数字"这类
  字面匹配场景不友好(例:查 "BERT-base" 可能丢失 "base" 这个关键 token)
- BM25 是经典关键词检索算法(TF-IDF 的改进版),正好补这个短板
- 两路结果用 EnsembleRetriever 融合,工业界标配方案
"""
from typing import List, Optional, Tuple

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.vectorstore import get_store


def _load_all_docs_from_chroma() -> List[Document]:
    """从 Chroma 把所有 chunk 拉出来,作为 BM25 的语料。

    设计取舍:
    - Chroma 是"真源",BM25 启动时从 Chroma 构建一次,之后缓存复用,
      仅当语料数量变化(新论文入库)时才失效重建。
    - 调的是 Chroma 私有属性 ._collection,这是 langchain-chroma 暴露底层
      chromadb client 的口子。返回的 dict 结构:
        {
          "ids": [...],
          "documents": ["chunk1 文本", "chunk2 文本", ...],
          "metadatas": [{"source": "xx.pdf", "page": 0}, ...],
        }
    """
    store = get_store()
    data = store._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


# (chunk_count, BM25Retriever) 缓存。chunk 数变化(新论文入库)即失效重建。
# 这样除非语料库变了,BM25 索引只在进程生命周期内构建一次。
_bm25_cache: Optional[Tuple[int, BM25Retriever]] = None


def _get_bm25(k: int) -> Optional[BM25Retriever]:
    """懒加载 + 缓存版 BM25。chroma.count() 是 O(1) 操作,做失效判断很便宜。"""
    global _bm25_cache
    store = get_store()
    count = store._collection.count()
    if count == 0:
        return None
    if _bm25_cache is None or _bm25_cache[0] != count:
        # 首次构建,或语料数量变了 -> 重建
        all_docs = _load_all_docs_from_chroma()
        bm25 = BM25Retriever.from_documents(all_docs)
        _bm25_cache = (count, bm25)
    bm25 = _bm25_cache[1]
    bm25.k = k  # k 可以每次调用变化,所以不进缓存 key
    return bm25


def build_hybrid_retriever(k: int = 4) -> Optional[BaseRetriever]:
    """构建混合检索器。返回 None 表示知识库为空。

    参数 k:每一路(BM25、向量)各自返回多少候选,融合后总数不会超过 2k 去重后的量。
    """
    bm25 = _get_bm25(k=k)
    if bm25 is None:
        return None

    # 向量检索器:复用已有 Chroma,擅长语义匹配
    vector_retriever = get_store().as_retriever(search_kwargs={"k": k})

    # 融合:EnsembleRetriever 内部用 RRF (Reciprocal Rank Fusion) 算法
    #    —— 不依赖原始分数,只看排名,把两路 rank 倒数加权求和后重新排序。
    #    weights 经验值:向量 0.6 / BM25 0.4(向量召回更稳,权重高一点)
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6],
    )


def hybrid_search(question: str, k: int = 4) -> List[Document]:
    """对外接口:输入 query,输出 top-k 融合后的文档。

    rag.py 只需要调这一个函数,不必关心 BM25 / 向量 / 融合的细节。
    这就是 LangChain Retriever 接口统一带来的好处:换实现不改调用方。
    """
    retriever = build_hybrid_retriever(k=k)
    if retriever is None:
        return []
    return retriever.invoke(question)
