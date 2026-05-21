"""向量化与 Chroma 持久化存储。

embedding 用 sentence-transformers 本地推理(BGE 系列),避免再申请 embedding API。
首次运行会自动下载模型(约 130MB),之后离线可用。
"""
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()


class STEmbeddings(Embeddings):
    """sentence-transformers 的 LangChain Embeddings 适配器。

    直接基于 sentence-transformers 实现,而非依赖 langchain-huggingface,
    减少包依赖、避免 LangChain 上游包结构频繁变更带来的兼容性问题。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        # 懒加载,避免 import 时就触发模型下载
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


_embeddings: Optional[STEmbeddings] = None


def get_embeddings() -> STEmbeddings:
    global _embeddings
    if _embeddings is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        _embeddings = STEmbeddings(model_name)
    return _embeddings


def get_store(persist_directory: Optional[str] = None) -> Chroma:
    persist_directory = persist_directory or os.getenv("CHROMA_DIR", "./data/chroma")
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
        collection_name="papers",
    )


def add_documents(docs: List[Document]) -> int:
    store = get_store()
    store.add_documents(docs)
    return len(docs)


def list_sources() -> List[str]:
    """返回当前知识库中所有已入库论文的 source 列表(去重、排序)。

    用途:Streamlit sidebar 展示"已入库论文",让用户知道库里有什么。
    只拉 metadata、不走相似度检索,因此很轻量。
    """
    store = get_store()
    data = store._collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    sources = {m.get("source") for m in metadatas if m and m.get("source")}
    return sorted(sources)
