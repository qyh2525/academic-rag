"""RAG 问答核心链路:检索 -> 拼 prompt -> 生成。"""
from typing import Iterator, List, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm
from app.reranker import rerank, rerank_with_scores
from app.retriever import hybrid_search

# 召回阶段拉宽,给精排留挑选空间。漏召回不可逆,宁多勿少。
RECALL_K = 20

SYSTEM_PROMPT = """你是一个严谨的学术论文问答助手。请基于下面提供的"参考资料"回答用户问题。

要求:
1. 只使用参考资料中的信息回答,不要编造、不要使用资料外的知识。
2. 如果参考资料无法回答该问题,直接说"根据已有论文无法回答该问题"。
3. 在答案中需要引用的位置用方括号标注资料编号,如 [1] [3]。
4. 使用中文回答,语言简练专业。

参考资料:
{context}
"""


class RetrieveTrace(TypedDict):
    """retrieve_verbose 的返回结构,给 UI 用的检索过程快照。"""

    candidates: List[Document]               # 混合检索召回的全部候选(BM25+向量融合后)
    ranked: List[Tuple[float, Document]]     # cross-encoder 精排后带分数的 top_k
    final: List[Document]                    # 真正送进 LLM 的 docs(= [d for _, d in ranked])


def format_context(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        lines.append(f"[{i}] (来源: {src}, 第 {page} 页)\n{d.page_content}")
    return "\n\n".join(lines)


def retrieve(question: str, k: int = 4) -> List[Document]:
    # 两阶段检索:hybrid 召回宽(保 recall) -> cross-encoder 精排紧(保 precision)
    candidates = hybrid_search(question, k=RECALL_K)
    return rerank(question, candidates, top_k=k)


def retrieve_verbose(question: str, k: int = 4) -> RetrieveTrace:
    """检索的"开盖版":同时返回召回候选、精排带分结果、最终 docs。

    用途:Streamlit UI 把整个检索过程展开给用户看,是"模型在干活"的可视化证据,
          也是面试演示时讲两阶段检索的最佳载体。
    """
    candidates = hybrid_search(question, k=RECALL_K)
    ranked = rerank_with_scores(question, candidates, top_k=k)
    final = [d for _, d in ranked]
    return {"candidates": candidates, "ranked": ranked, "final": final}


def _build_chain(streaming: bool = False):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", "{question}")]
    )
    return prompt | get_llm(streaming=streaming) | StrOutputParser()


def answer(question: str, k: int = 4) -> Tuple[str, List[Document]]:
    """同步版:返回 (答案, 引用文档)。"""
    docs = retrieve(question, k=k)
    if not docs:
        return "知识库为空,请先用 ingest.py 入库论文。", []
    chain = _build_chain(streaming=False)
    response = chain.invoke({"context": format_context(docs), "question": question})
    return response, docs


def stream_answer(question: str, docs: List[Document]) -> Iterator[str]:
    """流式版:调用方先用 retrieve 拿 docs,再用这个生成器拉取答案 chunk。"""
    chain = _build_chain(streaming=True)
    yield from chain.stream(
        {"context": format_context(docs), "question": question}
    )
