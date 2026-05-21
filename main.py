"""Streamlit Web 界面:聊天式 RAG 问答 + 侧边栏上传入库。"""
from pathlib import Path

import streamlit as st

from app.loader import load_pdf, split_documents
from app.rag import retrieve_verbose, stream_answer
from app.reranker import get_reranker
from app.vectorstore import add_documents, list_sources

st.set_page_config(page_title="论文 RAG 问答", page_icon=":books:", layout="wide")
st.title("学术论文 RAG 智能问答")
st.caption("基于 DeepSeek + Chroma + BGE Embedding")

# 启动预热:首次访问时同步加载 rerank 模型,避免第一次提问卡数十秒。
if "warmed_up" not in st.session_state:
    with st.spinner("首次启动:加载 rerank 模型(BGE-reranker-base, ~280MB),只需要等这一次…"):
        get_reranker()
    st.session_state.warmed_up = True
    st.toast("模型已就绪,可以开始提问 ✅", icon="🎉")

EXAMPLE_QUESTIONS = [
    "这些论文的核心贡献分别是什么?",
    "Transformer 中 self-attention 的计算复杂度是多少?",
    "BERT 的预训练任务有哪些?",
    "对比一下不同论文中使用的评测数据集",
]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("上传论文")
    uploaded = st.file_uploader(
        "选择 PDF(可多选)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded and st.button("入库", type="primary", use_container_width=True):
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        progress = st.progress(0.0, text="开始处理…")
        total_chunks = 0
        for i, f in enumerate(uploaded, 1):
            progress.progress(
                (i - 1) / len(uploaded), text=f"处理 {f.name} ({i}/{len(uploaded)})"
            )
            save_path = docs_dir / f.name
            save_path.write_bytes(f.getbuffer())
            raw = load_pdf(save_path)
            chunks = split_documents(raw)
            add_documents(chunks)
            total_chunks += len(chunks)
        progress.progress(1.0, text="完成")
        st.success(f"已入库 {len(uploaded)} 篇论文,共 {total_chunks} 个 chunk")

    st.divider()

    # 已入库论文列表(任务 4)
    st.subheader("📚 知识库")
    sources = list_sources()
    if sources:
        st.caption(f"已入库 {len(sources)} 篇:")
        for s in sources:
            st.markdown(f"- `{s}`")
    else:
        st.info("知识库为空,请先上传 PDF 并入库")

    st.divider()

    # 清空对话(任务 4)
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "**使用步骤**\n"
        "1. 复制 `.env.example` 为 `.env` 并填入 `DEEPSEEK_API_KEY`\n"
        "2. 上传论文 PDF 并点击「入库」\n"
        "3. 在右侧对话框提问"
    )


# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(docs):
    with st.expander(f"📑 引用来源({len(docs)} 条)"):
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            st.markdown(f"**[{i}] {src} — 第 {page} 页**")
            snippet = d.page_content[:300].replace("\n", " ")
            st.text(snippet + ("…" if len(d.page_content) > 300 else ""))


def _render_retrieval_trace(trace):
    """展开检索过程:召回 -> 精排前 vs 精排后 对比(任务 7,面试核心亮点)。"""
    candidates = trace["candidates"]
    ranked = trace["ranked"]

    with st.expander(f"🔍 检索过程(召回 {len(candidates)} 条 → 精排 top {len(ranked)})"):
        st.markdown("##### 两阶段检索流水线")
        st.caption(
            "Stage 1:BM25 + 向量混合检索(RRF 融合)拉宽候选;"
            "Stage 2:Cross-encoder 精排,选出真正相关的 top-k 送给 LLM。"
        )

        col1, col2 = st.columns(2)

        # 左侧:精排前(召回阶段排名)
        with col1:
            st.markdown(f"**📥 精排前(召回 top {min(len(candidates), len(ranked))})**")
            st.caption("混合检索原始顺序")
            for i, d in enumerate(candidates[: len(ranked)], 1):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", "?")
                st.markdown(f"`#{i}` **{src}** p.{page}")
                snippet = d.page_content[:120].replace("\n", " ")
                st.caption(snippet + "…")

        # 右侧:精排后(cross-encoder 打分后)
        with col2:
            st.markdown(f"**📤 精排后(top {len(ranked)})**")
            st.caption("Cross-encoder 重排序")
            for i, (score, d) in enumerate(ranked, 1):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", "?")
                st.markdown(f"`#{i}` **{src}** p.{page} · score=`{score:.3f}`")
                snippet = d.page_content[:120].replace("\n", " ")
                st.caption(snippet + "…")


# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            _render_retrieval_trace(msg["trace"])
        if msg.get("sources"):
            _render_sources(msg["sources"])


# ---------------- Empty state: 示例问题(任务 5)----------------
if not st.session_state.messages:
    st.markdown("##### 💡 试试这些问题:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()


# ---------------- Chat input ----------------
question = st.chat_input("向论文提问…")
# 示例问题点击后通过 session_state 注入
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # st.status 三阶段反馈(任务 6)
        with st.status("正在思考…", expanded=False) as status:
            st.write("🔎 Stage 1:混合检索召回候选段落…")
            trace = retrieve_verbose(question, k=4)
            st.write(
                f"   召回 {len(trace['candidates'])} 条候选(BM25 + 向量,RRF 融合)"
            )

            st.write("🎯 Stage 2:Cross-encoder 精排选出 top-k…")
            st.write(f"   精排选中 {len(trace['ranked'])} 条")

            if not trace["final"]:
                status.update(label="知识库为空", state="error", expanded=False)
            else:
                st.write("✍️ Stage 3:调用 DeepSeek 生成答案…")
                status.update(label="检索完成,正在生成…", state="running", expanded=False)

        docs = trace["final"]

        if not docs:
            text = "知识库为空,请先在左侧上传论文并入库。"
            st.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
        else:
            placeholder = st.empty()
            text = ""
            try:
                for chunk in stream_answer(question, docs):
                    text += chunk
                    placeholder.markdown(text + "▌")
                placeholder.markdown(text)
            except Exception as e:
                placeholder.error(f"调用大模型失败:{e}")
                text = f"[错误] {e}"
            _render_retrieval_trace(trace)
            _render_sources(docs)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "sources": docs,
                    "trace": trace,
                }
            )
