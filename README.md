# 学术论文 RAG 智能问答助手

基于 RAG (Retrieval-Augmented Generation) 的学术论文问答系统:上传 PDF,自然语言提问,返回带引用来源的答案。**两阶段检索 + Cross-encoder 精排**,流式输出。

## ✨ 核心特性

- **混合检索**:BM25(关键词)+ 向量(语义)双路召回,RRF 融合
- **两阶段精排**:Cross-encoder (BGE-reranker-base) 对召回结果做二次打分,显著提升精度
- **可视化检索过程**:UI 内展开"精排前 vs 精排后"双列对比,清晰看到模型在干什么
- **流式输出**:DeepSeek 答案逐 token 渲染
- **本地 Embedding**:BGE-small-en-v1.5 本地推理,无需 embedding API

## 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| LLM | DeepSeek (OpenAI 兼容) | 代码层做了抽象,可平滑切换 |
| Embedding | BGE-small-en-v1.5 | 本地推理(~130MB) |
| 向量库 | Chroma | 持久化存储 |
| 召回 | BM25 + 向量,RRF 融合 | weights = [0.4, 0.6] |
| 精排 | BGE-reranker-base | Cross-encoder(~280MB) |
| 编排 | LangChain | |
| Web | Streamlit | |

## 🏗️ 架构

```
                        用户提问
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Stage 1: 混合检索召回 (k=20)  │
              │  ┌────────┐    ┌──────────┐ │
              │  │  BM25   │    │  Vector  │ │
              │  │ 关键词 │    │  语义    │ │
              │  └───┬────┘    └────┬─────┘ │
              │      └─ RRF 融合 ─┘         │
              └─────────────┬───────────────┘
                            │ 20 条候选
                            ▼
              ┌─────────────────────────────┐
              │ Stage 2: Cross-encoder 精排  │
              │  [query, doc] -> BERT -> 分数 │
              └─────────────┬───────────────┘
                            │ top-4
                            ▼
              ┌─────────────────────────────┐
              │  LLM 生成 (DeepSeek, 流式)   │
              └─────────────┬───────────────┘
                            ▼
                  带 [n] 引用的答案
```

## 🚀 快速开始

```bash
# 1. 创建并激活虚拟环境
py -3.11 -m venv .venv
.venv\Scripts\activate          # PowerShell / CMD
# source .venv/Scripts/activate # Git Bash

# 2. 安装依赖(国内建议加镜像)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置环境变量
copy .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 4. (可选)批量入库:把 PDF 放入 docs/,执行
python ingest.py

# 5. 启动 Web 界面
streamlit run main.py
```

首次启动会下载 BGE-embedding(~130MB) + BGE-reranker(~280MB),只需要一次。

## 📁 项目结构

```
.
├── app/
│   ├── llm.py           # DeepSeek 客户端(OpenAI 兼容封装)
│   ├── loader.py        # PDF 加载 + chunk 切分
│   ├── vectorstore.py   # Chroma + BGE-embedding,提供 list_sources()
│   ├── retriever.py     # BM25 + 向量混合检索,RRF 融合
│   ├── reranker.py      # Cross-encoder 精排(rerank / rerank_with_scores)
│   └── rag.py           # 问答链路:retrieve / retrieve_verbose / stream_answer
├── data/chroma/         # Chroma 持久化(.gitignore)
├── docs/                # 待入库 PDF(.gitignore)
├── ingest.py            # 批量入库脚本
├── main.py              # Streamlit 入口
├── requirements.txt
└── .env.example
```

## 💡 关键设计点

- **两阶段检索**:召回拉宽到 20 保 recall,精排紧到 4 保 precision。漏召回不可逆,但精排能修正排序。
- **BM25 缓存**:`retriever.py` 中 BM25 索引按 chunk 数量做失效判断,新论文入库才重建。
- **检索可视化**:`rag.retrieve_verbose()` 返回三阶段中间结果(`candidates / ranked / final`),UI 可展开看精排前后对比 —— 既是调试工具,也是面试演示利器。
- **依赖隔离**:`STEmbeddings` 直接基于 `sentence-transformers`,避开 `langchain-huggingface` 上游频繁变更带来的兼容性问题。
