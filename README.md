# 学术论文 RAG 智能问答助手

基于 RAG (Retrieval-Augmented Generation) 的学术论文问答系统:上传 PDF,自然语言提问,返回带引用来源的答案。**两阶段检索 + Cross-encoder 精排**,流式输出。

## ✨ 核心特性

- **混合检索**:BM25(关键词)+ 向量(语义)双路召回,RRF 融合
- **两阶段精排**:Cross-encoder (BGE-reranker-base) 对召回结果做二次打分,显著提升精度
- **多语言支持**:Embedding 使用 BAAI/bge-m3,支持**中文 query 检索英文论文**
- **可视化检索过程**:UI 内展开"精排前 vs 精排后"双列对比,清晰看到模型在干什么
- **流式输出**:DeepSeek 答案逐 token 渲染
- **检索效果评估**:中英对照消融实验(详见 [`eval/SUMMARY.md`](eval/SUMMARY.md))

## 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| LLM | DeepSeek (OpenAI 兼容) | 代码层做了抽象,可平滑切换 |
| Embedding | BAAI/bge-m3 | 本地推理(~2.2GB),多语言支持 |
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
# 国内用户建议设置 HF_ENDPOINT=https://hf-mirror.com,加速 BGE 模型下载

# 4. (可选)批量入库:把 PDF 放入 docs/,执行
python ingest.py

# 5. 启动 Web 界面
streamlit run main.py
```

首次启动会下载 BGE-m3 embedding(~2.2GB) + BGE-reranker(~280MB),只需要一次。

## 📁 项目结构

```
.
├── app/
│   ├── __init__.py      # 统一加载 .env + HF 镜像配置
│   ├── llm.py           # DeepSeek 客户端(OpenAI 兼容封装)
│   ├── loader.py        # PDF 加载 + chunk 切分
│   ├── vectorstore.py   # Chroma + BGE-embedding,提供 list_sources()
│   ├── retriever.py     # BM25 + 向量混合检索,RRF 融合
│   ├── reranker.py      # Cross-encoder 精排(rerank / rerank_with_scores)
│   └── rag.py           # 问答链路:retrieve / retrieve_verbose / stream_answer
├── eval/                # 检索消融实验(见 SUMMARY.md)
│   ├── generate_questions.py  # LLM 批量生成候选评测题
│   ├── metrics.py             # Recall@k + MRR 实现
│   ├── run_eval.py            # 4 组消融评测主脚本
│   ├── dataset.json / dataset_zh.json  # 20 题中英评测集(手筛)
│   ├── results.md / results_zh.md      # 评测结果
│   └── SUMMARY.md             # 总结报告(关键发现)
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

## 📊 检索效果评估

基于 30 篇 arXiv RAG 论文构建评测集(中英各 20 题),对 4 组方案做消融对比:

| 方案 | 英文 R@4 / MRR | 中文 R@4 / MRR |
|---|---|---|
| 纯 BM25 | 0.900 / 0.818 | **0.250 / 0.250** 💥 |
| 纯向量 | 1.000 / 0.975 | 1.000 / 0.942 |
| 混合 RRF | 1.000 / **1.000** | 1.000 / 0.867 |
| **混合+精排(完整方案)** | 1.000 / 0.975 | 1.000 / **0.967** |

通过中英对照评测发现三个非平凡现象:
- **BM25 跨语言失效**:中文 query 检索英文文档时 R@4 从 0.9 暴跌至 0.25
- **RRF 融合的副作用**:某一路完全失效时,RRF 会被无意义排名拖累(中文 MRR 0.867 < 纯向量 0.942)
- **精排价值的场景依赖**:英文召回完美时精排是负优化(-0.025),中文 BM25 失效时精排是关键救场(+0.100)

完整方法学、实验设计和工程启示见 [`eval/SUMMARY.md`](eval/SUMMARY.md)。
