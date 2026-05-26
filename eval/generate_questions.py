"""LLM 量产候选评测题:对库里每篇论文,喂前几个 chunk 给 DeepSeek 出 2 道题。

用法:
    python eval/generate_questions.py            # 出英文题
    python eval/generate_questions.py --zh       # 出中文题(用于多语言评测)
    python eval/generate_questions.py --resume   # 跳过 candidates.json 里已生成的 source

输出:
    eval/candidates.json           # 英文候选题
    eval/candidates_zh.json        # 中文候选题
"""
import argparse
import json
import sys
import time
from pathlib import Path

# 让 import app.xxx 能找到(脚本从仓库根目录跑)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import get_llm
from app.vectorstore import get_store, list_sources


PROMPT_EN = """You are an expert at constructing evaluation sets for academic RAG systems.

# Task
Based on the paper excerpts below, generate **exactly 2 questions**:
- 1 `factual` question (e.g., "What is the definition of X?", "How many components does X have?")
- 1 `method` question (e.g., "How does X work?", "What algorithm does X use to solve Y?")

# Strict Requirements
1. Questions must be in **natural language**; do NOT copy sentences verbatim from the excerpts.
2. The answer must be **clearly supported** by the given excerpts (do not fabricate).
3. Questions must NOT be too broad (avoid generic questions like "What is RAG?").
4. Do NOT ask about metadata (author, year, institution).
5. Output questions in **English**.

# Output Format (strict JSON, no extra text)
{{
  "questions": [
    {{
      "question": "...",
      "type": "factual",
      "answer_hint": "<short summary of the answer from the excerpt, for human review>"
    }},
    {{
      "question": "...",
      "type": "method",
      "answer_hint": "..."
    }}
  ]
}}

# Paper excerpts (from file: {source})
---
{chunks}
---
"""


PROMPT_ZH = """你是为学术论文 RAG 系统构造评测集的专家。

# 任务
基于下方给出的论文片段(英文),生成 **恰好 2 道中文问题**:
- 1 道 `factual`(事实定义类,例如"X 的定义是什么"、"X 有几个组件")
- 1 道 `method`(方法/原理类,例如"X 是怎么工作的"、"X 用什么算法解决 Y 问题")

# 严格要求
1. 问题用**自然中文**,不要照抄论文原句。
2. 答案必须在给定片段中有**明确依据**,不能编造。
3. 问题不要太宽泛(避免"什么是 RAG"这种通用问题)。
4. **不要**问元数据(作者、年份、机构)。
5. 问题用**中文**输出(注意:片段是英文,你需要理解后用中文重新表述)。

# 输出格式(严格 JSON,不要加任何解释文字)
{{
  "questions": [
    {{
      "question": "...",
      "type": "factual",
      "answer_hint": "<片段里对应答案的简短中文摘要,方便人工审核>"
    }},
    {{
      "question": "...",
      "type": "method",
      "answer_hint": "..."
    }}
  ]
}}

# 论文片段(来自文件: {source})
---
{chunks}
---
"""


def get_intro_chunks(source: str, n: int = 3) -> list[str]:
    """取某篇论文按 page 排序后的前 n 个 chunk(近似 abstract+intro)。"""
    store = get_store()
    data = store._collection.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    if not docs:
        return []
    # 按 page 排序;page 可能不存在或不是 int,容错一下
    paired = list(zip(docs, metas))
    paired.sort(key=lambda x: (x[1] or {}).get("page", 0))
    return [d for d, _ in paired[:n]]


def clean_json_response(text: str) -> str:
    """LLM 偶尔会把 JSON 包在 ```json ... ``` 里,清洗一下。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉首行 ``` 或 ```json
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        # 去掉末尾 ```
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def gen_questions_for_paper(source: str, llm, prompt_tmpl: str) -> list[dict]:
    chunks = get_intro_chunks(source, n=3)
    if not chunks:
        print(f"  [!] {source}: 没取到 chunk,跳过")
        return []

    prompt = prompt_tmpl.format(source=source, chunks="\n\n".join(chunks))
    try:
        resp = llm.invoke(prompt)
    except Exception as e:
        print(f"  [!] {source}: API 调用失败 - {e}")
        return []

    text = clean_json_response(resp.content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [!] {source}: JSON 解析失败,原文片段 = {text[:200]}")
        return []

    questions = data.get("questions", [])
    # 给每道题打上 expected_sources(就是当前这篇)
    for q in questions:
        q["expected_sources"] = [source]
        q["source_paper"] = source
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh", action="store_true", help="生成中文题(默认英文)")
    parser.add_argument("--resume", action="store_true",
                        help="跳过 candidates.json 里已有的 source(用于断点续跑)")
    args = parser.parse_args()

    prompt_tmpl = PROMPT_ZH if args.zh else PROMPT_EN
    out_path = Path("eval") / ("candidates_zh.json" if args.zh else "candidates.json")
    out_path.parent.mkdir(exist_ok=True)

    # 断点续跑:读已有结果
    existing: list[dict] = []
    done_sources: set[str] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done_sources = {q["source_paper"] for q in existing if "source_paper" in q}
        print(f"[resume] 已有 {len(existing)} 道题,覆盖 {len(done_sources)} 篇论文")

    sources = list_sources()
    todo = [s for s in sources if s not in done_sources]
    print(f"[1/?] 共 {len(sources)} 篇论文,本次待生成 {len(todo)} 篇")
    print(f"[2/?] 语言:{'中文' if args.zh else '英文'},输出:{out_path}")

    llm = get_llm(temperature=0.5)  # 温度高一点,题更多样

    all_questions = list(existing)
    for i, source in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {source}")
        questions = gen_questions_for_paper(source, llm, prompt_tmpl)
        all_questions.extend(questions)
        # 增量写盘:挂了重启也不丢
        out_path.write_text(
            json.dumps(all_questions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        time.sleep(1)  # 别把 API 打太狠

    print(f"\n[OK] 共 {len(all_questions)} 道候选题 → {out_path}")


if __name__ == "__main__":
    main()
