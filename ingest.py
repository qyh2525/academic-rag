"""批量入库脚本:把 docs/ 目录下所有 PDF 切分并写入 Chroma。"""
import sys
from pathlib import Path

from app.loader import load_dir, split_documents
from app.vectorstore import add_documents


def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    pdfs = list(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[!] {docs_dir.resolve()} 目录下没有 PDF,请放入论文后再运行")
        sys.exit(1)

    print(f"[1/3] 加载 {len(pdfs)} 个 PDF…")
    raw_docs = load_dir(docs_dir)
    print(f"      共 {len(raw_docs)} 个页面")

    print("[2/3] 切分文本…")
    chunks = split_documents(raw_docs)
    print(f"      共 {len(chunks)} 个 chunk")

    print("[3/3] 向量化并写入 Chroma(首次运行需下载 embedding 模型,约 130MB)…")
    n = add_documents(chunks)
    print(f"      入库完成,新增 {n} 条向量")
    print("\n[OK] 可以运行 `streamlit run main.py` 开始问答了")


if __name__ == "__main__":
    main()
