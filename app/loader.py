"""PDF 加载与切分,产出带 metadata 的 Document 列表。"""
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: Union[str, Path]) -> List[Document]:
    """加载单个 PDF,每页一个 Document,metadata 含 source(文件名)与 page。"""
    path = Path(path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def load_dir(dir_path: Union[str, Path]) -> List[Document]:
    """批量加载目录下所有 PDF。"""
    dir_path = Path(dir_path)
    all_docs: List[Document] = []
    for pdf in sorted(dir_path.glob("*.pdf")):
        all_docs.extend(load_pdf(pdf))
    return all_docs


def split_documents(
    docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> List[Document]:
    """递归切分。800 字符是学术论文段落粒度的经验值。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
