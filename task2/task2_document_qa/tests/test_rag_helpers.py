from pathlib import Path
from langchain_core.documents import Document
from rag_pipeline import split_documents, list_supported_documents


def test_split_documents_preserves_metadata():
    docs = [Document(
        page_content="This is a test sentence. " * 100,
        metadata={"source": "sample.pdf", "page": 2},
    )]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "sample.pdf" for c in chunks)
    assert all(c.metadata["page"] == 2 for c in chunks)


def test_list_supported_documents_filters_extensions(tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.txt").write_text("text", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"")
    assert list_supported_documents(tmp_path) == ["a.pdf", "b.txt"]
