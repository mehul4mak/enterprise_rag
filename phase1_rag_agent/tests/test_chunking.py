from src.ingest import chunk_pages


def test_chunk_ids_are_unique_and_page_tagged():
    pages = ["line one\nline two\n" * 5, "page two content\n" * 5]
    chunks = chunk_pages(pages, chunk_size=50, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.page in (1, 2) for c in chunks)


def test_empty_page_produces_no_chunks():
    pages = ["real content here", "   ", ""]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert all(c.page == 1 for c in chunks)


def test_citation_format():
    pages = ["hello world"]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert chunks[0].citation == "[p1:c1]"
