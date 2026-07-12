"""
RAG ingestion: build the clinical-guideline vector store.

Chunks the markdown knowledge base by section, embeds each chunk with a local
sentence-transformers model, and stores vectors + metadata in a persistent
ChromaDB collection. Idempotent: re-running rebuilds the collection from
scratch so edits to the knowledge base are always reflected.

Corpus note: the knowledge base under src/rag/knowledge_base/ is original,
author-written cardiology reference text covering the five PTB-XL diagnostic
superclasses. It is educational reference material, not a substitute for
current clinical guidelines, and deliberately contains no copyrighted
guideline text so the repository can be shared freely. Expand only with
original prose or genuinely open-access sources.

Run:  python -m src.rag.ingest
"""

from __future__ import annotations

import glob
import os
import re

import chromadb
from chromadb.utils import embedding_functions

KB_DIR = os.path.join("src", "rag", "knowledge_base")
STORE_DIR = os.path.join("data", "rag_store")
COLLECTION = "cardiology_reference"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, CPU-friendly


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (section_title, section_text) by ## headings."""
    lines = markdown.splitlines()

    # Document title from the first '# ' line, if present.
    doc_title = ""
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break

    sections: list[tuple[str, str]] = []
    current_title = doc_title or "Overview"
    current_body: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
                current_body = []
            current_title = line[3:].strip()
        elif line.startswith("# "):
            continue  # doc title, already captured
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))

    # Prefix each chunk with the document title for retrieval context.
    return [(title, body) for title, body in sections if body]


def load_chunks() -> list[dict]:
    """Load and chunk every markdown file in the knowledge base."""
    paths = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {KB_DIR}")

    chunks: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        source = os.path.splitext(os.path.basename(path))[0]
        doc_title = source.replace("_", " ").title()

        for section_title, body in _split_sections(text):
            chunks.append({
                "id": f"{source}::{section_title}".replace(" ", "_"),
                "text": f"{doc_title} — {section_title}\n\n{body}",
                "source": source,
                "section": section_title,
            })

    return chunks


def build() -> None:
    os.makedirs(STORE_DIR, exist_ok=True)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {KB_DIR}")

    client = chromadb.PersistentClient(path=STORE_DIR)

    # Fresh build each time so KB edits are reflected.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"]}
                   for c in chunks],
    )

    print(f"Built collection '{COLLECTION}' with {collection.count()} vectors")
    print(f"Store location: {os.path.abspath(STORE_DIR)}")


if __name__ == "__main__":
    build()