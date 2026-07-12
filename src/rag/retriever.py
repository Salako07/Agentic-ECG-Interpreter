"""
RAG retriever: query the clinical-guideline vector store.

Thin, dependency-light interface over the ChromaDB collection built by
ingest.py. This is the surface the agent's `lookup_guidelines` tool calls:
given a diagnosis or free-text query, return the most relevant reference
passages with their source metadata for grounding and citation.

Run:  python -m src.rag.retriever      (runs a set of demo queries)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

STORE_DIR = os.path.join("data", "rag_store")
COLLECTION = "cardiology_reference"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Map classifier superclass codes to natural-language retrieval queries.
SUPERCLASS_QUERIES = {
    "NORM": "normal ECG sinus rhythm criteria",
    "MI": "myocardial infarction ST elevation Q waves lead territories",
    "STTC": "ST segment depression T wave inversion ischemia repolarization",
    "CD": "conduction disturbance bundle branch block AV block QRS widening",
    "HYP": "ventricular hypertrophy voltage criteria strain pattern",
}

# Map each superclass to the knowledge-base source(s) that should answer it.
# The interval reference is shared context relevant to several superclasses.
SUPERCLASS_SOURCES = {
    "NORM": ["normal_ecg", "ecg_intervals_reference"],
    "MI": ["myocardial_infarction", "ecg_intervals_reference"],
    "STTC": ["st_t_changes", "ecg_intervals_reference"],
    "CD": ["conduction_disturbance", "ecg_intervals_reference"],
    "HYP": ["hypertrophy", "ecg_intervals_reference"],
}


@dataclass
class Passage:
    text: str
    source: str
    section: str
    score: float   # cosine similarity in [0, 1], higher is closer

    def citation(self) -> str:
        return f"{self.source.replace('_', ' ').title()} — {self.section}"


class GuidelineRetriever:
    """Retrieves clinical reference passages from the vector store."""

    def __init__(self, store_dir: str = STORE_DIR, collection: str = COLLECTION):
        if not os.path.exists(store_dir):
            raise FileNotFoundError(
                f"Vector store not found at {store_dir}. Run ingest.py first."
            )

        client = chromadb.PersistentClient(path=store_dir)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.collection = client.get_collection(
            name=collection, embedding_function=embed_fn
        )

    def retrieve(self, query: str, k: int = 4, min_score: float = 0.30,
                 sources: list[str] | None = None) -> list[Passage]:
        """Return the top-k passages most relevant to the query.

        min_score: drop passages below this cosine similarity, so a query with
            no strong match returns fewer (or zero) passages rather than padding
            the result with weak, off-topic chunks. 0.30 rejects clear
            mismatches while keeping legitimate moderate matches (~0.45+).
        sources: if given, restrict retrieval to these source documents
            (metadata filter). Used to keep a superclass-specific query from
            pulling passages from unrelated diagnostic areas.
        """
        # Over-fetch generously: we filter by source and score in Python after
        # the query, which avoids depending on a specific ChromaDB `where`
        # operator syntax (the `$in` filter behaves differently across versions
        # and silently returned nothing on some). Post-filtering is robust and,
        # for a small corpus, has negligible cost.
        result = self.collection.query(
            query_texts=[query],
            n_results=self.collection.count(),  # fetch all, then filter locally
            include=["documents", "metadatas", "distances"],
        )

        source_set = set(sources) if sources else None
        passages: list[Passage] = []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            if source_set is not None and meta["source"] not in source_set:
                continue
            score = 1.0 - float(dist)   # cosine distance -> similarity
            if score < min_score:
                continue
            passages.append(Passage(
                text=doc,
                source=meta["source"],
                section=meta["section"],
                score=score,
            ))
            if len(passages) >= k:
                break

        return passages

    def retrieve_for_superclass(self, code: str, k: int = 3) -> list[Passage]:
        """Retrieve passages relevant to a predicted diagnostic superclass.

        Restricts retrieval to that superclass's own document (plus the shared
        interval reference), so a HYP query cannot return an MI passage.
        """
        query = SUPERCLASS_QUERIES.get(code, code)
        sources = SUPERCLASS_SOURCES.get(code)
        return self.retrieve(query, k=k, sources=sources)


def _demo() -> None:
    retriever = GuidelineRetriever()

    for code in ("MI", "CD"):
        print(f"\n=== retrieve_for_superclass('{code}') ===")
        for p in retriever.retrieve_for_superclass(code, k=3):
            print(f"  [{p.score:.3f}] {p.citation()}")

    print("\n=== free-text: 'why is the QRS wide' ===")
    for p in retriever.retrieve("why is the QRS complex wide", k=3):
        print(f"  [{p.score:.3f}] {p.citation()}")


if __name__ == "__main__":
    _demo()