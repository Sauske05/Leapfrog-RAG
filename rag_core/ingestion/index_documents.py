"""
Index job chunks into Qdrant with dense (BGE) + sparse (BM25) vectors.

Run after create_collection.py:

    python -m rag_core.ingestion.index_documents
    python -m rag_core.ingestion.index_documents --preview-sparse
"""

from __future__ import annotations

import argparse
import time
import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from rag_core.config import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
)
from rag_core.ingestion.preprocessing import (
    JobChunk,
    build_job_chunks,
    load_jobs_dataframe,
)


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=900,)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: JobChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "job_id": chunk.job_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "job_title": chunk.job_title,
        "company_name": chunk.company_name,
        "job_category": chunk.job_category,
        "job_level": chunk.job_level,
        "job_location": chunk.job_location,
        "publication_date": chunk.publication_date,
        "tags": chunk.tags,
    }


def _print_sparse_preview(text: str, indices: list[int], values: list[float]) -> None:
    print("\nSparse vector preview")
    print(f"Text: {text[:120]}...")
    print(f"Non-zero terms: {len(indices)}")
    print(f"indices[:12]: {indices[:12]}")
    print(f"values[:12]: {[round(v, 4) for v in values[:12]]}")


def index_chunks(
    client: QdrantClient | None = None,
    preview_sparse: bool = False,
    batch_size: int = BATCH_SIZE,
) -> None:
    client = client or get_client()

    print("Loading embedding models...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    print("Loading + chunking jobs...")
    df = load_jobs_dataframe()
    chunks = build_job_chunks(df)
    print(f"Indexing {len(chunks)} chunks in batches of {batch_size}...")

    points: list[PointStruct] = []
    for i, chunk in enumerate(chunks):
        text = chunk.text

        dense_vec = list(dense_model.embed([text]))[0].tolist()
        sparse_result = list(sparse_model.embed([text]))[0]
        sparse_vec = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        if preview_sparse and i == 0:
            _print_sparse_preview(text, list(sparse_vec.indices), list(sparse_vec.values))

        points.append(
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload=_payload(chunk),
            )
        )

        if len(points) >= batch_size:
            for attempt in range(3):
                try:
                    client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"Retry {attempt+1}: {e}")
                    time.sleep(2)
            #client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  Upserted {i + 1}/{len(chunks)}")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Done. {len(chunks)} chunks indexed into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview-sparse",
        action="store_true",
        help="Print sparse vector indices/values for the first chunk.",
    )
    args = parser.parse_args()
    index_chunks(preview_sparse=args.preview_sparse)