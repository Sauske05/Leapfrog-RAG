"""
Smoke-test dense / sparse / hybrid retrieval against the lf_jobs collection.

Requires:
  - Qdrant running at QDRANT_URL
  - Collection already created + indexed

Usage:
    python scripts/test_retrieve.py
    python scripts/test_retrieve.py --query "senior remote python engineer" --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from rag_core.config import (
    COLLECTION_NAME,
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    PREFETCH_LIMIT,
    QDRANT_API_KEY,
    QDRANT_URL,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
)


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def embed_query(query: str, dense_model: TextEmbedding, sparse_model: SparseTextEmbedding):
    dense_vec = list(dense_model.embed([query]))[0].tolist()
    sparse_raw = list(sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )
    return dense_vec, sparse_vec


def _format_hit(hit) -> dict:
    p = hit.payload or {}
    return {
        "score": round(float(hit.score or 0.0), 4),
        "chunk_id": p.get("chunk_id", str(hit.id)),
        "job_title": p.get("job_title", ""),
        "company_name": p.get("company_name", ""),
        "job_location": p.get("job_location", ""),
        "job_level": p.get("job_level", ""),
        "text_preview": (p.get("text") or "")[:200].replace("\n", " "),
    }


def search_dense(client: QdrantClient, dense_vec: list[float], top_k: int) -> list[dict]:
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    ).points
    return [_format_hit(h) for h in hits]


def search_sparse(client: QdrantClient, sparse_vec: SparseVector, top_k: int) -> list[dict]:
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vec,
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    ).points
    return [_format_hit(h) for h in hits]


def search_hybrid(
    client: QdrantClient,
    dense_vec: list[float],
    sparse_vec: SparseVector,
    top_k: int,
    prefetch_limit: int = PREFETCH_LIMIT,
) -> list[dict]:
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points
    return [_format_hit(h) for h in hits]


def print_results(title: str, results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{title}  ({len(results)} hits)")
    print("=" * 60)
    if not results:
        print("  (no hits)")
        return
    for i, r in enumerate(results, 1):
        print(
            f"{i}. score={r['score']:.4f} | {r['job_title']} @ {r['company_name']} "
            f"({r['job_location']}, {r['job_level']})"
        )
        print(f"   chunk_id={r['chunk_id']}")
        print(f"   {r['text_preview']}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test dense/sparse/hybrid retrieval.")
    parser.add_argument(
        "--query",
        default="kjdnjkafjksdfbjkasbjk",
        help="Natural-language search query",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per mode")
    args = parser.parse_args()

    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found at {QDRANT_URL}. "
            "Run create_collection + index_documents first."
        )

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection: {COLLECTION_NAME}  points={info.points_count}")
    print(f"Query: {args.query!r}  top_k={args.top_k}")

    print("Loading embedding models...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    dense_vec, sparse_vec = embed_query(args.query, dense_model, sparse_model)
    print(
        f"Dense dim={len(dense_vec)} | "
        f"Sparse non-zeros={len(sparse_vec.indices)}"
    )

    dense_hits = search_dense(client, dense_vec, args.top_k)
    sparse_hits = search_sparse(client, sparse_vec, args.top_k)
    hybrid_hits = search_hybrid(client, dense_vec, sparse_vec, args.top_k)

    print_results("DENSE (BGE cosine)", dense_hits)
    print_results("SPARSE (BM25)", sparse_hits)
    print_results("HYBRID (RRF dense + sparse)", hybrid_hits)


if __name__ == "__main__":
    main()