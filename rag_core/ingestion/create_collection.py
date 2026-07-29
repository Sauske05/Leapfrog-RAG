"""
Create a Qdrant collection with named dense + sparse vector spaces.
Run once before indexing.

    python -m rag_core.ingestion.create_collection
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from rag_core.config import (
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection(client: QdrantClient | None = None) -> None:
    client = client or get_client()

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,  # in-memory sparse index → lower latency
                ),
                modifier=Modifier.IDF,  # server-side IDF as collection grows
            )
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

    # Keyword payload indexes for filtered search later
    for field in ("job_category", "job_level", "job_location", "company_name"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema="keyword",
        )
        print(f"Payload index created on '{field}'.")


if __name__ == "__main__":
    create_collection()