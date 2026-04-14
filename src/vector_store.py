from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import uuid

import chromadb
from chromadb.config import Settings


class VectorStore:
    """Chroma-backed vector store wrapper.

    This class wraps a `chromadb` collection. Pass `persist_directory` to enable
    on-disk persistence (uses the Chroma client Settings). You can also supply
    an existing `chromadb.Client` via `client`.

    Methods:
        - `add(docs, embeddings)` — add documents and embeddings to the collection
        - `search(query_embedding, k)` — KNN search returning documents + scores + metadata
        - `persist()` — flush client to disk (if persistent client used)
        - `delete_collection()` — remove the collection
    """

    def __init__(self, collection_name: str = "default", persist_directory: Optional[str] = None, client: Optional[object] = None):
        if client is not None:
            self.client = client
        else:
            if persist_directory:
                settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
                self.client = chromadb.Client(settings)
            else:
                # in-memory client
                self.client = chromadb.Client()

        # get or create collection
        try:
            # newer chroma versions support get_or_create_collection
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception:
            # fallback to create_collection (will raise if exists)
            try:
                self.collection = self.client.create_collection(name=collection_name)
            except Exception:
                # last resort: try get_collection
                self.collection = self.client.get_collection(name=collection_name)

    def add(self, docs: List[Document], embeddings: List[List[float]]) -> List[str]:
        """Add documents and embeddings to Chroma collection.

        Returns list of generated ids for the added vectors.
        """
        if len(docs) != len(embeddings):
            raise ValueError("`docs` and `embeddings` must have the same length")

        ids: List[str] = [str(uuid.uuid4()) for _ in docs]
        texts: List[str] = [getattr(d, "page_content", None) or getattr(d, "content", "") for d in docs]
        metadatas: List[Dict[str, Any]] = [getattr(d, "metadata", None) or {} for d in docs]

        # chroma expects embeddings as lists of floats
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return ids

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Query the collection using a query embedding.

        Returns list of dicts: {"id","document","metadata","score"}
        """
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k, include=["metadatas", "documents", "distances", "ids"]) 
        results: List[Dict[str, Any]] = []
        # results are nested lists per query; we queried a single input at index 0
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        for i in range(len(ids)):
            results.append({"id": ids[i], "document": docs[i], "metadata": metadatas[i], "score": float(distances[i])})

        return results

    def persist(self) -> None:
        """Persist client state to disk if supported by the client."""
        if hasattr(self.client, "persist"):
            try:
                self.client.persist()
            except Exception:
                pass

    def delete_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection.name)
        except Exception:
            # fallback: ignore
            pass
