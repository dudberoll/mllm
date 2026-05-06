from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from embedding import SentenceEmbedder
from preprocessing import Preprocessor
from sentence_transformers import CrossEncoder


class VectorDB:
    """Vector database wrapper."""

    def __init__(
        self,
        path2data: str | Path,
        loaders: Mapping[str, Any],
        chunk_length: int,
        chunk_overlap: int,
        embedder: SentenceEmbedder | None = None,
        db_path: str | Path | None = None,
        collection_name: str = "Embeddings",
    ) -> None:
        self.path2data = Path(path2data)
        self.loaders = dict(loaders)
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.db_path = Path(db_path) if db_path else self.path2data.parent / "db"
        self.collection_name = collection_name
        self.sentence_embedder = embedder or SentenceEmbedder()
        self.cross_encoder = None
        self.preprocessor = Preprocessor(
            self.path2data,
            self.loaders,
            self.chunk_length,
            self.chunk_overlap,
        )
        self._start_db()

    def _start_db(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def _reset_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def preprocess_data(self) -> list[Document]:
        return self.preprocessor.preprocess_data()

    def get_embeddings(self, chunks: list[Document]) -> np.ndarray:
        return self.sentence_embedder.get_embeddings(chunks)

    def get_uuids(self, chunks: list[Document]) -> list[str]:
        """Generate deterministic ids from content and key metadata."""
        # TODO: YOUR CODE HERE
        uuids = []
        for chunk in chunks:
            content = chunk.page_content
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            idx = chunk.metadata.get("idx")
            full = content + source + str(page) + str(idx)
            uuids.append(sha256(full.encode("utf-8")).hexdigest())
        # raise NotImplementedError
        return uuids

    def fill_db(self, *, reset_collection: bool = True) -> int:
        """Split, embed, deduplicate, and store the documents."""
        if reset_collection:
            self._reset_collection()

        chunks = self.preprocess_data()
        embeddings = self.get_embeddings(chunks)
        uuids = self.get_uuids(chunks)

        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        existing_ids = set(self.collection.get()["ids"])
        unique_ids = []
        unique_embeddings = []
        unique_docs = []
        unique_meta = []

        seen_ids = set()
        # TODO: YOUR CODE HERE

        for i, uid in enumerate(uuids):
            if uid in existing_ids:
                continue
            if uid in seen_ids:
                continue

            unique_docs.append(documents[i])
            unique_meta.append(metadatas[i])
            unique_embeddings.append(embeddings[i])
            unique_ids.append(uid)
            seen_ids.add(uid)

        if unique_ids:
            self.collection.add(
            ids=unique_ids,
            documents=unique_docs,
            embeddings=unique_embeddings,
            metadatas=unique_meta,
        )
        
        
        #raise NotImplementedError
        return len(unique_ids)

    def remove_collection(self) -> None:
        self._reset_collection()

    def _init_cross_encoder(self, cross_encoder_name: str) -> None:
        self.cross_encoder = CrossEncoder(cross_encoder_name)

    def rerank(
        self,
        prompt: str,
        documents: list[str],
        top_k: int,
        cross_encoder_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the highest-scoring documents."""
        if self.cross_encoder is None:
            self._init_cross_encoder(cross_encoder_name)
        # TODO: YOUR CODE HERE

        pairs = [[prompt,doc] for doc in documents]

        scores = np.array(self.cross_encoder.predict(pairs))
        sorted_idx = np.argsort(-scores)[:top_k]

        documents = np.array(documents)[sorted_idx]
        scores = scores[sorted_idx]
        #raise NotImplementedError
        return documents, scores

    def query(
        self,
        prompt: str,
        threshold: float = 0.3,
        top_n: int = 5,
        top_k: int = 3,
        use_rerank: bool = True,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Retrieve the most relevant text chunks with manual cosine similarity."""
        # TODO: YOUR CODE HERE
        pair = self.collection.get(include=["documents", "embeddings"])
        docs = pair.get("documents")
        embs = pair.get("embeddings")

        if docs is None or embs is None:
            return None, None
        if len(docs) == 0 or len(embs) == 0:
            return None, None
        
        promt_emb = self.sentence_embedder.sentence_embedder.encode([prompt])

        scr = cosine_similarity(promt_emb, np.asarray(embs))[0]
        top_idx = np.argsort(-scr)[:top_n]
        
        top_docs = np.asarray(docs, dtype=object)[top_idx]
        top_scores = np.asarray(scr, dtype=float)[top_idx]

        mask = top_scores > threshold

        documents = top_docs[mask]
        scores = top_scores[mask]

        if len(documents) == 0:
            return None, None
        
        if use_rerank:
            return self.rerank(
                prompt=prompt,
                documents=documents.tolist(),
                top_k=top_k,
                cross_encoder_name=cross_encoder_name,
        )
        # raise NotImplementedError
        return documents, scores
