import re
import pickle
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, asdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import Config, logger


@dataclass
class VectorDocument:
    """Document with vector embedding."""
    doc_id: str

    category: str

    filename: str

    title: str

    content: str

    embedding: Optional[np.ndarray] = None

    word_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dict (excluding embedding for serialization)."""
        data = asdict(self)
        data.pop('embedding', None)
        return data


@dataclass
class SearchResult:
    """Search result with similarity score."""
    document: VectorDocument
    similarity_score: float
    rank: int

    @property
    def content(self) -> str:
        """Get document content for backward compatibility."""
        return self.document.content

    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "similarity_score": float(self.similarity_score),
            "rank": self.rank
        }


class VectorStore:
    """
    FAISS-based vector store for semantic search.

    Features:
    - Sentence Transformers embeddings (all-MiniLM-L6-v2)
    - FAISS IndexFlatIP for cosine similarity
    - Persistent storage (save/load index)
    - Hybrid search (vector + keyword)
    - Batch embedding for performance
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        store_path: str = ".cache/vector_db"
    ):
        """
        Initialize vector store.

        Args:
            model_name: HuggingFace model for embeddings
            index_path: Path to save/load FAISS index
        """
        self.model_name = model_name
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_path / "faiss.index"
        self.metadata_path = self.store_path / "metadata.pkl"

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # FAISS index (using Inner Product for cosine similarity)
        self.index: Optional[faiss.Index] = None
        self.documents: List[VectorDocument] = []

        # Metrics
        self.total_embeddings_generated = 0
        self.total_searches = 0

        logger.info(f"Vector store initialized (dim={self.embedding_dim})")

    def build_index(self, runbook_dir: Path) -> None:
        """
        Build FAISS index from runbook directory.

        This is a ONE-TIME operation (or when runbooks change).

        Args:
            runbook_dir: Path to enterprise_runbooks/
        """
        logger.info(f"Building vector index from runbooks in: {runbook_dir}")
        start_time = time.time()
        #  check for this load_documents function
        documents = self._load_documents(runbook_dir)
        logger.info(f"Loaded {len(documents)} documents from runbooks")

        embeddings = self._generate_embeddings(
            [doc.content for doc in documents])

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        self.documents = documents
        self._build_faiss_index(embeddings)

        self.save_index()

        elapsed = time.time() - start_time
        logger.info(f"Index built and saved in {elapsed:.2f} seconds")

    def _load_documents(self, runbook_dir: Path) -> List[VectorDocument]:
        """
        Load all runbook documents from directory.
        Supports nested directory structure:
        - AWS Playbooks/01-Compute/*.md
        - K8s Playbooks/03-Pods/*.md
        - Sentry Playbooks/01-Error-Tracking/*.md
        """
        documents = []

        logger.info(f"Scanning runbook directory: {runbook_dir}")

    # Recursively find all .md files (excluding README.md)
        for runbook_file in runbook_dir.rglob("*.md"):
            # Skip README files
            if runbook_file.name.lower() == "readme.md":
                logger.debug(f"Skipping README: {runbook_file}")
                continue

            try:
                with open(runbook_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract category from parent folder structure
                # Example: "AWS Playbooks/01-Compute" -> category="AWS-Compute"
                relative_path = runbook_file.relative_to(runbook_dir)
                path_parts = relative_path.parts

                # Determine category based on path
                if len(path_parts) >= 2:
                    # "AWS Playbooks" -> "aws"
                    # "K8s Playbooks" -> "kubernetes"
                    # "Sentry Playbooks" -> "sentry"
                    main_category = path_parts[0].lower().replace(
                        " playbooks", "").replace(" ", "-")

                    # Map common variations
                    category_mapping = {
                        "aws": "aws",
                        "k8s": "kubernetes",
                        "kubernetes": "kubernetes",
                        "sentry": "sentry"
                    }

                    main_category = category_mapping.get(
                        main_category, main_category)

                    # Add subcategory if present (01-Compute -> compute)
                    if len(path_parts) >= 3:
                        subcategory = path_parts[1].lower()
                        # Remove numbering prefix (01-, 02-, etc.)
                        import re
                        subcategory = re.sub(r'^\d+-', '', subcategory)
                        category = f"{main_category}-{subcategory}"
                    else:
                        category = main_category
                else:
                    category = "uncategorized"

                # Extract title
                title = self._extract_title(content)

                # Create document ID from relative path
                doc_id = str(relative_path).replace("\\", "/")

                doc = VectorDocument(
                    doc_id=doc_id,
                    category=category,
                    filename=runbook_file.name,
                    title=title,
                    content=content,
                    word_count=len(content.split())
                )

                documents.append(doc)
                logger.debug(f"Loaded: {doc_id} (category: {category})")

            except Exception as e:
                logger.error(f"Failed to load {runbook_file}: {e}")

        logger.info(f"Loaded {len(documents)} documents from runbooks")
        return documents

    @staticmethod
    def _extract_title(content: str) -> str:
        """Extract title from markdown (first # heading)."""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} documents...")

        # Normalize embeddings for cosine similarity
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # Critical for cosine similarity
        )

        self.total_embeddings_generated += len(texts)

        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """
        Build FAISS index from embeddings.

        Using IndexFlatIP (Inner Product) for cosine similarity.
        For large datasets (>10k docs), consider IndexIVFFlat.
        """
        logger.info("Building FAISS index...")

        # Create index (Inner Product = cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Add embeddings
        self.index.add(embeddings.astype('float32'))

        logger.info(f"FAISS index built: {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.5,
        categories: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Semantic search using vector similarity.

        Args:
            query: Search query (natural language)
            top_k: Number of results to return
            min_score: Minimum similarity threshold (0-1)
            categories: Optional category filter

        Returns:
            List of SearchResult objects ranked by similarity
        """
        if not self.index or not self.documents:
            raise RuntimeError(
                "Index not built. Call build_index() or load_index() first.")

        self.total_searches += 1
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Search in FAISS
        # Note: FAISS returns (distances, indices)
        # With IndexFlatIP, distance = cosine similarity (higher is better)
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            # Retrieve more for filtering
            k=min(top_k * 2, len(self.documents))
        )

        # Build results
        results = []
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0]), 1):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc = self.documents[idx]

            # Filter by category
            if categories and doc.category not in categories:
                continue

            # Filter by minimum score
            if similarity < min_score:
                continue

            results.append(SearchResult(
                document=doc,
                similarity_score=float(similarity),
                rank=rank
            ))

        # Take top K after filtering
        results = results[:top_k]

        search_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Vector search: {len(results)} results in {search_time_ms:.2f}ms")

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.7,
        categories: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.

        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Weight for vector score (0-1), keyword gets (1-weight)
            categories: Optional category filter

        Returns:
            List of SearchResult objects with combined scores
        """
        # Get vector results
        vector_results = self.search(
            query, top_k=top_k*2, min_score=0.0, categories=categories)

        # Calculate keyword scores
        query_keywords = set(query.lower().split())

        results_with_hybrid_score = []
        for result in vector_results:
            # Keyword score (simple TF matching)
            doc_text = f"{result.document.title} {result.document.filename} {result.document.content}".lower()
            keyword_matches = sum(1 for kw in query_keywords if kw in doc_text)
            keyword_score = keyword_matches / \
                len(query_keywords) if query_keywords else 0.0

            # Combined score
            hybrid_score = (
                vector_weight * result.similarity_score +
                (1 - vector_weight) * keyword_score
            )

            result.similarity_score = hybrid_score
            results_with_hybrid_score.append(result)

        # Re-sort by hybrid score
        results_with_hybrid_score.sort(
            key=lambda r: r.similarity_score, reverse=True)

        return results_with_hybrid_score[:top_k]

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if not self.index:
            raise RuntimeError("No index to save")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata (documents without embeddings)
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "documents": [doc.to_dict() for doc in self.documents],
            "total_embeddings": self.total_embeddings_generated,
            "created_at": time.time()
        }

        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Index saved to {self.index_path}")

    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if successful, False otherwise
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("Index files not found")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Reconstruct documents (without embeddings to save memory)
            self.documents = [
                VectorDocument(**doc_dict)
                for doc_dict in metadata["documents"]
            ]

            self.total_embeddings_generated = metadata.get(
                "total_embeddings", 0)

            logger.info(f"Index loaded: {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get vector store statistics."""
        return {
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "total_documents": len(self.documents),
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_searches": self.total_searches,
            "index_size_mb": self.index_path.stat().st_size / 1024 / 1024 if self.index_path.exists() else 0
        }


if __name__ == "__main__":
    """Test vector store."""

    print("🧪 Testing Vector Store\n")

    # Create vector store
    store = VectorStore()

    # Check if index exists
    if not store.load_index():
        print("❌ Index not found. Building index...")
        print("   Run: python scripts/build_vector_index.py")
        exit(1)

    # Print statistics
    print("📊 Vector Store Statistics:")
    stats = store.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test queries
    test_queries = [
        "postgres database high CPU usage",
        "kubernetes pod keeps restarting",
        "AWS EC2 instance performance issues",
    ]

    print("\n🔍 Testing Vector Search:")
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = store.search(query, top_k=3)

        for result in results:
            print(f"      {result.rank}. {result.document.doc_id} "
                  f"(score: {result.similarity_score:.3f})")

    print("\n🔍 Testing Hybrid Search:")
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = store.hybrid_search(query, top_k=3)

        for result in results:
            print(f"      {result.rank}. {result.document.doc_id} "
                  f"(hybrid score: {result.similarity_score:.3f})")
