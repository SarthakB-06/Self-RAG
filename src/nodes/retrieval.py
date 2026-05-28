"""
Retrieval node for Self-RAG agent.
Fetches relevant runbooks using vector search.
"""

import time
from pathlib import Path
from typing import Dict, Any
from src.state import AgentState
from src.config import Config, logger
from src.vector_store import VectorStore

_vector_store_instance = None


def get_vector_store_instance() -> VectorStore:
    """Initializes and returns a singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        logger.info("Initializing direct VectorStore instance for retrieval...")
        _vector_store_instance = VectorStore(
            store_path=Config.VECTOR_DB_PATH)
        
        # If running on Render, the index won't exist because .cache is in .gitignore.
        # We must build it dynamically.
        if not _vector_store_instance.load_index():
            logger.info("Index not found. Building FAISS index dynamically via Gemini API...")
            runbook_path = Path(Config.RUNBOOK_DIR)
            _vector_store_instance.build_index(runbook_path)
            
        logger.info("Direct VectorStore instance loaded.")
    return _vector_store_instance


def retrieve_documents(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve runbooks using direct vector search.
    """
    start_time = time.time()
    query = state["current_query"]
    logger.info(f"Retrieving documents for query: {query}")

    try:
        store = get_vector_store_instance()

        results = store.search(
            query=query,
            top_k=Config.TOP_K_DOCUMENTS,
            min_score=Config.MIN_RELEVANCE_SCORE
        )

        documents = [result.document.content for result in results]
        document_scores = [result.similarity_score for result in results]
        logger.info(
            f"Retrieved {len(documents)} documents with scores: {document_scores}")

        execution_path = state["execution_path"] + ["retrieve"]

        # Calculate retrieval score (average of top 3 or whatever was found)
        avg_score = sum(document_scores) / \
            len(document_scores) if document_scores else 0.0

        return {
            "documents": documents,
            "document_scores": document_scores,
            "retrieval_loop_count": state["retrieval_loop_count"] + 1,
            "execution_path": execution_path,
            "query_history": state["query_history"] + [query],
            "retrieval_time": time.time() - start_time,
            "retrieval_score": avg_score
        }
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        return {
            "documents": [],
            "document_scores": [],
            "errors": state["errors"] + [f"Retrieval error: {str(e)}"],
            "execution_path": state["execution_path"] + ["retrieve_error"],
            "retrieval_time": time.time() - start_time
        }
