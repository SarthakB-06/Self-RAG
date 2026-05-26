"""
Handles environment variables, logging setup, and system constants.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


load_dotenv()


class Config:
    """Centralized configuration for the Self-RAG agent."""

    ENV: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    # Or any other model like "mistral"
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    # Timeout in seconds. Increased for heavy metric test loads.
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "600"))

    MAX_RETRIEVAL_LOOPS: int = int(os.getenv("MAX_RETRIEVAL_LOOPS", "3"))
    MAX_GENERATION_RETRIES: int = int(os.getenv("MAX_GENERATION_RETRIES", "2"))
    MAX_TOTAL_EXECUTION_TIME: int = int(
        os.getenv("MAX_TOTAL_EXECUTION_TIME", "120"))

    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.3"))

    MIN_GENERATION_QUALITY: str = os.getenv("MIN_GENERATION_QUALITY", "medium")
    STRICT_HALLUCINATION_CHECK: bool = os.getenv(
        "STRICT_HALLUCINATION_CHECK", "true").lower() == "true"

    PROJECT_ROOT: Path = Path(__file__).parent.parent
    RUNBOOK_DIR: Path = PROJECT_ROOT / "enterprise_runbooks"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    CACHE_DIR: Path = PROJECT_ROOT / ".cache"

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[Path] = LOGS_DIR / "agent.log" if not DEBUG else None

    ENABLE_METRICS: bool = os.getenv(
        "ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8080"))

    USE_VECTOR_SEARCH: bool = os.getenv(
        "USE_VECTOR_SEARCH", "false").lower() == "true"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    VECTOR_DB_PATH: Optional[Path] = CACHE_DIR / \
        "vector_db" if USE_VECTOR_SEARCH else None

    @classmethod
    def validate(cls) -> None:
        """Validate configuration and create necessary directories."""
        if not cls.GEMINI_API_KEY and cls.LLM_MODEL.startswith("gemini"):
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini models")

        if not cls.RUNBOOK_DIR.exists():
            raise FileNotFoundError(
                f"Runbook directory not found: {cls.RUNBOOK_DIR}")

        # Create directories if they don't exist
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)

        if cls.USE_VECTOR_SEARCH and cls.VECTOR_DB_PATH:
            cls.VECTOR_DB_PATH.mkdir(exist_ok=True)

    @classmethod
    def setup_logging(cls) -> logging.Logger:
        """Configure logging for the application."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(
                    cls.LOG_FILE) if cls.LOG_FILE else logging.NullHandler()
            ]
        )

        logger = logging.getLogger("self_rag_agent")
        logger.info(
            f"Logging initialized (level={cls.LOG_LEVEL}, env={cls.ENV})")
        return logger


Config.validate()

logger = Config.setup_logging()
