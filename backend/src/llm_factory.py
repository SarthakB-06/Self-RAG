"""
LLM Factory to create model instances based on configuration.
Supports Google Gemini and Ollama models.
"""
import urllib.request
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from src.config import Config, logger


class RawOllamaWrapper:
    """A raw HTTP wrapper that conforms to LangChain's interface but bypasses HTTPX/Async bugs."""

    def __init__(self, model_name: str, base_url: str, timeout: int):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def invoke(self, messages):
        # Extract text from LangChain message objects
        prompt_text = "\n".join(
            [getattr(m, 'content', str(m)) for m in messages])
        url = f"{self.base_url}/api/generate"

        data = json.dumps({
            "model": self.model_name,
            "prompt": prompt_text,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers={
                                     "Content-Type": "application/json"})

        # Explicitly pass the timeout to the urlopen call
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

            class MockResponse:
                def __init__(self, content):
                    self.content = content
            return MockResponse(result.get("response", ""))


def create_llm() -> BaseChatModel:
    """
    Creates and returns a language model instance based on the configuration.

    Reads the `LLM_MODEL` from the Config. If it starts with "gemini", it
    creates a ChatGoogleGenerativeAI instance. Otherwise, it assumes it's an
    Ollama model and creates an Ollama instance.

    Returns:
        An instance of a LangChain chat model.
    """
    model_name = Config.LLM_MODEL

    logger.info(f"Creating LLM instance for model: {model_name}")

    if model_name.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            # timeout=Config.LLM_TIMEOUT, # timeout is not a standard parameter here
            convert_system_message_to_human=True
        )
        logger.info("Using Google Gemini model.")
    else:
        # Assume it's an Ollama model
        # Using custom raw wrapper due to HTTPX bugs in LangChain ChatOllama on Windows
        llm = RawOllamaWrapper(
            model_name=model_name,
            base_url=Config.OLLAMA_BASE_URL,
            timeout=Config.OLLAMA_TIMEOUT,
        )
        logger.info(
            f"Using Custom Raw Ollama model (model: {model_name}, url: {Config.OLLAMA_BASE_URL}, timeout: {Config.OLLAMA_TIMEOUT}s).")

    return llm
