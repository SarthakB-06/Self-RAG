import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.config import Config, logger


def main():
    """Build and save vector index."""
    
    # Initialize vector store
    store = VectorStore()
    
    # Build index
    try:
        store.build_index(Config.RUNBOOK_DIR)
        
        # Print statistics
        print("\n Index Statistics:")
        stats = store.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n Index built and saved successfully!")
        print(f"   Location: {store.index_path}")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()