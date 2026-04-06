import os
import certifi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ['SSL_CERT_FILE'] = certifi.where()

# === Legacy Configuration (Backward Compatibility) ===
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL")
MODEL_NAME = os.getenv("MOONSHOT_MODEL_NAME", "moonshot-v1-32k")

# === New Model Configurations ===
class EmbeddingConfig:
    MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
    API_URL = os.getenv("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings")
    API_KEY = os.getenv("EMBEDDING_API_KEY")
    BATCH_SIZE = 32
    MAX_RETRIES = 3
    TIMEOUT = 60.0

class RerankerConfig:
    MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-8B")
    API_URL = os.getenv("RERANKER_API_URL", "https://api.siliconflow.cn/v1/rerank")
    API_KEY = os.getenv("RERANKER_API_KEY")
    MAX_DOCUMENTS = 100
    THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.5"))
    TIMEOUT = 60.0

class LLMConfig:
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "kimi-k2.5")
    API_URL = os.getenv("LLM_API_URL", "https://api.moonshot.cn/v1/chat/completions")
    API_KEY = os.getenv("LLM_API_KEY")
    TEMPERATURE = 0.3
    TIMEOUT = 120.0  # Increased from 60 to 120 seconds for complex queries

class SearchConfig:
    # Hybrid Search Weights
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    
    # Progressive Disclosure
    LEVEL1_TOP_K = int(os.getenv("LEVEL1_TOP_K", "10"))
    LEVEL2_TOP_K = int(os.getenv("LEVEL2_TOP_K", "20"))
    LEVEL3_TOP_K = int(os.getenv("LEVEL3_TOP_K", "30"))
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))

class GenerationConfig:
    # Context Window (reduced for better performance)
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "50000"))  # Reduced from 100000
    CONVERSATION_HISTORY_TURNS = int(os.getenv("CONVERSATION_HISTORY_TURNS", "3"))
    
    # Route-specific Limits
    QUICK_ANSWER_MAX_CHUNKS = int(os.getenv("QUICK_ANSWER_MAX_CHUNKS", "3"))
    TUTORIAL_MAX_CHUNKS = int(os.getenv("TUTORIAL_MAX_CHUNKS", "8"))  # Reduced from 10
    DEEPDIVE_MAX_CHUNKS = int(os.getenv("DEEPDIVE_MAX_CHUNKS", "15"))  # Reduced from 20

class ChromaDBConfig:
    PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
    COLLECTION_NAME = "knowledge_base"

# 目录结构配置
BASE_DATA_DIR = "./data"
COURSE_CONTENT_DIR = os.path.join(BASE_DATA_DIR, "course_content")
QUESTION_DIR = os.path.join(BASE_DATA_DIR, "question")
UPLOAD_DIR = "./upload_file"
LOGS_DIR = "./logs"

# 确保所有必要目录存在
for path in [COURSE_CONTENT_DIR, QUESTION_DIR, UPLOAD_DIR, LOGS_DIR, ChromaDBConfig.PERSIST_DIR]:
    os.makedirs(path, exist_ok=True)

# Validate critical configuration
if not EmbeddingConfig.API_KEY:
    raise ValueError("EMBEDDING_API_KEY not found in environment variables")
if not RerankerConfig.API_KEY:
    raise ValueError("RERANKER_API_KEY not found in environment variables")
if not LLMConfig.API_KEY:
    raise ValueError("LLM_API_KEY not found in environment variables")