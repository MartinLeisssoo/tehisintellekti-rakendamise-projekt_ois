# ---------- Paths ----------
FEEDBACK_LOG_PATH = "tagasiside_log.csv"
DATA_PATH = "andmed/puhastatud_andmed.csv"
EMBEDDINGS_PATH = "andmed/embeddings.pkl"
BENCHMARK_CASES_PATH = "Testjuhtumid.csv"
BENCHMARK_RUNS_PATH = "benchmark_data/benchmark_runs.json"

# ---------- Model names ----------
EMBED_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LOCAL_RERANK_MODEL = "Qwen/Qwen3-0.6B"
LLM_MODEL = "google/gemma-3-27b-it"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------- Retrieval defaults ----------
DEFAULT_TOP_K = 5
CANDIDATE_POOL = 20

# ---------- Benchmark ----------
DEFAULT_EMPTY_CONTEXT = "Sobivaid kursusi ei leitud."

# ---------- Pricing (USD per million tokens) ----------
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "google/gemma-3-27b-it": (0.10, 0.20),
}
