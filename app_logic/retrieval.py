import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app_logic.config import CANDIDATE_POOL, DEFAULT_TOP_K


SMART_MIN_RESULTS = 3
SMART_MAX_RESULTS = 8
SMART_MIN_PROB_FLOOR = 0.40
SMART_RELATIVE_TO_BEST = 0.75
SMART_MASS_TARGET = 0.82


def _field(label: str, value: str) -> str | None:
    """Return 'Label: value' only if value is non-empty and not 'nan'."""
    v = value.strip()
    if v and v.lower() != "nan":
        return f"{label}: {v}"
    return None


def build_course_context(df: pd.DataFrame) -> str:
    """Format course rows into a rich text block for the LLM system prompt."""
    parts: list[str] = []
    for _, row in df.iterrows():
        lines: list[str] = []

        def add(label: str, col: str) -> None:
            result = _field(label, str(row.get(col, "")))
            if result:
                lines.append(result)

        add("Ainekood", "aine_kood")
        add("Nimi (ET)", "nimi_et")
        add("Name (EN)", "nimi_en")
        add("EAP", "eap")
        add("Semester", "semester")
        add("Hindamisskaala", "hindamisskaala")
        add("Oppekeeled", "oppekeeled")
        add("Linn", "linn")
        add("Oppeviis", "oppeviis")
        add("Oppeaste", "oppeaste")
        add("Oppejoud", "oppejoud")

        for label, col in [
            ("Eesmargid (ET)", "eesmark_et"),
            ("Goals (EN)", "eesmark_en"),
            ("Oppivaaljundid (ET)", "oppivaaljundid_et"),
            ("Learning outcomes (EN)", "oppivaaljundid_en"),
        ]:
            result = _field(label, str(row.get(col, "")))
            if result:
                lines.append(result)

        desc = str(row.get("description", "")).strip()
        if desc and desc.lower() != "nan":
            lines.append(f"Kirjeldus:\n{desc}")

        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def get_semantic_candidates(
    embedder,
    query: str,
    filtered_df: pd.DataFrame,
    filtered_embeddings: np.ndarray,
    candidate_pool: int = CANDIDATE_POOL,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Encode query, compute cosine similarity, return top-N candidates.

    Returns:
        (candidates_df, cosine_scores_for_candidates)
    """
    query_vec = embedder.encode([query])
    sem_scores = cosine_similarity(query_vec, filtered_embeddings)[0]
    candidate_k = min(candidate_pool, len(filtered_df))
    top_indices = np.argsort(sem_scores)[::-1][:candidate_k]
    candidates_df = filtered_df.iloc[top_indices].reset_index(drop=True)
    candidate_scores = sem_scores[top_indices]
    return candidates_df, candidate_scores


def rerank_candidates(
    reranker,
    query: str,
    candidates_df: pd.DataFrame,
    top_k: int | None = DEFAULT_TOP_K,
    return_scores: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Re-rank candidates with a cross-encoder and return the best ones.

    Args:
        reranker: CrossEncoder model instance.
        query: user query.
        candidates_df: pre-filtered candidates from semantic search.
        top_k: if set, return exactly this many; if None, use smart confidence cutoff.
        return_scores: if True, also return selected confidence scores (0..1).
    """
    pairs = [
        [query, desc]
        for desc in candidates_df["description"].fillna("").tolist()
    ]
    rerank_scores = reranker.predict(pairs)
    sorted_idx = np.argsort(rerank_scores)[::-1]

    if top_k is not None:
        best_indices = sorted_idx[:min(top_k, len(sorted_idx))]
    else:
        sorted_scores = rerank_scores[sorted_idx]
        keep_count = _adaptive_keep_count(sorted_scores, score_kind="logit")
        best_indices = sorted_idx[:keep_count]

    selected_df = candidates_df.iloc[best_indices]
    if not return_scores:
        return selected_df

    selected_scores = np.asarray(rerank_scores, dtype=float)[best_indices]
    selected_confidence = _to_confidence(selected_scores, score_kind="logit")
    return selected_df, selected_confidence


def rerank_candidates_with_local_llm(
    query: str,
    candidates_df: pd.DataFrame,
    rerank_runtime,
    top_k: int | None = DEFAULT_TOP_K,
    return_candidate_indices: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Re-rank candidates with a small local Transformers model.

    The model receives a compact JSON list of candidate snippets and returns
    an ordered list of candidate indices. If parsing fails, falls back to
    the original semantic order.
    """
    if candidates_df.empty:
        if return_candidate_indices:
            return candidates_df, np.array([], dtype=int)
        return candidates_df

    limit = len(candidates_df) if top_k is None else top_k
    limit = min(limit, len(candidates_df))
    prepared_df = candidates_df.reset_index(drop=True)

    payload: list[dict[str, str | int]] = []
    for idx, (_, row) in enumerate(prepared_df.iterrows(), start=1):
        payload.append({
            "index": idx,
            "aine_kood": str(row.get("aine_kood", "")),
            "nimi_et": str(row.get("nimi_et", "")),
            "description": str(row.get("description", ""))[:260],
        })

    prompt = (
        "Rank these University of Tartu courses by relevance to the user query.\n"
        "Return ONLY JSON in this format: {\"ranked_indices\": [3,1,2,...]}\n"
        "Use each index at most once and include only listed indices.\n\n"
        f"Query: {query}\n\n"
        f"Candidates: {json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        text = generate_local_rerank_response(
            rerank_runtime,
            prompt,
        )
    except Exception as exc:
        model_name = rerank_runtime.get("model_name", "unknown")
        raise RuntimeError(
            f"Kohalik Transformers rerank ebaõnnestus ({model_name}): {exc}"
        ) from exc

    ranked_indices = _parse_ranked_indices(text, len(prepared_df))
    if not ranked_indices:
        selected_indices = np.arange(limit, dtype=int)
        selected_df = prepared_df.iloc[selected_indices]
        if return_candidate_indices:
            return selected_df, selected_indices
        return selected_df

    zero_based = [i - 1 for i in ranked_indices]
    selected_indices = np.asarray(zero_based[:limit], dtype=int)
    selected_df = prepared_df.iloc[selected_indices]
    if return_candidate_indices:
        return selected_df, selected_indices
    return selected_df


def select_semantic_results(
    candidates_df: pd.DataFrame,
    candidate_scores: np.ndarray,
    top_k: int | None = DEFAULT_TOP_K,
    return_confidence: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Select semantic candidates using fixed top-k or smart confidence cutoff."""
    if candidates_df.empty:
        if return_confidence:
            return candidates_df, np.array([], dtype=float)
        return candidates_df

    keep_count = min(top_k, len(candidates_df)) if top_k is not None else _adaptive_keep_count(
        candidate_scores,
        score_kind="cosine",
    )
    selected_df = candidates_df.iloc[:keep_count]
    if not return_confidence:
        return selected_df

    selected_scores = np.asarray(candidate_scores[:keep_count], dtype=float)
    selected_confidence = _to_confidence(selected_scores, score_kind="cosine")
    return selected_df, selected_confidence


def _adaptive_keep_count(scores: np.ndarray, score_kind: str) -> int:
    """Choose how many ranked items to keep using confidence + mass criteria."""
    if scores.size == 0:
        return 0
    if scores.size == 1:
        return 1

    confidence = _to_confidence(scores, score_kind=score_kind)
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0)

    best_conf = float(confidence[0])
    threshold = max(SMART_MIN_PROB_FLOOR, best_conf * SMART_RELATIVE_TO_BEST)
    above_threshold = np.where(confidence >= threshold)[0]
    keep_by_threshold = int(above_threshold[-1] + 1) if above_threshold.size else 1

    total_mass = float(confidence.sum())
    if total_mass > 0:
        cumulative = np.cumsum(confidence) / total_mass
        keep_by_mass = int(np.searchsorted(cumulative, SMART_MASS_TARGET) + 1)
    else:
        keep_by_mass = 1

    keep_count = max(SMART_MIN_RESULTS, keep_by_threshold, keep_by_mass)
    keep_count = min(keep_count, SMART_MAX_RESULTS, len(scores))
    return int(max(1, keep_count))


def _to_confidence(scores: np.ndarray, score_kind: str) -> np.ndarray:
    """Convert raw ranking scores into 0..1 confidence values."""
    scores = np.asarray(scores, dtype=float)
    if score_kind == "cosine":
        return np.clip((scores + 1.0) / 2.0, 0.0, 1.0)
    if score_kind == "logit":
        clipped = np.clip(scores, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    raise ValueError(f"Unsupported score_kind: {score_kind}")


def load_local_transformers_reranker(model_name: str):
    """Load a small causal LM for local list reranking."""
    torch = __import__("torch")
    transformers = __import__("transformers", fromlist=["AutoTokenizer", "AutoModelForCausalLM"])

    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs: dict = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "model_name": model_name,
    }


def generate_local_rerank_response(rerank_runtime, prompt: str) -> str:
    """Generate local reranking output text with a cached Transformers model."""
    torch = __import__("torch")
    tokenizer = rerank_runtime["tokenizer"]
    model = rerank_runtime["model"]
    device = rerank_runtime["device"]

    messages = [
        {
            "role": "system",
            "content": (
                "You rank course candidates and respond ONLY with JSON: "
                "{\"ranked_indices\": [..]}."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            model_input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            model_input_text = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    else:
        model_input_text = f"{messages[0]['content']}\n\n{messages[1]['content']}"

    encoded = tokenizer(
        model_input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=96,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    completion_ids = output[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _parse_ranked_indices(text: str, n_items: int) -> list[int]:
    """Parse a ranked index list from LLM text and complete missing indices."""
    ranked: list[int] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if 1 <= idx <= n_items and idx not in seen:
            seen.add(idx)
            ranked.append(idx)

    extracted: list[int] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}

    if isinstance(payload, dict):
        for key in ("ranked_indices", "ranked", "order", "indices"):
            value = payload.get(key)
            if isinstance(value, list):
                extracted = value
                break

    if not extracted:
        extracted = [int(x) for x in re.findall(r"\b\d+\b", text)]

    for value in extracted:
        try:
            add(int(value))
        except (TypeError, ValueError):
            continue

    for idx in range(1, n_items + 1):
        add(idx)
    return ranked


def build_benchmark_context(ranked_df: pd.DataFrame) -> str:
    """Build context text for the benchmark LLM prompt.

    Includes an explicit list of allowed aine_kood values and a truncated
    JSON representation of the ranked courses.
    """
    if ranked_df.empty:
        from app_logic.config import DEFAULT_EMPTY_CONTEXT
        return DEFAULT_EMPTY_CONTEXT

    context_columns = [
        col for col in [
            "aine_kood", "nimi_et", "eap", "semester", "oppeaste",
            "linn", "oppeviis", "kirjeldus_et",
        ]
        if col in ranked_df.columns
    ]
    context_df = ranked_df[context_columns].copy() if context_columns else ranked_df.copy()

    if "kirjeldus_et" in context_df.columns:
        context_df["kirjeldus_et"] = (
            context_df["kirjeldus_et"].fillna("").astype(str).str.slice(0, 400)
        )

    allowed_ids = ", ".join(
        _normalize_ids(context_df["aine_kood"].tolist())
    ) if "aine_kood" in context_df.columns else "-"

    records_json = context_df.to_json(orient="records", force_ascii=False)
    return f"Allowed aine_kood values: {allowed_ids}\n\nCourses JSON:\n{records_json}"


def _normalize_ids(values: list) -> list[str]:
    """Strip, uppercase, deduplicate a list of ID values."""
    seen: set[str] = set()
    result: list[str] = []
    for v in values:
        norm = str(v).strip().upper()
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


# ---------------------------------------------------------------------------
# Batch operations for benchmark performance
# ---------------------------------------------------------------------------
def batch_encode_queries(embedder, queries: list[str]) -> np.ndarray:
    """Encode all queries in a single batch call.

    Returns an (N, D) array where N = len(queries) and D = embedding dim.
    """
    return embedder.encode(queries, show_progress_bar=False)


def batch_cosine_similarity(
    query_vectors: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between all query vectors and all embeddings.

    Args:
        query_vectors: (N, D) array of encoded queries.
        embeddings: (M, D) array of course embeddings.

    Returns:
        (N, M) similarity matrix.
    """
    return cosine_similarity(query_vectors, embeddings)


def get_semantic_candidates_from_scores(
    scores_row: np.ndarray,
    filtered_df: pd.DataFrame,
    candidate_pool: int = CANDIDATE_POOL,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract top-N candidates from a pre-computed similarity score row.

    This avoids re-encoding the query and re-computing cosine similarity,
    making benchmark loops much faster.

    Args:
        scores_row: 1-D array of cosine similarities (length = len(filtered_df)).
        filtered_df: the course DataFrame aligned with scores_row.
        candidate_pool: how many top candidates to return.

    Returns:
        (candidates_df, cosine_scores_for_candidates)
    """
    candidate_k = min(candidate_pool, len(filtered_df))
    top_indices = np.argsort(scores_row)[::-1][:candidate_k]
    candidates_df = filtered_df.iloc[top_indices].reset_index(drop=True)
    candidate_scores = scores_row[top_indices]
    return candidates_df, candidate_scores
