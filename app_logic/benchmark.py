"""Benchmark evaluation engine for the course recommendation pipeline.

Evaluates 3 stages:
  1. Vector search (cosine similarity) — are expected courses in the top-N candidates?
  2. Reranker (cross-encoder) — are expected courses in the reranked results?
  3. LLM final answer — does the LLM include expected courses in its structured response?
"""

import csv
import gc
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app_logic.config import (
    BENCHMARK_CASES_PATH,
    BENCHMARK_RUNS_PATH,
    CANDIDATE_POOL,
    DEFAULT_TOP_K,
)
from app_logic.llm import (
    build_benchmark_system_prompt,
    build_benchmark_user_prompt,
    create_benchmark_completion,
    parse_benchmark_ids,
)
from app_logic.retrieval import (
    batch_cosine_similarity,
    batch_encode_queries,
    build_benchmark_context,
    get_semantic_candidates,
    get_semantic_candidates_from_scores,
    rerank_candidates,
    rerank_candidates_with_local_llm,
    _normalize_ids,
)


def _cleanup_after_case() -> None:
    """Release transient framework cache between benchmark queries."""
    gc.collect()
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkCase:
    row_number: int
    query: str
    expected_ids: list[str]
    expects_empty: bool
    parse_error: str | None = None


@dataclass
class ComparisonResult:
    passed: bool
    missing_ids: list[str]
    unexpected_ids: list[str]


@dataclass
class StageResult:
    returned_ids: list[str]
    passed: bool
    missing_ids: list[str]
    unexpected_ids: list[str]
    raw_text: str | None = None


@dataclass
class CaseBenchmarkResult:
    case: BenchmarkCase
    retrieval: StageResult
    reranker: StageResult
    llm: StageResult


@dataclass
class BenchmarkRunResult:
    total_cases: int
    retrieval_correct: int
    retrieval_incorrect: int
    reranker_correct: int
    reranker_incorrect: int
    llm_correct: int
    llm_incorrect: int
    case_results: list[CaseBenchmarkResult]


# ---------------------------------------------------------------------------
# ID normalization and comparison
# ---------------------------------------------------------------------------
def _normalize_course_ids(values: list) -> list[str]:
    """Strip, uppercase, deduplicate a list of ID values."""
    return _normalize_ids(values)


def _parse_expected_ids(raw_value: str) -> tuple[list[str], bool]:
    """Parse the expected-IDs column from the test CSV.

    Handles:
      - '-' or 'ei soovita midagi' -> expects_empty=True
      - comma or semicolon separated IDs
    """
    cleaned = raw_value.strip()

    # Empty-expectation markers
    if cleaned == "-" or cleaned.lower() in ("ei soovita midagi",):
        return [], True

    tokens = re.split(r"[;,]", cleaned)
    ids = _normalize_course_ids(tokens)
    return ids, False


def compare_ids(
    expected_ids: list[str],
    actual_ids: list[str],
    expects_empty: bool,
) -> ComparisonResult:
    """Compare expected vs. actual course IDs.

    Pass criteria (recall-oriented):
      - Normal case: ALL expected IDs must be present in actual. Extra IDs are OK.
      - Empty-expected case: pass only if zero IDs are returned.
    """
    normalized_expected = _normalize_course_ids(expected_ids)
    normalized_actual = _normalize_course_ids(actual_ids)

    if expects_empty:
        return ComparisonResult(
            passed=len(normalized_actual) == 0,
            missing_ids=[],
            unexpected_ids=normalized_actual,
        )

    missing = [cid for cid in normalized_expected if cid not in normalized_actual]
    unexpected = [cid for cid in normalized_actual if cid not in normalized_expected]
    return ComparisonResult(
        passed=len(missing) == 0,
        missing_ids=missing,
        unexpected_ids=unexpected,
    )


# ---------------------------------------------------------------------------
# Helpers for building stage results
# ---------------------------------------------------------------------------
def _build_stage_result(
    case: BenchmarkCase,
    returned_ids: list[str],
    raw_text: str | None = None,
) -> StageResult:
    comparison = compare_ids(case.expected_ids, returned_ids, case.expects_empty)
    return StageResult(
        returned_ids=returned_ids,
        passed=comparison.passed,
        missing_ids=comparison.missing_ids,
        unexpected_ids=comparison.unexpected_ids,
        raw_text=raw_text,
    )


def _build_invalid_stage_result(case: BenchmarkCase) -> StageResult:
    raw_text = case.parse_error or "Invalid benchmark row."
    missing = case.expected_ids if not case.expects_empty else []
    return StageResult(
        returned_ids=[],
        passed=False,
        missing_ids=missing,
        unexpected_ids=[],
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_benchmark_cases(path: str = BENCHMARK_CASES_PATH) -> list[BenchmarkCase]:
    """Parse the test-cases CSV into a list of BenchmarkCase objects.

    Handles multi-line queries, semicolon/comma separators, and special
    empty-expectation markers ('-', 'ei soovita midagi').
    """
    cases: list[BenchmarkCase] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header

        for row_number, row in enumerate(reader, start=2):
            if len(row) < 2:
                query = row[0].strip() if row else ""
                cases.append(BenchmarkCase(
                    row_number=row_number,
                    query=query,
                    expected_ids=[],
                    expects_empty=False,
                    parse_error="Row has fewer than 2 columns.",
                ))
                continue

            query = row[0].strip()
            expected_ids, expects_empty = _parse_expected_ids(row[1])
            parse_error = None
            if not expects_empty and not expected_ids:
                parse_error = "Expected ID column is empty or invalid."

            cases.append(BenchmarkCase(
                row_number=row_number,
                query=query,
                expected_ids=expected_ids,
                expects_empty=expects_empty,
                parse_error=parse_error,
            ))

    return cases


# ---------------------------------------------------------------------------
# LLM ID resolution
# ---------------------------------------------------------------------------
def _resolve_llm_ids(returned_ids: list[str], ranked_df: pd.DataFrame) -> list[str]:
    """Map LLM-returned IDs to valid aine_kood values from the retrieval context.

    Handles matching by exact aine_kood and by base code (before '_' suffix).
    """
    if ranked_df.empty:
        return []

    resolved: list[str] = []
    seen: set[str] = set()

    # Build lookup maps
    aine_kood_set: set[str] = set()
    base_to_aine: dict[str, list[str]] = {}

    for _, row in ranked_df.iterrows():
        ak = str(row.get("aine_kood", "")).strip().upper()
        if ak:
            aine_kood_set.add(ak)
            base = ak.split("_")[0]
            base_to_aine.setdefault(base, [])
            if ak not in base_to_aine[base]:
                base_to_aine[base].append(ak)

    for rid in returned_ids:
        norm = str(rid).strip().upper()
        candidates: list[str] = []

        if norm in aine_kood_set:
            candidates = [norm]
        elif norm in base_to_aine:
            candidates = base_to_aine[norm][:1]

        for cid in candidates:
            if cid not in seen:
                seen.add(cid)
                resolved.append(cid)

    return resolved


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------
def evaluate_case_retrieval(
    case: BenchmarkCase,
    embedder,
    courses_df: pd.DataFrame,
    embeddings: np.ndarray,
    candidate_pool: int = CANDIDATE_POOL,
    *,
    precomputed_scores: np.ndarray | None = None,
) -> tuple[StageResult, pd.DataFrame]:
    """Stage 1: evaluate cosine-similarity retrieval for one test case.

    If *precomputed_scores* is supplied (a 1-D similarity array aligned with
    *courses_df*), candidates are extracted without re-encoding the query or
    recomputing cosine similarity.  Otherwise, falls back to the original
    encode-then-compare path.

    Returns (stage_result, candidates_df).
    """
    if case.parse_error:
        return _build_invalid_stage_result(case), pd.DataFrame()

    if precomputed_scores is not None:
        candidates_df, _ = get_semantic_candidates_from_scores(
            precomputed_scores, courses_df, candidate_pool=candidate_pool,
        )
    else:
        candidates_df, _ = get_semantic_candidates(
            embedder, case.query, courses_df, embeddings,
            candidate_pool=candidate_pool,
        )

    returned_ids = _normalize_course_ids(
        candidates_df["aine_kood"].tolist()
    ) if "aine_kood" in candidates_df.columns else []

    return _build_stage_result(case, returned_ids), candidates_df


def evaluate_case_reranker(
    case: BenchmarkCase,
    candidates_df: pd.DataFrame,
    reranker=None,
    local_rerank_runtime=None,
    ranking_mode: str = "cross_encoder",
    top_k: int | None = DEFAULT_TOP_K,
) -> tuple[StageResult, pd.DataFrame]:
    """Stage 2: evaluate reranking for one test case.

    Returns (stage_result, reranked_df).
    """
    if case.parse_error or candidates_df.empty:
        return _build_invalid_stage_result(case), pd.DataFrame()

    def semantic_fallback() -> pd.DataFrame:
        if top_k is None:
            return candidates_df.copy()
        return candidates_df.iloc[:min(top_k, len(candidates_df))]

    rerank_raw_text: str | None = None

    if ranking_mode == "semantic":
        reranked_df = semantic_fallback()
        rerank_raw_text = "Reranker disabled; used semantic order."
    elif ranking_mode == "cross_encoder":
        if reranker is None:
            reranked_df = semantic_fallback()
            rerank_raw_text = "Cross-encoder missing; used semantic order."
        else:
            try:
                reranked_df = rerank_candidates(reranker, case.query, candidates_df, top_k=top_k)
            except Exception as error:
                reranked_df = semantic_fallback()
                rerank_raw_text = f"Cross-encoder failed; used semantic order. ({error})"
    elif ranking_mode == "local_llm":
        if local_rerank_runtime is None:
            reranked_df = semantic_fallback()
            rerank_raw_text = "Local LLM reranker missing; used semantic order."
        else:
            try:
                reranked_df = rerank_candidates_with_local_llm(
                    case.query,
                    candidates_df,
                    rerank_runtime=local_rerank_runtime,
                    top_k=top_k,
                )
            except Exception as error:
                reranked_df = semantic_fallback()
                rerank_raw_text = f"Local LLM reranker failed; used semantic order. ({error})"
    else:
        raise ValueError(f"Unsupported benchmark ranking mode: {ranking_mode}")

    returned_ids = _normalize_course_ids(
        reranked_df["aine_kood"].tolist()
    ) if "aine_kood" in reranked_df.columns else []

    return _build_stage_result(case, returned_ids, raw_text=rerank_raw_text), reranked_df


def evaluate_case_llm(
    case: BenchmarkCase,
    api_key: str,
    ranked_df: pd.DataFrame,
    *,
    client=None,
) -> StageResult:
    """Stage 3: evaluate LLM response for one test case.

    If *client* (an OpenAI instance) is provided it is reused, avoiding
    TCP+TLS connection overhead per call.
    """
    if case.parse_error:
        return _build_invalid_stage_result(case)

    context_text = build_benchmark_context(ranked_df)
    messages = [
        build_benchmark_system_prompt(context_text),
        build_benchmark_user_prompt(case.query),
    ]

    try:
        response_text = create_benchmark_completion(api_key, messages, client=client)
    except Exception as error:
        return StageResult(
            returned_ids=[],
            passed=False,
            missing_ids=case.expected_ids if not case.expects_empty else [],
            unexpected_ids=[],
            raw_text=str(error),
        )

    try:
        returned_ids = parse_benchmark_ids(response_text)
        returned_ids = _resolve_llm_ids(returned_ids, ranked_df)
    except Exception:
        return StageResult(
            returned_ids=[],
            passed=False,
            missing_ids=case.expected_ids if not case.expects_empty else [],
            unexpected_ids=[],
            raw_text=response_text,
        )

    return _build_stage_result(case, returned_ids, raw_text=response_text)


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------
def run_benchmark_suite(
    cases: list[BenchmarkCase],
    embedder,
    reranker,
    courses_df: pd.DataFrame,
    embeddings: np.ndarray,
    api_key: str,
    local_rerank_runtime=None,
    case_limit: int | None = None,
    top_k: int | None = DEFAULT_TOP_K,
    ranking_mode: str = "semantic",
    progress_callback=None,
) -> BenchmarkRunResult:
    """Run the full 3-stage benchmark on all (or a subset of) test cases.

    Performance optimizations applied:
      1. All queries are batch-encoded in a single embedder.encode() call.
      2. Cosine similarity is computed as one matrix multiply for all queries.
      3. A single OpenAI client is reused across all LLM calls.

    Args:
        progress_callback: optional callable(completed, total, case, stage_name)
            called after each sub-step to update UI progress.
    """
    selected = cases if case_limit is None else cases[:case_limit]
    total = len(selected)
    case_results: list[CaseBenchmarkResult] = []

    if progress_callback is not None:
        progress_callback(0, total, None, "init")

    # --- Pre-compute: batch encode + batch cosine similarity ---------------
    # Collect queries (use "" for invalid/parse-error cases to keep indexing)
    queries = [case.query if not case.parse_error else "" for case in selected]
    query_vectors = batch_encode_queries(embedder, queries)  # (N, D)
    sim_matrix = batch_cosine_similarity(query_vectors, embeddings)  # (N, M)

    # --- Pre-compute: reusable OpenAI client --------------------------------
    from openai import OpenAI
    from app_logic.config import OPENROUTER_BASE_URL

    llm_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=60.0)

    # --- Per-case evaluation loop -------------------------------------------
    for idx, case in enumerate(selected):
        # Stage 1: Vector search (from pre-computed scores)
        if progress_callback is not None:
            progress_callback(idx, total, case, "retrieval")

        retrieval_result, candidates_df = evaluate_case_retrieval(
            case, embedder, courses_df, embeddings,
            precomputed_scores=sim_matrix[idx],
        )

        # Stage 2: Reranker
        if progress_callback is not None:
            progress_callback(idx, total, case, "reranker")

        reranker_result, reranked_df = evaluate_case_reranker(
            case,
            candidates_df,
            reranker=reranker,
            local_rerank_runtime=local_rerank_runtime,
            ranking_mode=ranking_mode,
            top_k=top_k,
        )

        # Stage 3: LLM (reuse client)
        if progress_callback is not None:
            progress_callback(idx, total, case, "llm")

        llm_result = evaluate_case_llm(
            case, api_key, reranked_df, client=llm_client,
        )

        case_results.append(CaseBenchmarkResult(
            case=case,
            retrieval=retrieval_result,
            reranker=reranker_result,
            llm=llm_result,
        ))

        if ranking_mode in ("cross_encoder", "local_llm"):
            _cleanup_after_case()

        if progress_callback is not None:
            progress_callback(idx + 1, total, case, "done")

    retrieval_correct = sum(r.retrieval.passed for r in case_results)
    reranker_correct = sum(r.reranker.passed for r in case_results)
    llm_correct = sum(r.llm.passed for r in case_results)

    return BenchmarkRunResult(
        total_cases=total,
        retrieval_correct=retrieval_correct,
        retrieval_incorrect=total - retrieval_correct,
        reranker_correct=reranker_correct,
        reranker_incorrect=total - reranker_correct,
        llm_correct=llm_correct,
        llm_incorrect=total - llm_correct,
        case_results=case_results,
    )


# ---------------------------------------------------------------------------
# Serialization / persistence
# ---------------------------------------------------------------------------
def _case_from_dict(payload: dict) -> BenchmarkCase:
    return BenchmarkCase(**payload)


def _stage_from_dict(payload: dict) -> StageResult:
    return StageResult(**payload)


def _case_result_from_dict(payload: dict) -> CaseBenchmarkResult:
    return CaseBenchmarkResult(
        case=_case_from_dict(payload["case"]),
        retrieval=_stage_from_dict(payload["retrieval"]),
        reranker=_stage_from_dict(payload["reranker"]),
        llm=_stage_from_dict(payload["llm"]),
    )


def serialize_benchmark_run(
    results: BenchmarkRunResult,
    saved_at: str | None = None,
) -> dict:
    return {
        "saved_at": saved_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": asdict(results),
    }


def deserialize_benchmark_run(payload: dict) -> tuple[BenchmarkRunResult, str | None]:
    rp = payload["results"]
    results = BenchmarkRunResult(
        total_cases=rp["total_cases"],
        retrieval_correct=rp["retrieval_correct"],
        retrieval_incorrect=rp["retrieval_incorrect"],
        reranker_correct=rp["reranker_correct"],
        reranker_incorrect=rp["reranker_incorrect"],
        llm_correct=rp["llm_correct"],
        llm_incorrect=rp["llm_incorrect"],
        case_results=[_case_result_from_dict(item) for item in rp["case_results"]],
    )
    return results, payload.get("saved_at")


def save_benchmark_run(
    results: BenchmarkRunResult,
    path: str = BENCHMARK_RUNS_PATH,
) -> str:
    """Append a benchmark run to the JSON results file. Returns saved_at timestamp."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    saved_run = serialize_benchmark_run(results)
    runs: list[dict] = []

    if file_path.exists():
        with open(file_path, encoding="utf-8") as fh:
            runs = json.load(fh)
        if not isinstance(runs, list):
            runs = []

    runs.append(saved_run)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)

    return saved_run["saved_at"]


def load_last_benchmark_run(
    path: str = BENCHMARK_RUNS_PATH,
) -> tuple[BenchmarkRunResult, str | None]:
    """Load the most recent benchmark run from the JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)

    with open(file_path, encoding="utf-8") as fh:
        runs = json.load(fh)

    if not isinstance(runs, list) or not runs:
        raise ValueError("No saved benchmark runs found.")

    return deserialize_benchmark_run(runs[-1])
