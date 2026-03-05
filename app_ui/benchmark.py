"""Streamlit UI components for the benchmark evaluation system.

Provides sidebar controls, progress display with ETA, and tabbed results.
"""

import time

import pandas as pd
import streamlit as st

from app_logic.benchmark import (
    BenchmarkRunResult,
    load_benchmark_cases,
    load_last_benchmark_run,
    run_benchmark_suite,
    save_benchmark_run,
)
from app_logic.config import BENCHMARK_CASES_PATH, BENCHMARK_RUNS_PATH, LOCAL_RERANK_MODEL


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def initialize_benchmark_state() -> None:
    defaults = {
        "benchmark_results": None,
        "benchmark_last_run_at": None,
        "benchmark_case_count": 0,
        "benchmark_source": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_benchmark_case_count() -> int:
    try:
        return len(load_benchmark_cases(BENCHMARK_CASES_PATH))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_benchmark_sidebar(
    api_key: str,
    benchmark_case_count: int,
) -> tuple[bool, bool, int, str, str]:
    """Render benchmark controls.

    Returns:
        (run_clicked, load_clicked, benchmark_limit, ranking_mode, local_model_name)
    """
    ranking_mode_labels = {
        "semantic": "Ilma rerankerita (semantiline jarjekord)",
        "cross_encoder": "Cross-encoder reranker",
        "local_llm": "Kohalik LLM reranker (Transformers)",
    }

    with st.sidebar:
        st.divider()
        st.subheader("Testikomplekt")
        st.caption(
            f"Kasutab faili `{BENCHMARK_CASES_PATH}`, ignoreerib aktiivseid "
            f"filtreid, salvestab jooksutused faili `{BENCHMARK_RUNS_PATH}` "
            "ja kontrollib nii otsingut, rerankerit kui ka LLM-i."
        )
        benchmark_limit = st.slider(
            "Mitu testjuhtumit kaivitada",
            min_value=0,
            max_value=benchmark_case_count,
            value=benchmark_case_count,
            step=1,
            help="Vali mitu esimest testjuhtumit failist jooksutada.",
        )

        ranking_mode = st.radio(
            "Benchmarki järjestusmeetod",
            options=list(ranking_mode_labels.keys()),
            format_func=lambda key: ranking_mode_labels[key],
            index=1,
            help=(
                "Vali, kas benchmark kasutab semantilist jarjekorda, cross-encoderit "
                "voi kohalikku LLM rerankerit."
            ),
        )

        local_rerank_model = LOCAL_RERANK_MODEL
        if ranking_mode == "local_llm":
            local_rerank_model = st.text_input(
                "Benchmarki kohalik rerank mudel",
                value=local_rerank_model,
                help="Naide: Qwen/Qwen3-0.6B voi lokaalne mudelitee.",
            ).strip() or local_rerank_model

        run_clicked = st.button(
            "Kaivita testikomplekt",
            disabled=not bool(api_key) or benchmark_case_count == 0,
            help=(
                None
                if api_key and benchmark_case_count > 0
                else (
                    "Lisa OPENROUTER_API_KEY keskkonnamuutujasse (.env või süsteemi env) "
                    "ja veendu, et benchmark fail sisaldab testjuhtumeid."
                )
            ),
            use_container_width=True,
        )
        load_clicked = st.button(
            "Naita viimast salvestatud tulemust",
            use_container_width=True,
            help="Laeb viimasena faili salvestatud testikomplekti tulemuse.",
        )
        return run_clicked, load_clicked, benchmark_limit, ranking_mode, local_rerank_model


# ---------------------------------------------------------------------------
# Title lookup and formatting helpers
# ---------------------------------------------------------------------------
def build_course_title_lookup(courses_df: pd.DataFrame) -> dict[str, str]:
    """Map aine_kood (uppercased) -> nimi_et for readable benchmark tables."""
    lookup: dict[str, str] = {}
    for _, row in courses_df.iterrows():
        code = str(row.get("aine_kood", "")).strip().upper()
        if not code or code in lookup:
            continue
        title = str(row.get("nimi_et", "")).strip()
        lookup[code] = title if title else "Pealkiri puudub"
    return lookup


def format_id_title_list(values: list[str], lookup: dict[str, str]) -> str:
    if not values:
        return "-"
    parts: list[str] = []
    for value in values:
        code = str(value).strip().upper()
        if not code:
            continue
        parts.append(f"{code} - {lookup.get(code, 'Pealkiri puudub')}")
    return "\n".join(parts) if parts else "-"


def format_percentage(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def format_ratio_percentage(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({format_percentage(numerator, denominator)})"


# ---------------------------------------------------------------------------
# DataFrame builders for the results tabs
# ---------------------------------------------------------------------------
def build_summary_dataframe(
    results: BenchmarkRunResult,
    lookup: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for r in results.case_results:
        rows.append({
            "rea_nr": r.case.row_number,
            "paring": r.case.query,
            "oodatud_kursused": format_id_title_list(r.case.expected_ids, lookup),
            "vektorotsing_ok": "Pass" if r.retrieval.passed else "Fail",
            "reranker_ok": "Pass" if r.reranker.passed else "Fail",
            "lopptulemus_ok": "Pass" if r.llm.passed else "Fail",
            "vektorotsingu_kursused": format_id_title_list(r.retrieval.returned_ids, lookup),
            "rerankeri_kursused": format_id_title_list(r.reranker.returned_ids, lookup),
            "lopptulemus_kursused": format_id_title_list(r.llm.returned_ids, lookup),
            "vektorotsingus_puudu": format_id_title_list(r.retrieval.missing_ids, lookup),
            "rerankeris_puudu": format_id_title_list(r.reranker.missing_ids, lookup),
            "lopptulemus_puudu": format_id_title_list(r.llm.missing_ids, lookup),
        })
    return pd.DataFrame(rows)


def build_stage_dataframe(
    case_results: list,
    stage_name: str,
    lookup: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for r in case_results:
        stage = getattr(r, stage_name)
        rows.append({
            "rea_nr": r.case.row_number,
            "paring": r.case.query,
            "oodatud_kursused": format_id_title_list(r.case.expected_ids, lookup),
            "tagastatud_kursused": format_id_title_list(stage.returned_ids, lookup),
            "puuduolevad_kursused": format_id_title_list(stage.missing_ids, lookup),
        })
    return pd.DataFrame(rows)


def build_reranker_correct_llm_wrong_dataframe(
    case_results: list,
    lookup: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for r in case_results:
        if not r.reranker.passed or r.llm.passed:
            continue
        rows.append({
            "rea_nr": r.case.row_number,
            "paring": r.case.query,
            "oodatud_kursused": format_id_title_list(r.case.expected_ids, lookup),
            "rerankeri_kursused": format_id_title_list(r.reranker.returned_ids, lookup),
            "lopptulemus_kursused": format_id_title_list(r.llm.returned_ids, lookup),
            "lopptulemus_puudu": format_id_title_list(r.llm.missing_ids, lookup),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------
def render_stage_results(
    case_results: list,
    stage_name: str,
    lookup: dict[str, str],
) -> None:
    incorrect = [r for r in case_results if not getattr(r, stage_name).passed]
    correct = [r for r in case_results if getattr(r, stage_name).passed]

    with st.expander("Vigased", expanded=True):
        if incorrect:
            st.dataframe(
                build_stage_dataframe(incorrect, stage_name, lookup),
                hide_index=True,
            )
        else:
            st.success("Vigaseid juhtumeid ei ole.")

    with st.expander("Korrektsed"):
        if correct:
            df = build_stage_dataframe(correct, stage_name, lookup)
            st.dataframe(
                df.drop(columns=["puuduolevad_kursused"], errors="ignore"),
                hide_index=True,
            )
        else:
            st.info("Korrektselt labitud juhtumeid ei ole.")


def render_benchmark_results(courses_df: pd.DataFrame) -> None:
    """Render the full benchmark results page with metrics and tabs."""
    results: BenchmarkRunResult | None = st.session_state.benchmark_results
    if not results:
        return

    lookup = build_course_title_lookup(courses_df)
    reranker_conditional_incorrect = max(0, results.retrieval_correct - results.reranker_correct)
    llm_conditional_incorrect = max(0, results.reranker_correct - results.llm_correct)

    st.subheader("Testikomplekti tulemused")
    if st.session_state.benchmark_last_run_at:
        source = st.session_state.benchmark_source or ""
        caption = f"Viimane tulemus: {st.session_state.benchmark_last_run_at}"
        if source:
            caption = f"{caption}. Allikas: {source}"
        st.caption(caption)

    # Metrics row 1: absolute counts
    cols = st.columns(6)
    cols[0].metric("Vektorotsing õigeid", results.retrieval_correct)
    cols[1].metric("Vektorotsing valesid", results.retrieval_incorrect)
    cols[2].metric("Reranker õigeid", results.reranker_correct)
    cols[3].metric("Reranker valesid", reranker_conditional_incorrect)
    cols[4].metric("LLM õigeid", results.llm_correct)
    cols[5].metric("LLM valesid", llm_conditional_incorrect)

    # Metrics row 2
    row_two = st.columns(3)
    row_two[0].metric(
        "Vektorotsing kokku",
        format_ratio_percentage(results.retrieval_correct, results.total_cases),
    )
    row_two[1].metric(
        "Reranker (ainult retrieval-õiged)",
        format_ratio_percentage(results.reranker_correct, results.retrieval_correct),
    )
    row_two[2].metric(
        "LLM (ainult reranker-õiged)",
        format_ratio_percentage(results.llm_correct, results.reranker_correct),
    )

    # Metrics row 3
    row_three = st.columns(3)
    row_three[1].metric(
        "LLM kokku",
        format_ratio_percentage(results.llm_correct, results.total_cases),
    )

    # Tabs
    summary_tab, retrieval_tab, reranker_tab, llm_tab, rr_ok_llm_fail_tab = st.tabs([
        "Kokkuvote",
        "Vektorotsing",
        "Reranker",
        "LLM lopptulemus",
        "Reranker õige, LLM vale",
    ])

    with summary_tab:
        st.dataframe(
            build_summary_dataframe(results, lookup),
            hide_index=True,
        )

    with retrieval_tab:
        render_stage_results(results.case_results, "retrieval", lookup)

    with reranker_tab:
        render_stage_results(results.case_results, "reranker", lookup)

    with llm_tab:
        render_stage_results(results.case_results, "llm", lookup)

    with rr_ok_llm_fail_tab:
        df = build_reranker_correct_llm_wrong_dataframe(results.case_results, lookup)
        if df.empty:
            st.success("Selliseid juhtumeid ei ole.")
        else:
            st.dataframe(df, hide_index=True)


# ---------------------------------------------------------------------------
# Benchmark runner (with enhanced progress bar + ETA)
# ---------------------------------------------------------------------------
def run_benchmark(
    api_key: str,
    embedder,
    reranker,
    local_rerank_runtime,
    courses_df: pd.DataFrame,
    embeddings,
    benchmark_limit: int,
    ranking_mode: str,
) -> None:
    """Orchestrate a full benchmark run with a live progress bar and ETA."""
    progress_bar = st.progress(0, text="Valmistan testikomplekti ette...")
    progress_status = st.empty()
    start_time = time.time()
    case_times: list[float] = []
    last_case_start = start_time

    def update_progress(completed: int, total: int, case, stage_name: str) -> None:
        nonlocal last_case_start

        if total == 0:
            progress_bar.progress(100, text="Testikomplekt lopetatud: 0 testjuhtumit.")
            progress_status.info("Testjuhtumeid ei valitud.")
            return

        if stage_name == "init":
            progress_bar.progress(0, text=f"Valmistan ette {total} testjuhtumit...")
            progress_status.info(f"Alustan {total} testjuhtumi jooksutamist.")
            return

        if stage_name == "done":
            # Track timing for ETA
            now = time.time()
            case_times.append(now - last_case_start)
            last_case_start = now

        pct = int((completed / total) * 100)
        query_text = case.query[:60] if case else ""

        # Calculate ETA
        eta_text = ""
        if case_times:
            avg_time = sum(case_times) / len(case_times)
            remaining = total - completed
            eta_seconds = avg_time * remaining
            if eta_seconds >= 60:
                eta_text = f" | ~{eta_seconds / 60:.1f} min aega jäänud"
            elif eta_seconds > 0:
                eta_text = f" | ~{eta_seconds:.0f}s aega jäänud"

        stage_labels = {
            "retrieval": "vektorotsing",
            "reranker": "reranker",
            "llm": "LLM",
            "done": "valmis",
        }
        stage_label = stage_labels.get(stage_name, stage_name)

        progress_bar.progress(
            pct,
            text=f"Testikomplekt: {completed}/{total} ({pct}%){eta_text}",
        )
        progress_status.caption(
            f"Juhtum {completed + 1 if stage_name != 'done' else completed}/{total}: "
            f"{query_text} [{stage_label}]"
        )

    try:
        cases = load_benchmark_cases(BENCHMARK_CASES_PATH)
        st.session_state.benchmark_results = run_benchmark_suite(
            cases=cases,
            embedder=embedder,
            reranker=reranker,
            local_rerank_runtime=local_rerank_runtime,
            courses_df=courses_df,
            embeddings=embeddings,
            api_key=api_key,
            case_limit=benchmark_limit,
            ranking_mode=ranking_mode,
            progress_callback=update_progress,
        )
        st.session_state.benchmark_last_run_at = save_benchmark_run(
            st.session_state.benchmark_results,
        )
        st.session_state.benchmark_source = f"salvestatud faili `{BENCHMARK_RUNS_PATH}`"

        elapsed = time.time() - start_time
        if elapsed >= 60:
            elapsed_text = f"{elapsed / 60:.1f} minutit"
        else:
            elapsed_text = f"{elapsed:.0f} sekundit"

        if benchmark_limit > 0:
            progress_bar.progress(
                100,
                text=f"Testikomplekt lopetatud: {benchmark_limit} testjuhtumit ({elapsed_text}).",
            )
            progress_status.success(
                "Testikomplekti jooksutamine valmis ja tulemus salvestati faili."
            )
    except FileNotFoundError:
        st.session_state.benchmark_results = None
        progress_bar.empty()
        progress_status.empty()
        st.error(f"Testikomplekti faili `{BENCHMARK_CASES_PATH}` ei leitud.")
    except Exception as error:
        st.session_state.benchmark_results = None
        progress_bar.empty()
        progress_status.empty()
        st.error(f"Testikomplekti jooksutamine ebaonnestus: {error}")


def load_saved_benchmark() -> None:
    """Load and display the most recent saved benchmark result."""
    try:
        results, saved_at = load_last_benchmark_run(BENCHMARK_RUNS_PATH)
        st.session_state.benchmark_results = results
        st.session_state.benchmark_last_run_at = saved_at
        st.session_state.benchmark_source = f"laetud failist `{BENCHMARK_RUNS_PATH}`"
        st.success("Viimane salvestatud testikomplekti tulemus laetud.")
    except FileNotFoundError:
        st.error(f"Salvestatud tulemuste faili `{BENCHMARK_RUNS_PATH}` ei leitud.")
    except Exception as error:
        st.error(f"Salvestatud tulemuse laadimine ebaonnestus: {error}")
