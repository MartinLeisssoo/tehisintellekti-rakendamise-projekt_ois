"""Tartu Ülikooli kursuste soovitaja – Streamlit UI shell.

All business logic lives in app_logic/; this file handles only Streamlit
page layout, session state, sidebar widgets, chat rendering, and
benchmark orchestration.
"""

import os
import re
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import streamlit_shadcn_ui as ui
except Exception:  # pragma: no cover - optional UI dependency
    ui = None

# Suppress tqdm progress bars that cause BrokenPipeError in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")

from app_logic.config import (
    DATA_PATH,
    DEFAULT_TOP_K,
    EMBED_MODEL,
    EMBEDDINGS_PATH,
    LLM_MODEL,
    LOCAL_RERANK_MODEL,
    MODEL_PRICING,
    RERANKER_MODEL,
)
from app_logic.data import load_courses, load_embeddings
from app_logic.feedback import log_feedback
from app_logic.filters import apply_filters, format_active_filters
from app_logic.llm import build_system_prompt, create_response_stream, detect_language
from app_logic.retrieval import (
    build_course_context,
    get_semantic_candidates,
    load_local_transformers_reranker,
    rerank_candidates,
    rerank_candidates_with_local_llm,
    select_semantic_results,
)
from app_ui.benchmark import (
    get_benchmark_case_count,
    initialize_benchmark_state,
    load_saved_benchmark,
    render_benchmark_results,
    render_benchmark_sidebar,
    run_benchmark,
)


# ---------- Token counting ----------
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _read_env_file_value(name: str, env_path: str = ".env") -> str:
    """Read a key from local .env file without extra dependencies."""
    path = Path(env_path)
    if not path.exists():
        return ""
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != name:
                continue
            cleaned = value.strip().strip('"').strip("'")
            return cleaned
    except Exception:
        return ""
    return ""


def _resolve_openrouter_api_key() -> str:
    """Resolve OpenRouter API key from env, secrets, or local .env file."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    try:
        key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass

    return _read_env_file_value("OPENROUTER_API_KEY")


_RANKING_MODE_LABELS = {
    "semantic": "Kiire: semantiline otsing",
    "cross_encoder": "Täpsus: cross-encoder reranker",
    "local_llm": "Kohalik LLM reranker (Transformers)",
}


def _confidence_to_score_10(confidence: np.ndarray) -> np.ndarray:
    """Map confidence values (0..1) to integer 1..10 suitability score."""
    clipped = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    return np.clip(np.rint(clipped * 9.0 + 1.0), 1, 10).astype(int)


def _score_pill_colors(confidence: float) -> tuple[str, str, str]:
    """Return subtle pastel badge colors for a confidence value."""
    c = float(np.clip(confidence, 0.0, 1.0))
    hue = 12.0 + (118.0 * c)
    lightness = 95.0 - (8.0 * c)
    border_lightness = max(74.0, lightness - 10.0)
    bg = f"hsl({hue:.0f}, 55%, {lightness:.0f}%)"
    border = f"hsl({hue:.0f}, 35%, {border_lightness:.0f}%)"
    text = "#334155"
    return bg, border, text


def _normalize_course_code(value: str) -> str:
    """Normalize course code for robust matching across outputs."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def _parse_llm_course_details(response_text: str) -> dict[str, dict[str, str | int]]:
    """Extract per-course score, overview, goals, and relevance from LLM output.

    Handles both ET and EN label variants.  Score can appear as
    ``*Sobivus:* 8/10`` or ``*Suitability:* 8/10`` etc.
    """
    details: dict[str, dict[str, str | int]] = {}
    current_code = ""

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Course header: "- **LTAT.03.003 – Kursuse nimi** (6 EAP, kevad)"
        header_match = re.match(r"^-\s*\*\*(.+?)\*\*", line)
        if header_match:
            header = header_match.group(1).strip()
            # Split on dash/en-dash between code and name
            code_part = re.split(r"\s*[\-–—]\s+", header, maxsplit=1)[0].strip()
            current_code = _normalize_course_code(code_part)
            if current_code:
                details.setdefault(current_code, {})
            continue

        if not current_code:
            continue

        # Strip leading bullet prefix "  - " for sub-items
        stripped = re.sub(r"^\s*-\s+", "", line)

        # Score line: "*Sobivus:* 8/10" or "*Suitability:* 9/10"
        score_match = re.search(
            r"(?:\*?Sobivus\*?|\*?Suitability\*?)\s*:?\*?\s*(\d{1,2})\s*/\s*10",
            stripped,
            re.IGNORECASE,
        )
        if score_match:
            score_10 = max(1, min(10, int(score_match.group(1))))
            details[current_code]["score_10"] = score_10
            continue

        # Goals line: "*Eesmärgid:* ..." or "*Goals:* ..."
        goals_match = re.match(
            r'\*?(?:Eesm[aä]rgi?d|Goals)\*?\s*:?\*?\s*["\u201c]?(.+?)["\u201d]?\s*$',
            stripped,
            re.IGNORECASE,
        )
        if goals_match:
            details[current_code]["goals"] = goals_match.group(1).strip().strip('"')
            continue

        # Relevance line: "*Asjakohasus:* ..." or "*Relevance:* ..."
        relevance_match = re.match(
            r'\*?(?:Asjakohasus|Relevance)\*?\s*:?\*?\s*["\u201c]?(.+?)["\u201d]?\s*$',
            stripped,
            re.IGNORECASE,
        )
        if relevance_match:
            details[current_code]["relevance"] = relevance_match.group(1).strip().strip('"')
            continue

        # Skip if line still contains bold markers (sub-header, not overview)
        if "**" in stripped:
            continue

        # The remaining sub-bullet is the one-sentence overview
        overview = stripped.strip().strip('"')
        if overview and "overview" not in details[current_code]:
            details[current_code]["overview"] = overview

    return details


def _maybe_auto_release_reranker_memory(
    current_mode: str,
    current_local_model: str,
    enabled: bool,
) -> None:
    """Auto-clear cached ranking models when switching ranking mode."""
    previous_mode = st.session_state.get("_previous_ranking_mode")
    previous_local_model = st.session_state.get("_previous_local_rerank_model")
    if enabled:
        if previous_mode != current_mode:
            if previous_mode == "cross_encoder":
                _load_reranker.clear()
            elif previous_mode == "local_llm":
                _load_local_llm_reranker.clear()
        elif (
            current_mode == "local_llm"
            and previous_local_model
            and previous_local_model != current_local_model
        ):
            _load_local_llm_reranker.clear()
    st.session_state["_previous_ranking_mode"] = current_mode
    st.session_state["_previous_local_rerank_model"] = current_local_model


def _release_torch_cache() -> None:
    """Release transient torch cache (useful after heavy reranking calls)."""
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------- Cached loaders (Streamlit wrappers around pure functions) ----------
@st.cache_data
def _load_courses() -> pd.DataFrame:
    return load_courses(DATA_PATH)


@st.cache_data
def _load_embeddings() -> np.ndarray:
    return load_embeddings(EMBEDDINGS_PATH)


@st.cache_resource
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, model_kwargs={"torch_dtype": "float16"})


@st.cache_resource
def _load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, model_kwargs={"torch_dtype": "float16"})


@st.cache_resource
def _load_local_llm_reranker(model_name: str):
    return load_local_transformers_reranker(model_name)


# ---------- Session state ----------
def _initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    initialize_benchmark_state()


# ---------- Sidebar: filter options derived from data ----------
def _derive_filter_options(df: pd.DataFrame) -> dict:
    semesters = df["semester"].dropna().astype(str).str.lower().unique().tolist()
    semester_options = sorted(set(semesters))
    if "kevad" in semester_options:
        semester_options.remove("kevad")
        semester_options.insert(0, "kevad")

    eap_series = pd.to_numeric(df["eap"], errors="coerce")
    if eap_series.notna().any():
        eap_min = float(np.nanmin(eap_series))
        eap_max = float(np.nanmax(eap_series))
    else:
        eap_min, eap_max = 0.0, 12.0

    teaching_options = sorted(df["oppeviis"].dropna().unique().tolist())
    city_options = sorted(
        df["linn"].dropna().value_counts()[lambda s: s >= 5].index.tolist()
    )
    domain_options = sorted(
        df["valdkond"].dropna().value_counts()[lambda s: s >= 10].index.tolist()
    )
    _level_keys = [
        "bakalaureuseõpe", "magistriõpe", "doktoriõpe",
        "rakenduskõrgharidusõpe", "integreeritud bakalaureuse- ja magistriõpe",
    ]
    study_level_options = [
        k for k in _level_keys
        if df["oppeaste"].str.contains(re.escape(k), case=False, na=False).any()
    ]
    language_options = ["eesti keel", "inglise keel", "vene keel", "saksa keel"]

    return {
        "semester_options": semester_options,
        "eap_min": eap_min,
        "eap_max": eap_max,
        "teaching_options": teaching_options,
        "city_options": city_options,
        "domain_options": domain_options,
        "study_level_options": study_level_options,
        "language_options": language_options,
    }


_FALLBACK_OPTIONS = {
    "semester_options": ["kevad", "sügis"],
    "eap_min": 0.0,
    "eap_max": 12.0,
    "teaching_options": [],
    "city_options": [],
    "domain_options": [],
    "study_level_options": [],
    "language_options": [],
}


# ---------- Sidebar rendering ----------
def _render_sidebar(
    df: pd.DataFrame | None,
    opts: dict,
    api_key: str,
    developer_mode: bool,
) -> dict:
    """Render sidebar widgets and return collected filter values."""
    with st.sidebar:
        if not api_key:
            st.warning("OpenRouter API key puudub.")
            with st.expander("Kuidas API võti lisada"):
                st.code("OPENROUTER_API_KEY=sk-or-v1-...", language="bash")
                st.caption("Lisa see `.env` faili projekti juuresse või süsteemi keskkonnamuutujana.")
        if st.button("Uus vestlus"):
            st.session_state.messages = []
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.rerun()
        show_debug = st.checkbox("Näita silumisinfot", value=True) if developer_mode else False
        st.divider()
        st.markdown(f"**Mudel:** `{LLM_MODEL}`")

        selected_semesters = st.multiselect(
            "Semester",
            opts["semester_options"],
            default=opts["semester_options"],
        )
        eap_range = st.slider(
            "EAP vahemik",
            min_value=opts["eap_min"],
            max_value=opts["eap_max"],
            value=(opts["eap_min"], opts["eap_max"]),
            step=0.5,
        )
        grading_choice = st.radio(
            "Hindamisskaala",
            ["Koik", "Eristav", "Mitteeristav"],
            index=0,
        )
        result_mode = st.radio(
            "Tulemuste valik",
            ["Tulemuste arv", "Ainult täpseimad vasted"],
            horizontal=True,
        )
        if result_mode == "Tulemuste arv":
            top_k = st.slider("Tulemuste arv", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        else:
            top_k = None

        with st.expander("Täpsemad filtrid"):
            selected_languages = st.multiselect("Õppekeel", opts["language_options"])
            selected_cities = st.multiselect("Linn", opts["city_options"])
            selected_levels = st.multiselect("Õppeaste", opts["study_level_options"])
            selected_teaching = st.multiselect("Õppeviis", opts["teaching_options"])
            selected_domains = st.multiselect("Valdkond", opts["domain_options"])

        if developer_mode:
            ranking_mode = st.radio(
                "Järjestusmeetod",
                options=list(_RANKING_MODE_LABELS.keys()),
                format_func=lambda key: _RANKING_MODE_LABELS[key],
                index=1,
                help=(
                    "Cross-encoder kasutab rohkem mälu. Kohalik LLM kasutab Hugging Face Transformers mudelit."
                ),
            )
            local_rerank_model = LOCAL_RERANK_MODEL
            if ranking_mode == "local_llm":
                local_rerank_model = st.text_input(
                    "Kohalik rerank mudel",
                    value=LOCAL_RERANK_MODEL,
                    help="Näide: Qwen/Qwen3-0.6B või lokaalne mudelitee.",
                ).strip() or LOCAL_RERANK_MODEL

            if st.button("Vabasta järjestusmudelite mälu"):
                _load_reranker.clear()
                _load_local_llm_reranker.clear()
                st.success("Järjestusmudelite cache tühjendatud.")

            auto_release_reranker = st.checkbox(
                "Vabasta rerankeri mälu automaatselt",
                value=True,
                help="Kui lahkud cross-encoder või kohaliku LLM-i režiimist, vabastatakse vastava mudeli cache automaatselt.",
            )
        else:
            ranking_mode = "cross_encoder"
            local_rerank_model = LOCAL_RERANK_MODEL
            auto_release_reranker = True

        # Active-filter summary
        active_filters_str = format_active_filters(
            selected_semesters, eap_range, grading_choice,
            languages=selected_languages,
            cities=selected_cities,
            study_levels=selected_levels,
            teaching_methods=selected_teaching,
            domains=selected_domains,
        )

        # Matching course count
        if df is not None:
            match_mask = apply_filters(
                df, selected_semesters, eap_range, grading_choice,
                languages=selected_languages,
                cities=selected_cities,
                study_levels=selected_levels,
                teaching_methods=selected_teaching,
                domains=selected_domains,
            )
            st.caption(f"{int(match_mask.sum())} kursust vastab filtritele")

        # Token / cost display
        st.divider()
        in_tok = st.session_state.get("total_input_tokens", 0)
        out_tok = st.session_state.get("total_output_tokens", 0)
        price_in, price_out = MODEL_PRICING.get(LLM_MODEL, (0, 0))
        cost = in_tok * price_in / 1_000_000 + out_tok * price_out / 1_000_000
        if ui is not None:
            ui.metric_card(
                title="Tokeneid",
                content=f"{in_tok + out_tok}",
                description=f"{in_tok} sisse / {out_tok} välja",
                key="sidebar_tokens",
            )
            ui.metric_card(
                title="Hinnanguline kulu",
                content=f"${cost:.6f}",
                description=f"Mudel: {LLM_MODEL.split('/')[-1]}",
                key="sidebar_cost",
            )
        else:
            st.caption(f"Tokeneid: {in_tok} sisse / {out_tok} välja")
            st.caption(f"Hinnanguline kulu: ${cost:.6f}")

    return {
        "api_key": api_key,
        "show_debug": show_debug,
        "selected_semesters": selected_semesters,
        "eap_range": eap_range,
        "grading_choice": grading_choice,
        "top_k": top_k,
        "selected_languages": selected_languages,
        "selected_cities": selected_cities,
        "selected_levels": selected_levels,
        "selected_teaching": selected_teaching,
        "selected_domains": selected_domains,
        "ranking_mode": ranking_mode,
        "local_rerank_model": local_rerank_model,
        "auto_release_reranker": auto_release_reranker,
        "active_filters_str": active_filters_str,
    }


# ---------- Global UI styles (injected once per render) ----------
def _inject_global_styles() -> None:
    """Inject page-wide CSS for chat, cards, tooltips, input, and typography."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* ---- Page-level typography ---- */
        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", "Avenir Next", "Helvetica Neue", system-ui, sans-serif;
        }
        h1 {
            font-family: "Space Grotesk", "Avenir Next", "Helvetica Neue", sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            font-size: 1.7rem !important;
            margin-bottom: 0 !important;
        }

        /* ---- Hide chat avatars (robot / human face) ---- */
        .stChatMessage [data-testid="chatAvatarIcon-assistant"],
        .stChatMessage [data-testid="chatAvatarIcon-user"],
        .stChatMessage .stChatMessageAvatar,
        /* Broader selector for avatar container */
        [data-testid="stChatMessageAvatarContainer"] {
            display: none !important;
        }
        /* Remove the left gap that the avatar used to occupy */
        .stChatMessage [data-testid="stChatMessageContent"] {
            margin-left: 0 !important;
        }

        /* ---- User message bubble ---- */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
        .stChatMessage[data-testid="stChatMessage"]:nth-of-type(odd) {
            background: transparent !important;
            border: none !important;
        }
        .stChatMessage {
            padding: 0.35rem 0 !important;
            gap: 0 !important;
            background: transparent !important;
        }
        /* Tighter gap between consecutive messages */
        .stChatMessage + .stChatMessage { margin-top: -0.2rem; }

        /* ---- Chat input box ---- */
        [data-testid="stChatInput"] {
            border-radius: 12px !important;
            border: 1.5px solid #d4dae2 !important;
            padding: 0.1rem 0.2rem !important;
            box-shadow: 0 1px 3px rgba(2,6,23,0.05) !important;
            transition: border-color 150ms ease !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #94a3b8 !important;
            box-shadow: 0 0 0 2px rgba(148,163,184,0.15) !important;
        }
        [data-testid="stChatInput"] textarea {
            font-family: "IBM Plex Sans", "Avenir Next", "Helvetica Neue", sans-serif !important;
            font-size: 0.92rem !important;
        }
        /* Hide the default send-button icon styling noise */
        [data-testid="stChatInput"] button {
            color: #64748b !important;
        }

        /* ---- Course card styles ---- */
        .course-results-heading {
            margin: 0.3rem 0 0.55rem;
            font-family: "Space Grotesk", "Avenir Next", "Helvetica Neue", sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: #0f172a;
        }
        .course-card {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
            margin: 0 0 0.38rem 0;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(2, 6, 23, 0.03);
            transition: border-color 120ms ease, box-shadow 120ms ease;
            position: relative;
        }
        .course-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 2px 6px rgba(2, 6, 23, 0.06);
        }
        .course-card-head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.5rem;
        }
        .course-title {
            font-family: "Space Grotesk", "Avenir Next", "Helvetica Neue", sans-serif;
            line-height: 1.28;
            font-size: 0.94rem;
            font-weight: 600;
            color: #0f172a;
        }
        .course-code-link {
            color: #1e4f78;
            font-weight: 700;
            text-decoration: underline;
            text-underline-offset: 2px;
            cursor: pointer;
        }
        .course-code-link.no-link {
            text-decoration: none;
            color: #1e4f78;
        }
        .course-name {
            color: #1e293b;
            font-weight: 500;
        }
        .course-overview {
            margin-top: 0.18rem;
            font-family: "IBM Plex Sans", "Avenir Next", "Helvetica Neue", sans-serif;
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.38;
        }
        .course-meta {
            margin-top: 0.14rem;
            font-family: "IBM Plex Sans", "Avenir Next", "Helvetica Neue", sans-serif;
            color: #94a3b8;
            font-size: 0.76rem;
        }
        .score-pill {
            font-family: "Space Grotesk", "Avenir Next", "Helvetica Neue", sans-serif;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2rem;
            border-radius: 999px;
            padding: 0.02rem 0.36rem;
            border: 1px solid transparent;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
            cursor: help;
            flex-shrink: 0;
        }

        /* ---- Custom CSS tooltip ---- */
        .course-card .card-tooltip {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            left: 0;
            right: 0;
            top: 100%;
            z-index: 1000;
            margin-top: 4px;
            padding: 0.65rem 0.8rem;
            background: #1e293b;
            color: #f1f5f9;
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(2, 6, 23, 0.22);
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.82rem;
            line-height: 1.52;
            pointer-events: none;
            transition: opacity 140ms ease, visibility 0s 140ms;
            max-width: 480px;
            width: 100%;
        }
        .course-card:hover .card-tooltip {
            visibility: visible;
            opacity: 1;
            transition: opacity 140ms ease 180ms, visibility 0s 180ms;
        }
        .card-tooltip .tt-label {
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.1rem;
            display: block;
        }
        .card-tooltip .tt-label:not(:first-child) {
            margin-top: 0.35rem;
        }
        .card-tooltip .tt-text {
            color: #e2e8f0;
            font-size: 0.82rem;
            line-height: 1.48;
        }

        /* ---- Responsive ---- */
        @media (max-width: 900px) {
            .course-results-heading { font-size: 1.05rem; }
            .course-title { font-size: 0.88rem; }
            .course-overview { font-size: 0.8rem; }
            .score-pill { font-size: 0.7rem; min-width: 1.8rem; }
            .course-card .card-tooltip { font-size: 0.78rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Course cards ----------
def _render_course_cards(results_df: pd.DataFrame, card_prefix: str = "card") -> None:
    """Display course cards with linked code, CSS tooltip, and score badge."""
    if results_df is None or results_df.empty:
        return
    _ = card_prefix  # reserved for compatibility

    st.markdown(
        "<div class='course-results-heading'>Leitud kursused</div>",
        unsafe_allow_html=True,
    )

    for _, row in results_df.iterrows():
        code = str(row.get("aine_kood", "")).strip()
        name = str(row.get("nimi_et", "")).strip()
        if not name or name.lower() == "nan":
            name = str(row.get("nimi_en", "")).strip()
        if name.lower() == "nan":
            name = ""

        url = str(row.get("ois_url", "")).strip()
        eap = str(row.get("eap", "")).strip()
        semester = str(row.get("semester", "")).strip()
        level = str(row.get("oppeaste", "")).strip()

        meta_parts: list[str] = []
        if eap and eap.lower() != "nan":
            meta_parts.append(f"{eap} EAP")
        if semester and semester.lower() != "nan":
            meta_parts.append(semester)
        if level and level.lower() != "nan":
            meta_parts.append(level)
        meta_line = " \u00b7 ".join(meta_parts)

        overview = str(row.get("_llm_overview", "")).strip()
        if not overview or overview.lower() == "nan":
            overview = str(row.get("eesmark_et", "")).strip()
        if overview.lower() == "nan":
            overview = ""
        overview = re.sub(r"\s+", " ", overview)
        if len(overview) > 220:
            overview = overview[:217].rstrip() + "\u2026"

        # --- Build tooltip sections ---
        tooltip_parts: list[str] = []
        goals_text = str(row.get("_llm_goals", "")).strip()
        if goals_text and goals_text.lower() != "nan":
            tooltip_parts.append(
                f"<span class='tt-label'>Eesm\u00e4rgid</span>"
                f"<span class='tt-text'>{html.escape(goals_text)}</span>"
            )
        relevance_text = str(row.get("_llm_relevance", "")).strip()
        if relevance_text and relevance_text.lower() != "nan":
            tooltip_parts.append(
                f"<span class='tt-label'>Asjakohasus</span>"
                f"<span class='tt-text'>{html.escape(relevance_text)}</span>"
            )
        if overview:
            tooltip_parts.append(
                f"<span class='tt-label'>Kokkuv\u00f5te</span>"
                f"<span class='tt-text'>{html.escape(overview)}</span>"
            )
        if meta_line:
            tooltip_parts.append(
                f"<span class='tt-label'>Info</span>"
                f"<span class='tt-text'>{html.escape(meta_line)}</span>"
            )
        tooltip_html = "".join(tooltip_parts) if tooltip_parts else ""
        tooltip_block = f"<div class='card-tooltip'>{tooltip_html}</div>" if tooltip_html else ""

        # --- Score ---
        score_value = row.get("_display_score_10", row.get("_match_score_10", np.nan))
        if pd.notna(score_value):
            score_10 = max(1, min(10, int(float(score_value))))
        else:
            score_10 = 6
        confidence = float((score_10 - 1) / 9)
        bg_color, border_color, text_color = _score_pill_colors(confidence)

        # --- Title HTML ---
        code_html = html.escape(code)
        name_html = html.escape(name)
        if code and url:
            safe_url = html.escape(url, quote=True)
            code_segment = (
                f"<a class='course-code-link' href='{safe_url}' target='_blank'>"
                f"{code_html}</a>"
            )
        elif code:
            code_segment = f"<span class='course-code-link no-link'>{code_html}</span>"
        else:
            code_segment = ""

        if code_segment and name:
            title_html = f"{code_segment}<span class='course-name'> \u2014 {name_html}</span>"
        elif code_segment:
            title_html = code_segment
        else:
            title_html = f"<span class='course-name'>{name_html}</span>"

        overview_html = html.escape(overview)
        overview_block = (
            f"<div class='course-overview'>{overview_html}</div>" if overview_html else ""
        )
        meta_html = html.escape(meta_line)

        st.markdown(
            f"""
            <div class="course-card">
              <div class="course-card-head">
                <div class="course-title">{title_html}</div>
                <span class="score-pill" style="background:{bg_color}; border-color:{border_color}; color:{text_color};">{score_10}/10</span>
              </div>
              {overview_block}
              <div class="course-meta">{meta_html}</div>
              {tooltip_block}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Debug / feedback expanders ----------
def _render_debug_info(debug: dict, index: int) -> None:
    with st.expander("Vaata kapoti alla (RAG ja filtrid)"):
        st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', 'Info puudub')}")
        ranking_mode = debug.get("ranking_mode")
        if ranking_mode:
            st.caption(f"**Järjestusmeetod:** {_RANKING_MODE_LABELS.get(ranking_mode, ranking_mode)}")
        st.write(f"Filtreeritud kursuste arv: **{debug.get('filtered_count', 0)}**")
        st.write(f"Kandidaatide arv (semantiline otsing): **{debug.get('candidate_count', 0)}**")

        st.write("**Lõpptulemused (järjestatud):**")
        rdf = debug.get("results_df")
        if rdf is not None and not rdf.empty:
            display_cols = ["aine_kood", "nimi_et", "eap", "semester", "oppeaste"]
            cols_to_show = [c for c in display_cols if c in rdf.columns]
            st.dataframe(rdf[cols_to_show], hide_index=True)
        else:
            st.warning("Ühtegi kursust ei leitud.")

        st.text_area(
            "LLM-ile saadetud süsteemiviip:",
            debug.get("system_prompt", ""),
            height=150,
            disabled=True,
            key=f"prompt_area_{index}",
        )


def _render_feedback_form(debug: dict, msg: dict, index: int) -> None:
    with st.expander("Hinda vastust"):
        with st.form(key=f"feedback_form_{index}"):
            rating = st.radio(
                "Hinnang vastusele:",
                ["Hea", "Halb"],
                horizontal=True,
                key=f"rating_{index}",
            )
            error_cat = st.selectbox(
                "Kui vastus oli halb, mis läks valesti?",
                [
                    "",
                    "Filtrid olid liiga karmid/valed",
                    "RAG otsing leidis valed ained",
                    "LLM hallutsineeris/vastas valesti",
                ],
                key=f"kato_{index}",
            )
            comment = st.text_area(
                "Vaba kommentaar (valikuline):",
                key=f"comment_{index}",
                height=80,
            )
            if st.form_submit_button("Salvesta hinnang"):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rdf = debug.get("results_df")
                ctx_ids = rdf["aine_kood"].tolist() if (rdf is not None and not rdf.empty) else []
                ctx_names = rdf["nimi_et"].tolist() if (rdf is not None and not rdf.empty) else []
                log_feedback(
                    ts, debug.get("user_prompt", ""),
                    debug.get("filters", ""), ctx_ids, ctx_names,
                    msg["content"], rating, error_cat, comment,
                )
                st.success("Tagasiside salvestatud!")


# ---------- Chat history ----------
def _render_chat_history(show_debug: bool, developer_mode: bool) -> None:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            has_course_results = (
                msg.get("role") == "assistant"
                and "debug_info" in msg
                and msg["debug_info"].get("results_df") is not None
            )

            if not has_course_results:
                st.markdown(msg["content"])

            if msg["role"] == "assistant" and "debug_info" in msg:
                _render_course_cards(msg["debug_info"].get("results_df"), card_prefix=f"hist_{i}")
                if show_debug:
                    _render_debug_info(msg["debug_info"], i)
                    if developer_mode:
                        _render_feedback_form(msg["debug_info"], msg, i)


# ---------- Handle user prompt ----------
def _handle_user_prompt(
    prompt: str,
    sidebar: dict,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        api_key = sidebar["api_key"]
        if not api_key:
            error_msg = (
                "OpenRouter API võti puudub. Lisa `OPENROUTER_API_KEY=sk-or-v1-...` "
                "keskkonnamuutujana, `.env` faili või Streamlit secretsi."
            )
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return

        context_text = None

        with st.spinner("Otsin sobivaid kursusi..."):
            # 1. Apply sidebar filters
            mask = apply_filters(
                df,
                sidebar["selected_semesters"],
                sidebar["eap_range"],
                sidebar["grading_choice"],
                languages=sidebar["selected_languages"],
                cities=sidebar["selected_cities"],
                study_levels=sidebar["selected_levels"],
                teaching_methods=sidebar["selected_teaching"],
                domains=sidebar["selected_domains"],
            )
            filtered_df = df[mask].copy()
            filtered_embeddings = embeddings[mask.values]

            if filtered_df.empty:
                st.warning("Filtritele vastavaid kursusi ei leitud.")
            else:
                # 2. Semantic search
                embedder = _load_embedder()
                candidates_df, candidate_scores = get_semantic_candidates(
                    embedder, prompt, filtered_df, filtered_embeddings,
                )
                candidate_count = len(candidates_df)

                ranking_mode = sidebar["ranking_mode"]
                match_confidence = np.array([], dtype=float)
                if ranking_mode == "cross_encoder":
                    reranker = _load_reranker()
                    rerank_result = rerank_candidates(
                        reranker, prompt, candidates_df, top_k=sidebar["top_k"],
                        return_scores=True,
                    )
                    if isinstance(rerank_result, tuple):
                        results_df, match_confidence = rerank_result
                    else:
                        results_df = rerank_result
                    _release_torch_cache()
                elif ranking_mode == "local_llm":
                    try:
                        llm_top_k = sidebar["top_k"]
                        if llm_top_k is None:
                            semantic_pick = select_semantic_results(
                                candidates_df,
                                candidate_scores,
                                top_k=None,
                                return_confidence=True,
                            )
                            if isinstance(semantic_pick, tuple):
                                llm_candidates, llm_confidence = semantic_pick
                            else:
                                llm_candidates = semantic_pick
                                llm_confidence = np.array([], dtype=float)
                            llm_top_k = len(llm_candidates)
                        else:
                            llm_candidates = candidates_df
                            llm_confidence = np.clip(
                                (np.asarray(candidate_scores, dtype=float) + 1.0) / 2.0,
                                0.0,
                                1.0,
                            )

                        local_runtime = _load_local_llm_reranker(sidebar["local_rerank_model"])
                        llm_result = rerank_candidates_with_local_llm(
                            prompt,
                            llm_candidates,
                            rerank_runtime=local_runtime,
                            top_k=llm_top_k,
                            return_candidate_indices=True,
                        )
                        if isinstance(llm_result, tuple):
                            results_df, selected_indices = llm_result
                            if len(llm_confidence):
                                match_confidence = np.asarray(llm_confidence, dtype=float)[selected_indices]
                        else:
                            results_df = llm_result
                    except Exception as llm_rerank_error:
                        st.warning(
                            "Kohalik rerank ebaõnnestus, kasutan semantilist järjestust. "
                            f"({llm_rerank_error})"
                        )
                        semantic_result = select_semantic_results(
                            candidates_df,
                            candidate_scores,
                            sidebar["top_k"],
                            return_confidence=True,
                        )
                        if isinstance(semantic_result, tuple):
                            results_df, match_confidence = semantic_result
                        else:
                            results_df = semantic_result
                else:
                    semantic_result = select_semantic_results(
                        candidates_df,
                        candidate_scores,
                        sidebar["top_k"],
                        return_confidence=True,
                    )
                    if isinstance(semantic_result, tuple):
                        results_df, match_confidence = semantic_result
                    else:
                        results_df = semantic_result

                results_df = results_df.copy().reset_index(drop=True)
                confidence = np.asarray(match_confidence, dtype=float)
                if confidence.size == 0:
                    confidence = np.full(len(results_df), 0.5, dtype=float)
                elif confidence.size != len(results_df):
                    confidence = confidence[:len(results_df)]
                    if confidence.size < len(results_df):
                        confidence = np.pad(
                            confidence,
                            (0, len(results_df) - confidence.size),
                            constant_values=0.5,
                        )

                confidence = np.clip(confidence, 0.0, 1.0)
                results_df["_match_confidence"] = confidence
                results_df["_match_score_10"] = _confidence_to_score_10(confidence)

                context_text = build_course_context(results_df)
                st.caption(
                    f"Näitan {len(results_df)} kursust {len(filtered_df)}-st "
                    f"({_RANKING_MODE_LABELS[ranking_mode]})"
                )

        if context_text is None:
            no_results = "Sobivaid kursusi ei leitud praeguste filtritega."
            st.markdown(no_results)
            st.session_state.messages.append({"role": "assistant", "content": no_results})
            return

        # 4. LLM response
        try:
            response_lang = detect_language(prompt)
            rec_count = len(results_df)
            system_content = build_system_prompt(
                context_text, sidebar["active_filters_str"], rec_count, response_lang,
            )

            messages_to_send = [
                {"role": "system", "content": system_content},
            ] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            stream = create_response_stream(api_key, messages_to_send)

            full_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            response = full_response

            # Token accounting (approximate)
            input_text = "".join(m["content"] for m in messages_to_send)
            st.session_state.total_input_tokens += _count_tokens(input_text)
            st.session_state.total_output_tokens += _count_tokens(response)

            llm_details = _parse_llm_course_details(response)
            llm_overviews: list[str] = []
            llm_scores: list[float] = []
            llm_goals: list[str] = []
            llm_relevances: list[str] = []
            for _, course_row in results_df.iterrows():
                code_key = _normalize_course_code(str(course_row.get("aine_kood", "")))
                item = llm_details.get(code_key, {})
                llm_overviews.append(str(item.get("overview", "")).strip())
                llm_goals.append(str(item.get("goals", "")).strip())
                llm_relevances.append(str(item.get("relevance", "")).strip())
                score_value = item.get("score_10")
                llm_scores.append(float(score_value) if isinstance(score_value, int) else np.nan)

            results_df["_llm_overview"] = llm_overviews
            results_df["_llm_score_10"] = llm_scores
            results_df["_llm_goals"] = llm_goals
            results_df["_llm_relevance"] = llm_relevances

            llm_score_series = pd.to_numeric(results_df["_llm_score_10"], errors="coerce")
            match_score_series = pd.to_numeric(results_df["_match_score_10"], errors="coerce")
            results_df["_display_score_10"] = llm_score_series.fillna(match_score_series).fillna(6).astype(int)

            display_cols = [
                "aine_kood",
                "nimi_et",
                "nimi_en",
                "description",
                "eap",
                "semester",
                "oppeaste",
                "ois_url",
                "eesmark_et",
                "_match_score_10",
                "_match_confidence",
                "_llm_overview",
                "_llm_goals",
                "_llm_relevance",
                "_llm_score_10",
                "_display_score_10",
            ]
            results_display = results_df[[c for c in display_cols if c in results_df.columns]].copy()

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "debug_info": {
                    "user_prompt": prompt,
                    "filters": sidebar["active_filters_str"],
                    "ranking_mode": sidebar["ranking_mode"],
                    "filtered_count": len(filtered_df),
                    "candidate_count": candidate_count,
                    "results_df": results_display,
                    "system_prompt": system_content,
                },
            })
            st.rerun()
        except Exception as e:
            error_msg = f"Viga API päringul: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ==========================================================================
# Main
# ==========================================================================
def main(developer_mode: bool = False) -> None:
    page_title = "Kursuste nõustaja" if not developer_mode else "Kursuste nõustaja - Arendaja"
    st.set_page_config(page_title=page_title, layout="wide")

    # Inject all custom styles before any content renders
    _inject_global_styles()

    st.markdown(
        "<h1>Kursuste n\u00f5ustaja</h1>",
        unsafe_allow_html=True,
    )
    if developer_mode:
        st.caption("Arendaja vaade: sisaldab benchmarki ja RAG silumisvaateid.")
    else:
        st.caption("Tartu \u00dclikooli kursuste soovitaja.")

    _initialize_session_state()

    # --- Load data ---
    df = None
    embeddings = None
    df_error: Exception | None = None
    emb_error: Exception | None = None

    try:
        df = _load_courses()
    except Exception as e:
        df_error = e

    try:
        embeddings = _load_embeddings()
        if df is not None:
            assert len(embeddings) == len(df), (
                f"Embeddings ({len(embeddings)}) ja andmestiku ({len(df)}) ridade arv ei klapi. "
                "Käivita build_embeddings.py uuesti."
            )
    except FileNotFoundError as e:
        emb_error = e
    except Exception as e:
        emb_error = e

    if df_error:
        st.error(f"Andmete laadimine ebaõnnestus: {df_error}")
    if emb_error:
        st.warning(
            "Embeddingute fail puudub või ei lae. Käivita esmalt: `python build_embeddings.py`"
        )

    data_ready = df is not None and embeddings is not None
    api_key = _resolve_openrouter_api_key()

    # --- Sidebar ---
    opts = _derive_filter_options(df) if df is not None else _FALLBACK_OPTIONS
    sidebar = _render_sidebar(df, opts, api_key=api_key, developer_mode=developer_mode)
    _maybe_auto_release_reranker_memory(
        current_mode=sidebar["ranking_mode"],
        current_local_model=sidebar["local_rerank_model"],
        enabled=sidebar["auto_release_reranker"],
    )

    if developer_mode:
        # --- Benchmark sidebar & actions ---
        benchmark_case_count = get_benchmark_case_count()
        (
            run_clicked,
            load_clicked,
            benchmark_limit,
            benchmark_ranking_mode,
            benchmark_local_model,
        ) = render_benchmark_sidebar(
            sidebar["api_key"], benchmark_case_count,
        )

        if run_clicked and data_ready:
            with st.spinner("Laadin mudeleid..."):
                embedder = _load_embedder()
                reranker = _load_reranker() if benchmark_ranking_mode == "cross_encoder" else None
                local_runtime = (
                    _load_local_llm_reranker(benchmark_local_model)
                    if benchmark_ranking_mode == "local_llm"
                    else None
                )
            run_benchmark(
                sidebar["api_key"],
                embedder,
                reranker,
                local_runtime,
                df,
                embeddings,
                benchmark_limit,
                benchmark_ranking_mode,
            )
        if load_clicked:
            load_saved_benchmark()

        render_benchmark_results(df if df is not None else pd.DataFrame())

    # --- Chat ---
    _render_chat_history(sidebar["show_debug"], developer_mode=developer_mode)

    if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
        if not data_ready:
            st.error("Andmed pole laaditud. Käivita esmalt `python build_embeddings.py`.")
        else:
            _handle_user_prompt(prompt, sidebar, df, embeddings)


if __name__ == "__main__":
    main(developer_mode=False)
