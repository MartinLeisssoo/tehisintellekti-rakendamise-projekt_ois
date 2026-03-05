# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University of Tartu course discovery app — semantic/RAG-based search over OIS2 course data, built for a university AI course.

## Environment

```bash
conda env create -f environment.yml   # first time
conda activate oisi_projekt
```

## Key commands

```bash
# Run the app
streamlit run app.py

# Rebuild embeddings (required after CSV changes)
python build_embeddings.py

# Execute a notebook in place
jupyter nbconvert --to notebook --execute --inplace andmete_puhastamine.ipynb
```

No lint config, no test suite (pytest if added: `pytest`).

## Architecture

The app is a single-file Streamlit chat UI (`app.py`) that implements a RAG pipeline:

1. **Pre-filter** (`apply_filters`) — pandas boolean mask on `andmed/puhastatud_andmed.csv` by semester, EAP range, and grading scale before any vector work.
2. **Semantic search** — query encoded with `BAAI/bge-m3` (SentenceTransformer, loaded via `@st.cache_resource`), cosine similarity against pre-computed embeddings in `andmed/embeddings.pkl`, top-20 candidates selected.
3. **Language detection** (`detect_language`) — stopword heuristic to pick ET or EN system prompt.
4. **LLM call** — OpenRouter API (OpenAI-compatible client), model `google/gemma-3-27b-it`, streamed response. API key entered by the user in the sidebar at runtime.

`build_embeddings.py` must be re-run whenever `puhastatud_andmed.csv` changes; it encodes the `description` column and writes `andmed/embeddings.pkl`. Row count must match between the CSV and pkl file — `app.py` asserts this on load.

## Data

- `andmed/toorandmed_aasta.csv` — raw OIS2 export (~3031 rows, 223 columns)
- `andmed/puhastatud_andmed.csv` — cleaned output used by the app
- `andmed/embeddings.pkl` — numpy array of embeddings, one row per course
- Notebooks handle EDA (`andmetega_tutvumine.ipynb`) and cleaning (`andmete_puhastamine.ipynb`)

## Important conventions

- When `version__*` and base fields both exist in the CSV, prefer `version__*` and fall back to base.
- The `description` column is the single concatenated text field used for embedding — it must exist in the cleaned CSV.
- Never commit `.env` or API keys; use `.env` for local secrets (already gitignored).
