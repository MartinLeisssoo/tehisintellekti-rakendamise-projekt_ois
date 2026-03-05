# AGENTS

Guidance for agentic coding assistants working in this repository.

## 1) Scope and intent
- Project domain: University of Tartu course discovery and recommendation.
- Main work type: CSV data understanding, cleaning, and prep for filtering/RAG.
- Codebase state: notebook-heavy, minimal package structure, no CI yet.
- Keep changes small, practical, and easy for students to review.

## 2) Rule files discovered
- Cursor rules in `.cursor/rules/`: not found.
- `.cursorrules`: not found.
- Copilot instructions in `.github/copilot-instructions.md`: not found.
- If any of these files are added later, treat them as higher-priority guidance.

## 3) Current repository snapshot
- Key docs: `projektiplaan.md`, `README.md`, `Most_selected_features_andcomments.csv`.
- Key notebooks: `andmetega_tutvumine.ipynb`, `andmete_puhastamine.ipynb`, `andmete_puhastamine_naide.ipynb`.
- Data directory: `toorandmed/`.
- Current data files:
  - `toorandmed/toorandmed_aasta.csv` (raw input)
  - `toorandmed/puhastatud_andmed.csv` (cleaned output)
  - `toorandmed/ids.csv`
- Example app: `hello_ai.py` (simple Streamlit sanity app).

## 4) Environment and dependencies
- Environment file: `environment.yml`.
- Python target: 3.10 (conda).
- Core libs: pandas, numpy, scikit-learn, requests.
- AI-related libs are present (OpenAI/Ollama/Groq/LangChain) but optional by task.

## 5) Build, run, lint, and test commands

### Environment setup
- Create env: `conda env create -f environment.yml`
- Activate env: `conda activate oisi_projekt`
- Update env: `conda env update -f environment.yml --prune`

### Run workflows
- Run Streamlit example: `streamlit run hello_ai.py`
- Execute notebook in place:
  - `jupyter nbconvert --to notebook --execute --inplace andmete_puhastamine.ipynb`
  - `jupyter nbconvert --to notebook --execute --inplace andmetega_tutvumine.ipynb`

### Build status
- There is no formal build pipeline.
- Treat notebook/script execution as the main validation step.

### Lint/format status
- No mandatory lint config (no `pyproject.toml`, `ruff`, `flake8`, `black` config detected).
- If you add a formatter, prefer Black defaults and document it.

### Tests (important)
- No tests currently exist in the repo.
- If tests are added, use pytest.
- Run all tests: `pytest`
- Run one test file: `pytest tests/test_some_module.py`
- Run a single test by node id: `pytest tests/test_some_module.py::test_specific_case`
- Run a single test by keyword: `pytest tests/test_some_module.py -k "specific_case"`

## 6) Coding style guidelines

### Imports
- Group imports as: standard library, third-party, local.
- Keep imports at the top of modules.
- Avoid wildcard imports.
- Prefer `import pandas as pd` and `import numpy as np`.

### Formatting
- Use 4-space indentation.
- Prefer line length around 88-100 chars.
- Prefer double quotes for user-facing strings.
- Keep blank lines between top-level functions/classes.
- Do not perform broad reformat-only changes.

### Naming
- Variables/functions: `snake_case`.
- Classes: `CapWords`.
- Constants: `UPPER_SNAKE_CASE`.
- Data columns and derived features should have explicit, descriptive names.

### Types
- The codebase is mostly untyped today.
- Add type hints for new non-trivial functions.
- Prefer built-in generics (`list[str]`, `dict[str, Any]`) on Python 3.10+.

### Functions and structure
- Keep functions small and single-purpose.
- Separate pure transforms from IO when possible.
- Reuse helper functions for repeated JSON parsing/normalization patterns.

### Error handling
- Validate required input files and required columns early.
- Fail fast with clear error messages when schema assumptions break.
- Do not silently swallow exceptions.
- For external requests, include timeout/retry behavior.

### Data quality and preprocessing
- Expect many missing values; handle `NaN` explicitly.
- Normalize whitespace and text artifacts before vectorization.
- When fields exist in both base and `version__*`, prefer `version__*` and fallback to base.
- For JSON-like string fields, parse safely and return consistent output types.
- Preserve row counts unless filtering is explicitly requested.

### Reproducibility
- Keep transformations deterministic.
- Set random seeds for any stochastic operations.
- Avoid notebook state bugs; execute notebooks top-to-bottom before finalizing.

### Notebook practices
- Keep notebook outputs meaningful but not excessively noisy.
- Add short markdown context before major transformation blocks.
- Prefer moving reusable logic to helper functions instead of large monolithic cells.

## 7) File organization conventions
- Use root for top-level entry points and key notebooks only.
- If code grows, create a `src/` package and migrate logic there.
- Keep raw/processed data in `toorandmed/` unless a new data dir is justified.
- Do not rename or reorder CSV columns without a clear reason.

## 8) Security and privacy
- Never commit secrets or API keys.
- Use `.env` for local secrets when needed; ensure `.env` is gitignored.
- Course metadata is generally public, but avoid logging personal user input.
- Be cautious about prompt injection if building LLM-facing pipelines.

## 9) Agent workflow expectations
- Read relevant docs/data schema before editing.
- Respect existing conventions in notebooks/scripts.
- Make minimal, targeted changes.
- Explain assumptions clearly when schema or business logic is ambiguous.
- Do not introduce major tools/frameworks without a brief rationale.

## 10) If you add tests
- Prefer layout: `tests/test_*.py`.
- Use fixtures for repeated sample data.
- Keep tests fast, deterministic, and isolated from network calls.
- Include at least one schema validation test for preprocessing outputs.

## 11) Definition of done for most tasks
- Code/notebook runs successfully in the project conda env.
- Output artifacts (if any) are written to expected paths.
- Relevant validation commands were run.
- README/usage notes updated when workflow changes.
- AGENTS.md stays aligned with the repository reality.
