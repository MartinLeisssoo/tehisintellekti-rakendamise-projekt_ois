import re

import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    semesters: list[str],
    eap_range: tuple[float, float],
    grading_choice: str,
    languages: list[str] | None = None,
    cities: list[str] | None = None,
    study_levels: list[str] | None = None,
    teaching_methods: list[str] | None = None,
    domains: list[str] | None = None,
) -> pd.Series:
    """Return a boolean mask based on sidebar filters."""
    semester_series = df["semester"].astype(str).str.lower()
    semesters_clean = [s.strip().lower() for s in semesters if s.strip()]
    if semesters_clean:
        mask = semester_series.isin(semesters_clean)
    else:
        mask = pd.Series(True, index=df.index)

    eap_series = pd.to_numeric(df["eap"], errors="coerce")
    mask &= eap_series.between(eap_range[0], eap_range[1])

    if grading_choice != "Koik":
        is_eristav = df["hindamisskaala"].str.contains("Eristav", case=False, na=False)
        if grading_choice == "Eristav":
            mask &= is_eristav
        else:
            mask &= ~is_eristav

    if languages:
        lang_mask = pd.Series(False, index=df.index)
        for lang in languages:
            lang_mask |= df["oppekeeled"].str.contains(lang, case=False, na=False)
        mask &= lang_mask

    if cities:
        mask &= df["linn"].astype(str).isin(cities)

    if study_levels:
        level_mask = pd.Series(False, index=df.index)
        for level in study_levels:
            level_mask |= df["oppeaste"].str.contains(re.escape(level), case=False, na=False)
        mask &= level_mask

    if teaching_methods:
        mask &= df["oppeviis"].astype(str).isin(teaching_methods)

    if domains:
        mask &= df["valdkond"].astype(str).isin(domains)

    return mask


def format_active_filters(
    semesters: list[str],
    eap_range: tuple[float, float],
    grading_choice: str,
    languages: list[str] | None = None,
    cities: list[str] | None = None,
    study_levels: list[str] | None = None,
    teaching_methods: list[str] | None = None,
    domains: list[str] | None = None,
) -> str:
    """Build a human-readable summary of active filters."""
    parts = [
        f"semester={', '.join(semesters) or 'koik'}",
        f"eap={eap_range[0]}-{eap_range[1]}",
        f"hindamisskaala={grading_choice}",
    ]
    if languages:
        parts.append(f"keel={', '.join(languages)}")
    if cities:
        parts.append(f"linn={', '.join(cities)}")
    if study_levels:
        parts.append(f"aste={', '.join(study_levels)}")
    if teaching_methods:
        parts.append(f"viis={', '.join(teaching_methods)}")
    if domains:
        parts.append(f"valdkond={', '.join(domains)}")
    return ", ".join(parts)
