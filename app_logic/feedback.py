import csv
import os

from app_logic.config import FEEDBACK_LOG_PATH


def log_feedback(
    timestamp: str,
    prompt: str,
    filters: str,
    context_ids: list[str],
    context_names: list[str],
    response: str,
    rating: str,
    error_category: str,
    comment: str = "",
    path: str = FEEDBACK_LOG_PATH,
) -> None:
    """Append a feedback row to the CSV log."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Aeg", "Kasutaja päring", "Filtrid", "Leitud aine_koodid",
                "Leitud nimed", "LLM Vastus", "Hinnang", "Veatüüp", "Kommentaar",
            ])
        writer.writerow([
            timestamp, prompt, filters, str(context_ids),
            str(context_names), response, rating, error_category, comment,
        ])
