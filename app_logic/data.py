import pickle

import numpy as np
import pandas as pd

from app_logic.config import DATA_PATH, EMBEDDINGS_PATH


def load_courses(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned course catalogue CSV and add OIS URL column."""
    df = pd.read_csv(path)
    df["ois_url"] = df["aine_kood"].apply(
        lambda code: f"https://ois2.ut.ee/#/courses/{code}/details"
        if pd.notna(code) and str(code).strip()
        else ""
    )
    return df


def load_embeddings(path: str = EMBEDDINGS_PATH) -> np.ndarray:
    """Load pre-computed course embeddings from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
