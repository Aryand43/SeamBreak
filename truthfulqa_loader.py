from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


DATASET_PATH = "Data/TruthfulQA.csv"


@dataclass(frozen=True)
class TruthfulQARow:
    question: str
    correct_answers: List[str]
    incorrect_answers: List[str]


def _split_answers(s: str) -> List[str]:
    # Deterministic normalization: strip + lower, split on ';'
    parts = [p.strip().lower() for p in str(s).split(";")]
    return [p for p in parts if p]


def load_truthfulqa_rows(path: str = DATASET_PATH) -> List[TruthfulQARow]:
    """
    Load Data/TruthfulQA.csv and extract:
      - Question
      - Correct Answers
      - Incorrect Answers

    Returns rows with normalized (strip().lower()) question and answers,
    and answers split on ';'.
    """
    df = pd.read_csv(path)
    required = ["Question", "Correct Answers", "Incorrect Answers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    rows: List[TruthfulQARow] = []
    # O(1) deterministic work per row
    for q, ca, ia in zip(
        df["Question"].astype(str), df["Correct Answers"].astype(str), df["Incorrect Answers"].astype(str)
    ):
        rows.append(
            TruthfulQARow(
                question=q.strip().lower(),
                correct_answers=_split_answers(ca),
                incorrect_answers=_split_answers(ia),
            )
        )
    return rows

