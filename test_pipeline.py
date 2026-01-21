from __future__ import annotations

import sys

from langdb_client import LangDBClient
from models_config import MODELS
from truthfulqa_loader import load_truthfulqa_rows
from attacks.self_contradiction import apply_self_contradiction_attack


def main() -> None:
    # Windows consoles can default to cp1252; force UTF-8 to avoid UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = load_truthfulqa_rows()

    for model in MODELS:
        client = LangDBClient(model=model)
        print("=" * 80)
        print(f"MODEL: {model}")

        for i, row in enumerate(rows):
            baseline_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row.question},
            ]
            attacked_messages = apply_self_contradiction_attack(baseline_messages, row)

            baseline_out = client.chat(baseline_messages)
            attacked_out = client.chat(attacked_messages)

            print("-" * 80)
            print(f"ROW {i}")
            print("BASELINE_TEXT:")
            print(baseline_out["text"])
            print("ATTACKED_TEXT:")
            print(attacked_out["text"])


if __name__ == "__main__":
    main()

