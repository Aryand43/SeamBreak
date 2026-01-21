from __future__ import annotations

from langdb_client import LangDBClient
from models_config import MODELS


def main() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]

    for model in MODELS:
        client = LangDBClient(model=model)
        out = client.chat(messages)
        print("=" * 80)
        print(f"model:   {out['model']}")
        print(f"latency: {out['latency']:.3f}s")
        print(f"text:    {out['text']}")


if __name__ == "__main__":
    main()

