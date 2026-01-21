from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from openai import OpenAI


LANGDB_BASE_URL = "https://api.us-east-1.langdb.ai"


class NormalizedOutput(TypedDict):
    model: str
    text: str
    raw: object
    latency: float


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Minimal .env loader (no external deps).
    Loads KEY=VALUE pairs into process env if key is not already set.
    """
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Set it in .env or your shell environment."
        )
    return value


def _validate_messages(messages: List[Mapping[str, Any]]) -> None:
    if not isinstance(messages, list) or len(messages) == 0:
        raise TypeError("messages must be a non-empty list of {role, content} dicts.")

    for i, m in enumerate(messages):
        if not isinstance(m, Mapping):
            raise TypeError(f"messages[{i}] must be a mapping/dict.")
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant", "developer"):
            raise ValueError(
                f"messages[{i}].role must be one of system/user/assistant/developer."
            )
        # Text-only prompts: require content be a plain string.
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}].content must be a string (text-only).")


@dataclass(frozen=True)
class LangDBClient:
    """
    Minimal, scalable LangDB pipeline wrapper.

    Invariant interface:
      - bind a model at construction time
      - call only accepts `messages` (no model-specific params)
      - no provider-specific branching; model is an opaque string route
    """

    model: str
    api_key_env: str = "LANGDB_API_KEY"
    project_id_env: str = "LANGDB_PROJECT_ID"
    base_url: str = LANGDB_BASE_URL
    dotenv_path: str = ".env"

    def __post_init__(self) -> None:
        _load_dotenv(self.dotenv_path)

        api_key = _require_env(self.api_key_env)
        _require_env(self.project_id_env)

        # store client on the instance (dataclass is frozen -> use object.__setattr__)
        client = OpenAI(base_url=self.base_url, api_key=api_key)
        object.__setattr__(self, "_client", client)

    def chat(self, messages: List[Mapping[str, Any]]) -> NormalizedOutput:
        """
        Execute a single chat completion using the invariant request shape.
        Input: messages only.
        Output: { model, text, raw, latency }
        """
        _validate_messages(messages)

        project_id = _require_env(self.project_id_env)

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[dict(m) for m in messages],
            extra_headers={"x-project-id": project_id},
        )
        latency = time.perf_counter() - t0

        text = ""
        try:
            choice0 = resp.choices[0]
            msg = getattr(choice0, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            text = content if isinstance(content, str) else ""
        except Exception:
            text = ""

        raw: object
        if hasattr(resp, "model_dump"):
            raw = resp.model_dump()
        elif hasattr(resp, "to_dict"):
            raw = resp.to_dict()
        else:
            raw = resp

        return NormalizedOutput(
            model=str(getattr(resp, "model", self.model)),
            text=text,
            raw=raw,
            latency=float(latency),
        )

