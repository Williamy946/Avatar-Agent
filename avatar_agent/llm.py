from __future__ import annotations

import json
import os
import re
from typing import Any


class OpenAICompatibleJSON:
    """Lazy JSON adapter for Qwen3-Omni or another compatible endpoint."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("AVATAR_AGENT_API_KEY", "")
        self.base_url = base_url or os.environ.get("AVATAR_AGENT_BASE_URL", "")
        self.model = model or os.environ.get("AVATAR_AGENT_MODEL", "qwen3-omni-flash")
        if not self.api_key or not self.base_url:
            raise RuntimeError("set AVATAR_AGENT_API_KEY and AVATAR_AGENT_BASE_URL before using --llm")

    def request(self, system: str, user: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install openai to use the LLM adapter") from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}|\[.*\]", content, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON")
        return json.loads(match.group(0))
