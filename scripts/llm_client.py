"""Small HTTP client for a locally running Ollama chat model."""

from __future__ import annotations

import os

import requests


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.2},
            },
            timeout=300,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
            ) from error
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
