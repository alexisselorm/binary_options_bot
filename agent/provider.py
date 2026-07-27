from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from utils.config import Config


def get_llm(cfg: Config) -> ChatOpenAI:
    provider = getattr(cfg, "agent_provider", "openai").lower()
    model = getattr(cfg, "agent_model", "gpt-4o-mini")
    temperature = float(getattr(cfg, "agent_temperature", 0.0))

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=getattr(cfg, "agent_openai_api_key", "") or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
        )

    if provider == "deepseek":
        return ChatOpenAI(
            model=model,
            api_key=getattr(cfg, "agent_deepseek_api_key", "") or os.getenv("DEEPSEEK_API_KEY"),
            base_url=getattr(cfg, "agent_deepseek_base_url", "https://api.deepseek.com/v1"),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported agent provider: {provider}")

