"""Pluggable LLM backend. Default: local Ollama (no API key). Optional: OpenAI / Anthropic / Gemini via .env."""

import json
import urllib.request

from .config import Config


class LLMError(RuntimeError):
    pass


def _ollama_generate(prompt: str, system: str, config: Config) -> str:
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        f"{config.ollama_host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()
    except Exception as e:  # noqa: BLE001
        raise LLMError(
            f"Ollama call failed ({e}). Is `ollama serve` running and model "
            f"'{config.ollama_model}' pulled? Try: ollama pull {config.ollama_model}"
        ) from e


def _openai_generate(prompt: str, system: str, config: Config) -> str:
    if not config.openai_api_key:
        raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty in .env")
    from openai import OpenAI

    client = OpenAI(api_key=config.openai_api_key)
    resp = client.chat.completions.create(
        model=config.openai_model,
        temperature=0.0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def _anthropic_generate(prompt: str, system: str, config: Config) -> str:
    if not config.anthropic_api_key:
        raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty in .env")
    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    resp = client.messages.create(
        model=config.anthropic_model,
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _gemini_generate(prompt: str, system: str, config: Config) -> str:
    if not config.google_api_key:
        raise LLMError("LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty in .env")
    import time

    import google.generativeai as genai
    from google.api_core import exceptions as gexc

    genai.configure(api_key=config.google_api_key)
    model = genai.GenerativeModel(config.gemini_model, system_instruction=system)

    # Free-tier keys are rate-limited (HTTP 429). Back off politely and retry a few times.
    delays = [0, 20, 40, 60]
    last_err: Exception | None = None
    for wait in delays:
        if wait:
            time.sleep(wait)
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0.0})
            return resp.text.strip()
        except gexc.ResourceExhausted as e:  # 429 quota
            last_err = e
            continue
    raise LLMError(
        f"Gemini rate-limited (429) after retries on model '{config.gemini_model}'. "
        f"Free tier is very limited; wait a minute or switch GEMINI_MODEL to a higher-quota model. "
        f"Underlying: {last_err}"
    )


_PROVIDERS = {
    "ollama": _ollama_generate,
    "openai": _openai_generate,
    "anthropic": _anthropic_generate,
    "gemini": _gemini_generate,
}


def generate(prompt: str, system: str, config: Config) -> str:
    provider = config.llm_provider
    if provider not in _PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](prompt, system, config)
