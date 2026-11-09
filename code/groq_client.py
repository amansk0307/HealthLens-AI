"""Small HTTP client for Groq's OpenAI-compatible chat completions API."""

import json
import re
from collections.abc import Iterator
from typing import Any

import requests


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT_SECONDS = 60
MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â\u0080", "ï»¿", "�")
CITATION_REPEAT_PATTERN = re.compile(r"(\[S\d+\])(?:\s*\1)+")


def normalize_generated_text(text: str) -> str:
    """Repair common UTF-8 mojibake and collapse adjacent duplicate citations."""
    for encoding in ("latin-1", "cp1252"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if sum(text.count(marker) for marker in MOJIBAKE_MARKERS) > sum(
            repaired.count(marker) for marker in MOJIBAKE_MARKERS
        ):
            text = repaired
            break
    return CITATION_REPEAT_PATTERN.sub(r"\1", text)


class GroqAPIError(RuntimeError):
    """Raised when the Groq API cannot return a usable response."""


def _request_payload(
    messages: list[dict[str, str]], json_mode: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_completion_tokens": 1024,
        "temperature": 0.2,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _post_completion(
    messages: list[dict[str, str]], api_key: str, *, stream: bool, json_mode: bool
) -> requests.Response:
    if not api_key.strip():
        raise GroqAPIError("The GROQ_API_KEY setting is empty.")
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                **_request_payload(messages, json_mode),
                "stream": stream,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=stream,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        response_text = error.response.text[:500] if error.response is not None else ""
        status = error.response.status_code if error.response is not None else "unknown"
        detail = f": {response_text}" if response_text else ""
        raise GroqAPIError(f"Groq API returned HTTP {status}{detail}") from error
    except requests.RequestException as error:
        raise GroqAPIError(f"Could not reach the Groq API: {error}") from error
    return response


def generate_groq_response(
    messages: list[dict[str, str]], api_key: str, *, json_mode: bool = False
) -> str:
    """Return one non-streamed response from Groq."""
    response = _post_completion(
        messages, api_key, stream=False, json_mode=json_mode
    )
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise GroqAPIError("Groq returned an invalid chat completion response.") from error
    if not isinstance(content, str) or not content.strip():
        raise GroqAPIError("Groq returned an empty response.")
    return normalize_generated_text(content.strip())


def stream_groq_response(
    messages: list[dict[str, str]], api_key: str
) -> Iterator[str]:
    """Yield visible text chunks from a streamed Groq chat completion."""
    with _post_completion(messages, api_key, stream=True, json_mode=False) as response:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
                content = payload["choices"][0]["delta"].get("content", "")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise GroqAPIError("Groq returned an invalid streaming response.") from error
            if content:
                yield content
