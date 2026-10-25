"""Small OpenAI-compatible client for explicitly selected local models."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx


DEFAULT_BASE_URL = "http://localhost:8080/v1"
TIMEOUT_SECONDS = 120.0


def normalize_model_id(value: str) -> str:
    """Normalize punctuation differences in model IDs."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def model_matches(requested: str, returned: Any) -> bool:
    """Accept the requested model ID or that exact ID with a date suffix."""
    if not isinstance(returned, str) or not requested:
        return False
    requested_normalized = normalize_model_id(requested)
    returned_normalized = normalize_model_id(returned)
    if returned_normalized == requested_normalized:
        return True
    if not returned_normalized.startswith(requested_normalized):
        return False
    suffix = returned_normalized[len(requested_normalized) :]
    if not re.fullmatch(r"20\d{6}", suffix):
        return False
    try:
        dt.datetime.strptime(suffix, "%Y%m%d")
    except ValueError:
        return False
    return True


def validate_local_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "http":
        raise ValueError("OPENAI_BASE_URL must use plain HTTP for the local API.")
    if parsed.hostname is None or parsed.hostname.lower() not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("OPENAI_BASE_URL must point to localhost, 127.0.0.1, or ::1.")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("OPENAI_BASE_URL must not include credentials, a query, or a fragment.")
    if parsed.port is None:
        raise ValueError("OPENAI_BASE_URL must specify a port.")
    return value.rstrip("/")


def validate_environment() -> tuple[str, str]:
    """Return the validated local API base URL and the current process key."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the current process environment.")
    base_url = validate_local_base_url(os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    return base_url, api_key


def _redact_text(value: str, api_key: str) -> str:
    if api_key:
        value = value.replace(api_key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{20,}\b", "[REDACTED]", value)


def _redact_value(value: Any, api_key: str) -> Any:
    if isinstance(value, str):
        return _redact_text(value, api_key)
    if isinstance(value, list):
        return [_redact_value(item, api_key) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item, api_key) for key, item in value.items()}
    return value


def _error_message(response: httpx.Response, api_key: str) -> str:
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        payload = response.text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        else:
            message = json.dumps(payload, ensure_ascii=False)
    elif isinstance(payload, str):
        message = payload
    else:
        message = json.dumps(payload, ensure_ascii=False)
    return _redact_text(message, api_key)[:1200]


def _error_result(
    *,
    requested_model: str,
    params: dict[str, Any],
    error_type: str,
    error: str,
    elapsed_seconds: float | None = None,
    http_status: int | None = None,
    returned_model: str | None = None,
    response_id: str | None = None,
    content: Any = None,
    usage: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    api_key: str = "",
) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": error_type,
        "error": _redact_text(error, api_key)[:1200],
        "http_status": http_status,
        "model_requested": requested_model,
        "model_returned": returned_model,
        "response_id": response_id,
        "content": _redact_value(content, api_key),
        "finish_reason": finish_reason,
        "usage": _redact_value(usage, api_key) if usage is not None else None,
        "params": params,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
    }


def request(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 2048,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    """Issue one Chat Completions request without retries or model fallback."""
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer.")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")

    base_url, api_key = validate_environment()
    params = {"max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort}
    body = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    started = time.monotonic()
    try:
        with httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(TIMEOUT_SECONDS),
            trust_env=False,
        ) as client:
            response = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - started
        return _error_result(
            requested_model=model,
            params=params,
            error_type="timeout",
            error=str(exc) or "Local API request timed out.",
            elapsed_seconds=elapsed,
            api_key=api_key,
        )
    except httpx.HTTPError as exc:
        elapsed = time.monotonic() - started
        return _error_result(
            requested_model=model,
            params=params,
            error_type="transport_error",
            error=str(exc) or "Local API request failed.",
            elapsed_seconds=elapsed,
            api_key=api_key,
        )

    elapsed = time.monotonic() - started
    if response.status_code < 200 or response.status_code >= 300:
        return _error_result(
            requested_model=model,
            params=params,
            error_type="api_error",
            error=_error_message(response, api_key),
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            api_key=api_key,
        )
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        return _error_result(
            requested_model=model,
            params=params,
            error_type="invalid_json",
            error=f"The API returned invalid JSON: {exc}",
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            api_key=api_key,
        )
    if not isinstance(payload, dict):
        return _error_result(
            requested_model=model,
            params=params,
            error_type="invalid_response",
            error="The API response was not a JSON object.",
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            api_key=api_key,
        )

    returned_model = payload.get("model") if isinstance(payload.get("model"), str) else None
    response_id = payload.get("id") if isinstance(payload.get("id"), str) else None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return _error_result(
            requested_model=model,
            params=params,
            error_type="invalid_response",
            error="The API response did not contain a standard chat-completion choice.",
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            returned_model=returned_model,
            response_id=response_id,
            usage=usage,
            api_key=api_key,
        )

    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = _redact_value(message.get("content"), api_key)
    finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None

    if not model_matches(model, returned_model):
        return _error_result(
            requested_model=model,
            params=params,
            error_type="model_mismatch",
            error=f"Requested {model}, but the API returned a different model ID.",
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            returned_model=returned_model,
            response_id=response_id,
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            api_key=api_key,
        )

    common = {
        "model_requested": model,
        "model_returned": returned_model,
        "response_id": response_id,
        "content": content,
        "finish_reason": finish_reason,
        "usage": _redact_value(usage, api_key) if usage is not None else None,
        "params": params,
        "elapsed_seconds": round(elapsed, 3),
        "http_status": response.status_code,
    }
    if not isinstance(content, str):
        return {
            **common,
            "status": "error",
            "error_type": "non_text_content",
            "error": "The API returned no text message content.",
        }
    if finish_reason in {"length", "max_tokens"}:
        return {**common, "status": "truncated", "error_type": None, "error": None}
    if finish_reason == "stop":
        return {**common, "status": "complete", "error_type": None, "error": None}
    error_type = "content_filter" if finish_reason == "content_filter" else "unexpected_finish"
    return {
        **common,
        "status": "error",
        "error_type": error_type,
        "error": f"The API finished with reason {finish_reason!r}.",
    }
