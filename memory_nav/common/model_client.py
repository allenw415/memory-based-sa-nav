from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SUPPORTED_API_KINDS = {"responses", "chat_completions"}
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 5
HTTP_RETRY_BACKOFF_SECONDS = 1.0


def resolve_api_kind(api_kind: str | None = None) -> str:
    value = (api_kind or os.environ.get("ST_NAV_API_KIND") or "responses").strip().lower()
    if value not in SUPPORTED_API_KINDS:
        supported = ", ".join(sorted(SUPPORTED_API_KINDS))
        raise ValueError(f"Unsupported API kind: {value}. Expected one of: {supported}.")
    return value


def extract_output_text(payload: dict) -> str | None:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    candidate_text = _extract_gemini_candidate_text(payload)
    if isinstance(candidate_text, str) and candidate_text.strip():
        return candidate_text

    fragments: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                fragments.append(text)
    if fragments:
        return "".join(fragments)

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
    return None


def parse_json_output(payload: dict) -> dict:
    output_text = extract_output_text(payload)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Model API payload did not include output text.")
    return parse_json_text(output_text)


def parse_json_text(text: str) -> dict:
    stripped = text.strip()
    candidates = [stripped]

    fenced = _extract_fenced_code_block(stripped)
    if fenced and fenced not in candidates:
        candidates.append(fenced)

    bracketed = _extract_first_balanced_json(stripped)
    if bracketed and bracketed not in candidates:
        candidates.append(bracketed)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = stripped[:400].replace("\n", "\\n")
    raise ValueError(f"Model returned non-JSON output: {preview}")


def _extract_fenced_code_block(text: str) -> str | None:
    marker = "```"
    start = text.find(marker)
    if start < 0:
        return None
    end = text.find(marker, start + len(marker))
    if end < 0:
        return None
    body = text[start + len(marker):end].strip()
    if "\n" in body:
        first_line, rest = body.split("\n", 1)
        if first_line.strip().lower() in {"json", "javascript", "js"}:
            return rest.strip()
    return body.strip()


def _extract_first_balanced_json(text: str) -> str | None:
    openings = {"{": "}", "[": "]"}
    for index, char in enumerate(text):
        if char not in openings:
            continue
        closing = openings[char]
        depth = 0
        in_string = False
        escaped = False
        for end_index in range(index, len(text)):
            current = text[end_index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == char:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    return text[index : end_index + 1].strip()
    return None


class ModelResponseClient:
    """
    Shared transport wrapper for hosted closed-source APIs and self-hosted
    OpenAI-compatible servers.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        api_base: str = DEFAULT_OPENAI_API_BASE,
        api_kind: str | None = None,
        request_timeout: float = 30.0,
        num_ctx: int | None = None,
        temperature: float | None = None,
        response_client: Callable[[dict], dict] | None = None,
        max_http_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ):
        self.provider = (provider or "").strip().lower() or None
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.api_kind = resolve_api_kind(api_kind)
        self.request_timeout = request_timeout
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.response_client = response_client
        self.max_http_retries = max(1, int(max_http_retries or _parse_int_env("ST_NAV_MAX_HTTP_RETRIES", MAX_HTTP_RETRIES)))
        self.retry_backoff_seconds = max(
            0.0,
            float(retry_backoff_seconds if retry_backoff_seconds is not None else _parse_float_env(
                "ST_NAV_HTTP_RETRY_BACKOFF_SECONDS",
                HTTP_RETRY_BACKOFF_SECONDS,
            )),
        )

    def is_configured(self) -> bool:
        return self.response_client is not None or bool(self.api_key) or self.api_base != DEFAULT_OPENAI_API_BASE

    def create(self, request_body: dict) -> dict:
        if self.response_client is not None:
            return self._normalize_payload(self.response_client(request_body))

        if self.provider in {"gemini", "gemini_api", "google_gemma_api"}:
            if not self.api_key:
                raise RuntimeError("Missing GEMINI_API_KEY, GOOGLE_API_KEY, or ST_NAV_API_KEY for Gemini API provider.")
            endpoint = self._gemini_endpoint(request_body)
            payload = self._responses_to_gemini_generate_content_payload(request_body)
        elif self.provider == "ollama":
            endpoint = f"{self._ollama_api_base()}/api/chat"
            payload = self._responses_to_ollama_chat_payload(request_body)
        elif self.api_kind == "responses":
            endpoint = f"{self.api_base}/responses"
            payload = request_body
        else:
            endpoint = f"{self.api_base}/chat/completions"
            payload = self._responses_to_chat_completions_payload(request_body)

        headers = {"Content-Type": "application/json"}
        if self.provider in {"gemini", "gemini_api", "google_gemma_api"}:
            headers["x-goog-api-key"] = self.api_key
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(1, self.max_http_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    return self._normalize_payload(json.loads(response.read().decode("utf-8")))
            except TimeoutError as exc:
                if attempt < self.max_http_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise TimeoutError(
                    f"Model API timed out after {self.request_timeout:.0f}s while waiting for {self.api_kind} output."
                ) from exc
            except socket.timeout as exc:
                if attempt < self.max_http_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise TimeoutError(
                    f"Model API timed out after {self.request_timeout:.0f}s while waiting for {self.api_kind} output."
                ) from exc
            except urllib.error.HTTPError as exc:
                body = _read_http_error_body(exc)
                if exc.code in TRANSIENT_HTTP_STATUS_CODES and attempt < self.max_http_retries:
                    self._sleep_before_retry(attempt)
                    continue
                detail = _format_http_error_detail(body)
                raise RuntimeError(
                    f"Model API request failed with HTTP {exc.code} from {endpoint}: {exc.reason}.{detail}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_http_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"Model API request failed from {endpoint}: {exc.reason}") from exc

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.retry_backoff_seconds * float(attempt))

    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        normalized = json.loads(json.dumps(payload))
        output_text = extract_output_text(normalized)
        if isinstance(output_text, str) and output_text.strip():
            normalized["output_text"] = output_text
        return normalized

    def _ollama_api_base(self) -> str:
        if self.api_base.endswith("/v1"):
            return self.api_base[:-3]
        return self.api_base

    def _gemini_endpoint(self, request_body: dict) -> str:
        api_base = self.api_base or DEFAULT_GEMINI_API_BASE
        encoded_model = urllib.parse.quote(str(request_body.get("model")), safe="")
        return f"{api_base.rstrip('/')}/models/{encoded_model}:generateContent"

    @staticmethod
    def _iter_input_messages(input_value: object) -> list[dict]:
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]
        if isinstance(input_value, dict):
            return [input_value]
        if not isinstance(input_value, list):
            return []

        messages: list[dict] = []
        for item in input_value:
            if isinstance(item, dict):
                messages.append(item)
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
        return messages

    @staticmethod
    def _responses_to_chat_completions_payload(request_body: dict) -> dict:
        messages: list[dict] = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        for item in ModelResponseClient._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = "user"
            messages.append(
                {
                    "role": role,
                    "content": ModelResponseClient._convert_content_blocks(item.get("content")),
                }
            )

        payload = {
            "model": request_body.get("model"),
            "messages": messages,
        }

        response_format = ModelResponseClient._response_format_from_request_body(request_body)
        if response_format is not None:
            payload["response_format"] = response_format

        return payload

    def _responses_to_ollama_chat_payload(self, request_body: dict) -> dict:
        messages: list[dict] = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        for item in self._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = "user"
            converted = self._convert_content_blocks_for_ollama(item.get("content"))
            messages.append(
                {
                    "role": role,
                    "content": converted["content"],
                    **({"images": converted["images"]} if converted["images"] else {}),
                }
            )

        payload: dict[str, object] = {
            "model": request_body.get("model"),
            "messages": messages,
            "stream": False,
        }

        ollama_format = self._ollama_format_from_request_body(request_body)
        if ollama_format is not None:
            payload["format"] = ollama_format

        options: dict[str, object] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        elif ollama_format is not None:
            options["temperature"] = 0
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        elif ollama_format is not None:
            options["num_ctx"] = 4096
        if options:
            payload["options"] = options
        return payload

    def _responses_to_gemini_generate_content_payload(self, request_body: dict) -> dict:
        contents: list[dict] = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": instructions}],
                }
            )

        for item in self._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if role == "assistant":
                mapped_role = "model"
            else:
                mapped_role = "user"
            parts = self._convert_content_blocks_for_gemini(item.get("content"))
            contents.append({"role": mapped_role, "parts": parts})

        payload: dict[str, object] = {"contents": contents}

        generation_config: dict[str, object] = {}
        gemini_schema = self._gemini_response_schema_from_request_body(request_body)
        if gemini_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseJsonSchema"] = gemini_schema
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        elif gemini_schema is not None:
            generation_config["temperature"] = 0
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    @staticmethod
    def _convert_content_blocks(content: object) -> object:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content

        converted: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str):
                    converted.append({"type": "text", "text": text})
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    image_payload = {"url": image_url}
                    detail = block.get("detail")
                    if isinstance(detail, str) and detail:
                        image_payload["detail"] = detail
                    converted.append({"type": "image_url", "image_url": image_payload})
        return converted

    @staticmethod
    def _convert_content_blocks_for_ollama(content: object) -> dict[str, object]:
        if isinstance(content, str):
            return {"content": content, "images": []}
        if not isinstance(content, list):
            return {"content": "", "images": []}

        text_parts: list[str] = []
        images: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    images.append(ModelResponseClient._data_url_payload(image_url))
        return {"content": "\n".join(text_parts), "images": images}

    @staticmethod
    def _convert_content_blocks_for_gemini(content: object) -> list[dict]:
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return []

        parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append({"text": text})
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    mime_type, data = ModelResponseClient._decode_data_url(image_url)
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": data,
                            }
                        }
                    )
        return parts

    @staticmethod
    def _response_format_from_request_body(request_body: dict) -> dict | None:
        text_config = request_body.get("text")
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict) or format_config.get("type") != "json_schema":
            return None

        name = format_config.get("name")
        schema = format_config.get("schema")
        strict = format_config.get("strict")
        if not isinstance(name, str) or not isinstance(schema, dict):
            return None

        json_schema: dict[str, object] = {"name": name, "schema": schema}
        if isinstance(strict, bool):
            json_schema["strict"] = strict
        return {"type": "json_schema", "json_schema": json_schema}

    @staticmethod
    def _ollama_format_from_request_body(request_body: dict) -> dict | str | None:
        text_config = request_body.get("text")
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict):
            return None
        if format_config.get("type") != "json_schema":
            return None
        schema = format_config.get("schema")
        if isinstance(schema, dict):
            return schema
        return "json"

    @staticmethod
    def _gemini_response_schema_from_request_body(request_body: dict) -> dict | None:
        text_config = request_body.get("text")
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict) or format_config.get("type") != "json_schema":
            return None
        schema = format_config.get("schema")
        if isinstance(schema, dict):
            return schema
        return None

    @staticmethod
    def _data_url_payload(image_url: str) -> str:
        marker = ";base64,"
        if marker in image_url:
            return image_url.split(marker, 1)[1]
        return image_url

    @staticmethod
    def _decode_data_url(image_url: str) -> tuple[str, str]:
        prefix = "data:"
        marker = ";base64,"
        if image_url.startswith(prefix) and marker in image_url:
            header, data = image_url[len(prefix):].split(marker, 1)
            return header, data
        return "image/png", image_url


def _extract_gemini_candidate_text(payload: dict) -> str | None:
    fragments: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                fragments.append(text)
    if fragments:
        return "".join(fragments)
    return None


def _read_http_error_body(exc: urllib.error.HTTPError) -> str | None:
    try:
        payload = exc.read()
    except Exception:
        return None
    if not payload:
        return None
    return payload.decode("utf-8", errors="replace")


def _format_http_error_detail(body: str | None) -> str:
    if not body:
        return ""
    compact = " ".join(body.strip().split())
    if len(compact) > 300:
        compact = compact[:300].rstrip() + "..."
    return f" Response body: {compact}"


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default
