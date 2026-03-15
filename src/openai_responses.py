# -*- coding: utf-8 -*-
"""Helpers for OpenAI Responses API calls and conversions."""

from __future__ import annotations

import json
import logging
import random
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from openai import OpenAI

from src.config import Config, extra_litellm_params, get_api_keys_for_model

logger = logging.getLogger(__name__)


def should_use_openai_responses(model: str, config: Config) -> bool:
    """Return True when OpenAI Responses API should be used for this model."""
    if not getattr(config, "openai_use_responses", True):
        return False
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    return provider == "openai"


def should_use_responses_compat(config: Optional[Config], api_base: Optional[str]) -> bool:
    """Prefer a compatibility payload for non-OpenAI base URLs."""
    env_flag = os.getenv("OPENAI_RESPONSES_COMPAT", "").strip().lower()
    if env_flag in {"0", "false", "no", "off"}:
        return False
    if env_flag in {"1", "true", "yes", "on"}:
        return True
    # Auto: enable for non-official base URLs
    return not _is_official_openai_base_url(api_base)


def normalize_openai_model(model: str) -> str:
    """Strip provider prefix for OpenAI-compatible endpoints."""
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def convert_tools_for_responses(tools: List[dict], *, strict: bool = True) -> List[dict]:
    """Convert Chat Completions tool schema into Responses tool schema."""
    converted: List[dict] = []
    for tool in tools or []:
        if tool.get("type") != "function":
            converted.append(tool)
            continue
        if "function" in tool:
            func = tool.get("function") or {}
            payload = {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters") or {},
            }
            if strict:
                payload["strict"] = True
            converted.append(payload)
            continue
        # Already in Responses format (or close to it)
        tool_copy = dict(tool)
        if strict:
            tool_copy.setdefault("strict", True)
        converted.append(tool_copy)
    return converted


def convert_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal message history to Responses API input items."""
    items: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": str(msg.get("content") or ""),
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": _convert_content_to_input_parts(content),
                    }
                )
            for tc in msg.get("tool_calls") or []:
                args = tc.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {}, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": tc.get("name") or "",
                        "arguments": args,
                    }
                )
            continue

        items.append(
            {
                "type": "message",
                "role": role,
                "content": _convert_content_to_input_parts(msg.get("content")),
            }
        )
    return items


def build_responses_request(
    messages: List[Dict[str, Any]],
    *,
    compat_mode: bool = False,
) -> Tuple[Optional[str], Any]:
    """Return (instructions, input) for Responses API from internal messages."""
    if compat_mode and _can_use_simple_messages(messages):
        input_items: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                # Fall back to standard for tool items
                input_items = []
                break
            content = _content_to_text(msg.get("content"))
            input_items.append({"role": role, "content": content})
        if input_items:
            return None, input_items

    instructions = _extract_instructions(messages)
    filtered = [m for m in messages if m.get("role") not in {"system", "developer"}]
    input_items = convert_messages_to_responses_input(filtered)
    return instructions, _simplify_input_if_possible(input_items)


def call_openai_responses(
    *,
    model: str,
    input_items: Any,
    config: Optional[Config] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    tools: Optional[List[dict]] = None,
    instructions: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Call OpenAI Responses API with optional explicit credentials."""
    api_key, api_base, extra_headers = _resolve_openai_credentials(
        model=model,
        config=config,
        api_key=api_key,
        api_base=api_base,
        extra_headers=extra_headers,
    )

    client = OpenAI(
        api_key=api_key,
        base_url=api_base or None,
        default_headers=extra_headers or None,
    )

    request: Dict[str, Any] = {
        "model": normalize_openai_model(model),
        "input": input_items,
    }
    if instructions:
        request["instructions"] = instructions
    if temperature is not None:
        request["temperature"] = temperature
    if max_output_tokens is not None:
        request["max_output_tokens"] = int(max_output_tokens)
    if timeout is not None:
        request["timeout"] = float(timeout)
    if tools:
        strict_tools = should_use_responses_strict(config, api_base)
        request["tools"] = convert_tools_for_responses(tools, strict=strict_tools)

    return client.responses.create(**request)


def extract_output_text(response: Any) -> str:
    """Extract text from a Responses API response."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output_items = _get_attr(response, "output") or []
    texts: List[str] = []
    for item in output_items:
        item_type = _get_attr(item, "type")
        if item_type in {"message", "output_message", "assistant_message"}:
            content = _get_attr(item, "content")
            texts.extend(_extract_text_from_content(content))
        elif item_type in {"output_text"}:
            text_val = _get_attr(item, "text") or _get_attr(item, "output_text")
            if text_val:
                texts.append(str(text_val))
    return "\n".join([t for t in texts if t]).strip()


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from a Responses API response."""
    output_items = _get_attr(response, "output") or []
    tool_calls: List[Dict[str, Any]] = []
    for item in output_items:
        item_type = _get_attr(item, "type")
        if item_type not in {"function_call", "tool_call", "custom_tool_call"}:
            continue
        name = _get_attr(item, "name") or ""
        call_id = _get_attr(item, "call_id") or _get_attr(item, "id") or ""
        args_raw = (
            _get_attr(item, "arguments")
            or _get_attr(item, "input")
            or _get_attr(item, "args")
            or ""
        )
        args_obj: Dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args_obj = args_raw
        else:
            try:
                args_obj = json.loads(args_raw)
            except Exception:
                args_obj = {"raw": str(args_raw)}
        tool_calls.append({"id": call_id, "name": name, "arguments": args_obj})
    return tool_calls


def extract_usage(response: Any) -> Dict[str, Any]:
    """Normalize Responses usage into prompt/completion/total tokens."""
    usage_obj = _get_attr(response, "usage") or {}
    input_tokens = _to_int(_get_attr(usage_obj, "input_tokens"))
    output_tokens = _to_int(_get_attr(usage_obj, "output_tokens"))
    total_tokens = _to_int(_get_attr(usage_obj, "total_tokens"))
    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _resolve_openai_credentials(
    *,
    model: str,
    config: Optional[Config],
    api_key: Optional[str],
    api_base: Optional[str],
    extra_headers: Optional[Dict[str, str]],
) -> Tuple[str, Optional[str], Optional[Dict[str, str]]]:
    """Resolve API key/base/headers from explicit params or config."""
    if api_key or api_base or extra_headers:
        resolved_key = (api_key or "").strip()
        resolved_base = (api_base or "").strip() or None
        resolved_headers = extra_headers or None
        return _finalize_credentials(resolved_key, resolved_base, resolved_headers)

    if config is None:
        raise ValueError("Config is required when api_key is not provided.")

    candidates = _collect_channel_params(model, config)
    if candidates:
        params = random.choice(candidates)
        resolved_key = str(params.get("api_key") or "").strip()
        resolved_base = str(params.get("api_base") or "").strip() or None
        resolved_headers = params.get("extra_headers") or None
        return _finalize_credentials(resolved_key, resolved_base, resolved_headers)

    keys = get_api_keys_for_model(model, config)
    resolved_key = keys[0] if keys else ""
    extra = extra_litellm_params(model, config)
    resolved_base = extra.get("api_base")
    resolved_headers = extra.get("extra_headers")
    return _finalize_credentials(resolved_key, resolved_base, resolved_headers)


def _finalize_credentials(
    api_key: str,
    api_base: Optional[str],
    extra_headers: Optional[Dict[str, str]],
) -> Tuple[str, Optional[str], Optional[Dict[str, str]]]:
    if not api_key:
        if _is_local_base_url(api_base):
            api_key = "EMPTY"
        else:
            raise ValueError("OpenAI API key is required for Responses API.")
    return api_key, api_base, extra_headers


def _collect_channel_params(model: str, config: Config) -> List[Dict[str, Any]]:
    params_list: List[Dict[str, Any]] = []
    for entry in getattr(config, "llm_model_list", []) or []:
        params = entry.get("litellm_params", {}) or {}
        model_name = str(entry.get("model_name") or params.get("model") or "").strip()
        if not model_name:
            continue
        if model_name == model or params.get("model") == model:
            params_list.append(params)
    return params_list


def _is_local_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    return parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}


def _convert_content_to_input_parts(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"type": "input_text", "text": item.get("text") or ""})
                    continue
                if item_type == "image_url":
                    image_url = item.get("image_url") or {}
                    url_value = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url_value:
                        parts.append({"type": "input_image", "image_url": url_value})
                        continue
            parts.append({"type": "input_text", "text": json.dumps(item, ensure_ascii=False)})
        if parts:
            return parts

    if content is None:
        text_value = ""
    elif isinstance(content, str):
        text_value = content
    else:
        text_value = json.dumps(content, ensure_ascii=False)
    return [{"type": "input_text", "text": text_value}]


def _simplify_input_if_possible(input_items: List[Dict[str, Any]]) -> Any:
    """Return a simplified input payload when possible (still spec-compliant)."""
    if len(input_items) != 1:
        return input_items
    item = input_items[0]
    if item.get("type") != "message" or item.get("role") != "user":
        return input_items
    content = item.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return input_items
    part = content[0]
    if isinstance(part, dict) and part.get("type") == "input_text":
        return part.get("text") or ""
    return input_items


def _extract_text_from_content(content: Any) -> List[str]:
    if not content:
        return []
    if isinstance(content, str):
        return [content]
    texts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text_val = part.get("text") or part.get("output_text")
            if text_val:
                texts.append(str(text_val))
    return texts


def _extract_instructions(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Collect system/developer messages into a single instructions string."""
    chunks: List[str] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"system", "developer"}:
            continue
        chunks.append(_content_to_text(msg.get("content")))
    combined = "\n\n".join([c for c in chunks if c])
    return combined or None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url") or {}
                    url_value = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url_value:
                        parts.append(f"[image_url]{url_value}")
                        continue
            parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join([p for p in parts if p])
    return json.dumps(content, ensure_ascii=False)


def _can_use_simple_messages(messages: List[Dict[str, Any]]) -> bool:
    """Check if all messages can be represented as role/content strings."""
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            return False
        if role == "assistant" and msg.get("tool_calls"):
            return False
        if not _content_is_text_only(msg.get("content")):
            return False
    return True


def _content_is_text_only(content: Any) -> bool:
    if content is None or isinstance(content, str):
        return True
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    continue
                # Any non-text (image/audio/etc) not supported in compat mode
                return False
            else:
                return False
        return True
    return False


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _is_official_openai_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return True
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"api.openai.com"}


def should_use_responses_strict(config: Optional[Config], api_base: Optional[str]) -> bool:
    """Decide whether to include strict=true in Responses tool definitions."""
    env_flag = os.getenv("OPENAI_RESPONSES_STRICT", "").strip().lower()
    if env_flag in {"0", "false", "no", "off"}:
        return False
    if env_flag in {"1", "true", "yes", "on"}:
        return True
    return _is_official_openai_base_url(api_base)


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0
