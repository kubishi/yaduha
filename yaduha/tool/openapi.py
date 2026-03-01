"""Generate yaduha-compatible tools from an OpenAPI spec.

Usage:
    from yaduha.tool.openapi import openapi_tools

    tools = openapi_tools("https://dictionary.kubishi.com/openapi.json")
"""

from __future__ import annotations

import re
from typing import Any, ClassVar
from uuid import uuid4

import httpx
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import PrivateAttr

from yaduha.logger import Logger, NoLogger, get_log_context, inject_logs
from yaduha.tool import Tool


def _sanitize_name(raw: str) -> str:
    """Turn an operationId or path into a valid Python identifier."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or not name[0].isalpha():
        name = "op_" + name
    return name


def _openapi_type_to_json_schema(param_schema: dict) -> dict:
    """Convert an OpenAPI schema fragment to a JSON Schema property."""
    t = param_schema.get("type", "string")
    out: dict[str, Any] = {}

    if t == "integer":
        out["type"] = "integer"
    elif t == "number":
        out["type"] = "number"
    elif t == "boolean":
        out["type"] = "boolean"
    elif t == "array":
        out["type"] = "array"
        items = param_schema.get("items", {})
        out["items"] = _openapi_type_to_json_schema(items)
    else:
        out["type"] = "string"

    if "description" in param_schema:
        out["description"] = param_schema["description"]
    if "enum" in param_schema:
        out["enum"] = param_schema["enum"]
    if "default" in param_schema:
        out["default"] = param_schema["default"]

    return out


def _build_tool_schema(
    name: str,
    description: str,
    operation: dict,
) -> ChatCompletionToolParam:
    """Build a ChatCompletionToolParam from an OpenAPI operation.

    Produces a strict-mode-compliant schema: all properties are required
    (optional params are typed as ``["type", "null"]`` so the LLM can
    omit them by passing ``null``), and every object carries
    ``additionalProperties: false``.
    """
    properties: dict[str, Any] = {}
    required_by_spec: set[str] = set()

    # Query and path parameters
    for param in operation.get("parameters", []):
        param_name = param["name"]
        schema = param.get("schema", {"type": "string"})
        prop = _openapi_type_to_json_schema(schema)
        if "description" not in prop and "description" in param:
            prop["description"] = param["description"]
        properties[param_name] = prop
        if param.get("required", False):
            required_by_spec.add(param_name)

    # Request body (flatten top-level properties)
    request_body = operation.get("requestBody", {})
    body_content = request_body.get("content", {})
    json_body = body_content.get("application/json", {})
    body_schema = json_body.get("schema", {})
    if body_schema.get("type") == "object":
        for prop_name, prop_schema in body_schema.get("properties", {}).items():
            properties[prop_name] = _openapi_type_to_json_schema(prop_schema)
        for req_name in body_schema.get("required", []):
            required_by_spec.add(req_name)

    # Strict mode: all properties must be in "required".
    # Optional params become nullable so the LLM can pass null.
    for prop_name, prop in properties.items():
        if prop_name not in required_by_spec:
            original_type = prop.get("type", "string")
            prop["type"] = [original_type, "null"]

    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }

    return ChatCompletionToolParam(
        type="function",
        function={
            "name": name,
            "description": description,
            "parameters": parameters_schema,
            "strict": True,
        },
    )


# ---------------------------------------------------------------------------
# OpenAPITool
# ---------------------------------------------------------------------------


class OpenAPITool(Tool[str]):
    """A tool backed by a single OpenAPI endpoint."""

    name: ClassVar[str] = "openapi_tool"
    description: ClassVar[str] = ""

    base_url: str
    path: str
    method: str
    path_params: list[str]
    _tool_schema: ChatCompletionToolParam = PrivateAttr()

    def __init__(self, *, _tool_schema: ChatCompletionToolParam, **data: Any):
        super().__init__(**data)
        self._tool_schema = _tool_schema

    # -- Override validations that introspect _run() -----------------------

    @classmethod
    def _validate_run(cls) -> None:
        pass

    def _validate_examples(self) -> None:
        pass

    # -- Schema comes from the stored OpenAPI-derived schema ---------------

    def get_tool_call_schema(self) -> ChatCompletionToolParam:
        return self._tool_schema

    # -- Call: make HTTP request -------------------------------------------

    def __call__(self, **kwargs: Any) -> str:  # type: ignore[override]
        toolchain = get_log_context().get("TOOLCHAIN", "")
        toolchain = f"{toolchain}/{uuid4()}" if toolchain else str(uuid4())
        with inject_logs(tool=self.name, toolchain=toolchain):
            return self._run(**kwargs)

    def _run(self, **kwargs: Any) -> str:
        # Drop null values (optional params the LLM chose not to fill)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Substitute path parameters into the URL
        url = self.base_url.rstrip("/") + self.path
        for p in self.path_params:
            if p in kwargs:
                url = url.replace(f"{{{p}}}", str(kwargs.pop(p)))

        if self.method == "get":
            resp = httpx.get(url, params=kwargs, timeout=30)
        else:
            resp = httpx.request(self.method.upper(), url, json=kwargs, timeout=30)

        resp.raise_for_status()

        self.log(
            {
                "event": "openapi_tool_call",
                "url": url,
                "method": self.method,
                "params": kwargs,
                "status": resp.status_code,
            }
        )

        return resp.text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def openapi_tools(
    spec_url: str,
    *,
    logger: Logger | None = None,
) -> list[Tool]:
    """Fetch an OpenAPI spec and return one Tool per operation.

    Args:
        spec_url: URL to an openapi.json file.
        logger: Optional logger for tool calls.

    Returns:
        List of Tool instances, one per endpoint.
    """
    spec = httpx.get(spec_url, timeout=30).json()

    # Resolve base URL
    servers = spec.get("servers", [])
    if servers:
        base_url = servers[0].get("url", "")
    else:
        # Derive from spec_url (strip the filename)
        base_url = spec_url.rsplit("/", 1)[0]

    # Make relative server URLs absolute
    if base_url.startswith("/"):
        from urllib.parse import urlparse

        parsed = urlparse(spec_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}{base_url}"

    effective_logger = logger or NoLogger()
    tools: list[Tool] = []

    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue

            op_id = operation.get("operationId")
            if not op_id:
                op_id = f"{method}_{path}"
            op_name = _sanitize_name(op_id)

            summary = operation.get("summary", "")
            desc = operation.get("description", summary) or summary
            if not desc:
                desc = f"{method.upper()} {path}"

            tool_schema = _build_tool_schema(op_name, desc, operation)

            # Collect path parameter names
            path_params = re.findall(r"\{(\w+)\}", path)

            # Dynamic subclass so each tool gets its own ClassVar name/description
            ToolCls = type(
                f"OpenAPI_{op_name}",
                (OpenAPITool,),
                {"name": op_name, "description": desc},
            )

            tools.append(
                ToolCls(
                    base_url=base_url,
                    path=path,
                    method=method,
                    path_params=path_params,
                    logger=effective_logger,
                    _tool_schema=tool_schema,
                )
            )

    return tools
