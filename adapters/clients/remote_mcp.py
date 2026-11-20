"""A bounded stdio forwarding transport for the entry's stateless HTTP MCP service.

Authentication is read from the existing client's token file for each request.
This process owns no memory database and creates no human authorization. Tool
schemas, validation and results remain the remote server's, not a second API.
"""

from __future__ import annotations

import json
import sys
from typing import Any

MAX_LINE = 1024 * 1024


def forward(config: dict, request: object, post) -> dict[str, Any] | None:
    """Forward one protocol message, preserving server errors and never reporting a failed call as success."""
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    response = post(config, request, 20.0, endpoint="/mcp")
    if "id" not in request:
        return None
    if isinstance(response, dict) and response.get("jsonrpc") == "2.0" and response.get("id") == request["id"]:
        return response
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "error": {"code": -32000, "message": "Scope Recall remote MCP unavailable or unauthorized"},
    }


def serve(config: dict, post) -> int:
    """Serve JSON-lines until input closes; no retries or durable copies of tool requests."""
    while raw := sys.stdin.buffer.readline(MAX_LINE + 1):
        if len(raw) > MAX_LINE:
            return 1
        try:
            request = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            request = None
        response = forward(config, request, post)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            sys.stdout.flush()
    return 0
