#!/usr/bin/env python3
"""Inventory Hermes source without importing it; compare reviewed host surfaces.

This is syntax inspection, not control-flow or provenance analysis. Literal roles,
module string constants and named calls are visible; aliases, dynamic roles and
runtime dictionary updates need human review. Provider calls require "memory" in
the receiver expression or file path. This name heuristic misses aliases such as
p.sync_turn outside memory-named files and can include unrelated memory objects.
Test/evaluation trees and environments are excluded from runtime observations.
Hook loss is relative to observed baseline calls; an unobserved dispatch wrapper
is not evidence that a call was removed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"tests", "test", "evals", "benchmarks", "node_modules", "venv", "__pycache__"}
FORMAT = 2
REFRESH_USAGE = "python scripts/hermes_surface.py refresh <old-inventory> <Hermes-source> --output <new-inventory>"


def encode(value: Any) -> str:
    """Serialize portable data with stable ordering and a final LF."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def digest(value: Any) -> str:
    """Return a compact deterministic identity from the complete serialized value."""
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()[:20]


def syntax(path: Path) -> ast.Module:
    """Parse bytes so Python encoding declarations are honored, never executed."""
    return ast.parse(path.read_bytes(), filename=str(path))


def assignments(tree: ast.AST) -> dict[str, ast.AST]:
    """Read direct assignments only; inner scopes cannot redefine module constants."""
    result = {}
    for node in getattr(tree, "body", []):
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        value = getattr(node, "value", None)
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                result[target.id] = value
    return result


def literal(node: ast.AST | None, constants: dict[str, ast.AST], seen: tuple[str, ...] = ()) -> Any:
    """Resolve a deliberately small literal language, with cycles left unresolved."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants and node.id not in seen:
        return literal(constants[node.id], constants, (*seen, node.id))
    if isinstance(node, (ast.Tuple, ast.List)):
        items = []
        for item in node.elts:
            value = literal(item.value if isinstance(item, ast.Starred) else item, constants, seen)
            if isinstance(item, ast.Starred) and isinstance(value, (tuple, list)):
                items.extend(value)
            else:
                items.append(value)
        return items
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left, constants, seen), literal(node.right, constants, seen)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    return None


def string_head(node: ast.AST | None, constants: dict[str, ast.AST]) -> str:
    """Return the constant leading text before the first dynamic fragment."""
    value = literal(node, constants)
    if isinstance(value, str):
        return value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return string_head(node.left, constants)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for item in node.values:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                break
            parts.append(item.value)
        return "".join(parts)
    return ""


def describe(node: ast.AST | None, constants: dict[str, ast.AST]) -> dict:
    """Keep readable beginnings and a full digest, so truncation cannot hide edits."""
    expression = ast.unparse(node) if node is not None else "<absent>"
    head = string_head(node, constants)
    return {"expression": expression[:300], "prefix": head[:160], "digest": digest([expression, head])}


def nodes(tree: ast.AST, scope: str = "<module>"):
    """Yield lexical qualified names, including classes and nested functions."""
    if isinstance(tree, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        scope = tree.name if scope == "<module>" else f"{scope}.{tree.name}"
    yield tree, scope
    for child in ast.iter_child_nodes(tree):
        yield from nodes(child, scope)


def leaf(node: ast.AST) -> str:
    """Return a syntactic call name without resolving aliases or receiver types."""
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else ""


def marker_definitions(repo: Path = ROOT) -> dict[str, str]:
    """Extract every module string or string-tuple member in the boundary AST."""
    constants = assignments(syntax(repo / "adapters/hermes/boundary.py"))
    result = {}
    for name, node in constants.items():
        value = literal(node, constants)
        if isinstance(value, str):
            result[name] = value
        elif isinstance(value, list):
            result.update({f"{name}[{index}]": text for index, text in enumerate(value) if isinstance(text, str)})
    if not result:
        raise ValueError("no boundary markers extracted; inspect adapters/hermes/boundary.py")
    return dict(sorted(result.items()))


def registered_hooks(repo: Path) -> list[str]:
    """Read the profile-backed registration loop and explicit registrations from our AST."""
    hook_tree = syntax(repo / "adapters/hermes/hooks.py")
    hooks = assignments(hook_tree)
    observers = hooks.get("_OBSERVERS")
    if observers is None or 'HOSTS["hermes"].hook_observers' not in ast.unparse(observers).replace("'", '"'):
        raise ValueError("hook registration shape changed; inspect adapters/hermes/hooks.py")
    profile = syntax(repo / "host_registry.py")
    registered = []
    for node in ast.walk(profile):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "hermes" and isinstance(value, ast.Call):
                    for kw in value.keywords:
                        if kw.arg == "hook_observers":
                            registered = [pair[0] for pair in ast.literal_eval(kw.value)]
    supported = hooks.get("_SUPPORTED_HOOKS")
    if supported is not None:
        explicit = literal(supported, hooks)
        if isinstance(explicit, list) and all(isinstance(name, str) for name in explicit):
            registered = explicit
        elif ast.unparse(supported) != "tuple(_OBSERVERS)":
            raise ValueError("supported hook shape changed; inspect adapters/hermes/hooks.py")
    for node in ast.walk(hook_tree):
        if isinstance(node, ast.Call) and leaf(node.func) == "register_hook" and node.args:
            event = literal(node.args[0], hooks)
            if isinstance(event, str):
                registered.append(event)
    if not registered:
        raise ValueError("no registered hooks extracted; inspect adapters/hermes/hooks.py")
    return sorted(set(registered))


def interface_definitions(repo: Path = ROOT) -> dict[str, list[str]]:
    """Read public provider implementations and registered hooks without importing them.

    Provider calls are name candidates, not resolved receiver types. A changed
    registration shape fails explicitly instead of yielding an empty hook list.
    """
    methods = set()
    for name in ("provider.py", "tool_surface.py", "protocol.py"):
        for node in syntax(repo / "adapters/hermes" / name).body:
            if isinstance(node, ast.ClassDef):
                methods.update(
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_")
                )
    if not methods:
        raise ValueError("no host interfaces extracted; inspect hook registration and provider classes")
    return {"hooks": registered_hooks(repo), "provider_methods": sorted(methods)}


def is_link(path: Path) -> bool:
    """Windows reparse points include junctions, even on Python without is_junction."""
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def source_files(root: Path):
    """Prune environments before descent; never follow source symlinks or junctions."""

    def fail(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in EXCLUDED and not name.startswith(".") and not is_link(Path(directory, name))
        )
        for name in sorted(files):
            path = Path(directory, name)
            if path.suffix == ".py" and not is_link(path):
                yield path


def fields(node: ast.AST) -> dict[str, ast.AST]:
    """Read explicit dictionary keys or call keywords; unpacking remains unresolved."""
    if isinstance(node, ast.Dict):
        return {
            k.value: v
            for k, v in zip(node.keys, node.values, strict=True)
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
    if isinstance(node, ast.Call):
        return {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
    return {}


def mapping_keys(tree: ast.AST) -> dict:
    """Collect possible spread keys within each lexical scope, without flow claims."""
    result: dict = defaultdict(lambda: defaultdict(set))
    for node, scope in nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[scope][target.id].update(k for k in fields(node.value) if isinstance(k, str))
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    key = literal(target.slice, {})
                    if isinstance(key, str):
                        result[scope][target.value.id].add(key)
    helpers = spread_helpers(tree)
    return {scope: {**helpers, **values} for scope, values in result.items()} | {"<helpers>": helpers}


def spread_helpers(tree: ast.AST) -> dict[str, set[str]]:
    """Expose possible fields from local dataclass-to-dict spread helpers.

    Only an explicit constructor followed by a method returning an asdict(self)
    comprehension is recognized. Variable receivers and arbitrary helpers remain
    unresolved; these are possible fields, not a proof of runtime types.
    """
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        names = {n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returned = [n.value for n in ast.walk(method) if isinstance(n, ast.Return)]
            if len(returned) == 1 and isinstance(returned[0], ast.DictComp):
                comp = returned[0]
                if any(ast.unparse(g.iter) == "asdict(self).items()" for g in comp.generators):
                    result[f"@{node.name}.{method.name}"] = names
    return result


def call_parameters(node: ast.Call, keys: dict[str, set[str]]) -> dict:
    """Record explicit keywords and possible keys of unresolved spreads."""
    spreads = []
    for kw in node.keywords:
        if kw.arg is None:
            expression = ast.unparse(kw.value)
            known = keys.get(expression, set()) | {key for key in fields(kw.value) if isinstance(key, str)}
            if isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Attribute):
                receiver = kw.value.func.value
                if isinstance(receiver, ast.Call):
                    known |= keys.get(f"@{leaf(receiver.func)}.{kw.value.func.attr}", set())
            spreads.append({"expression": expression, "possible_keys": sorted(known)})
    return {"keywords": sorted(kw.arg for kw in node.keywords if kw.arg), "spreads": spreads}


def constructions(node: ast.AST, constants: dict[str, ast.AST]) -> dict | None:
    """Recognize literal user-role dictionaries and calls, including SessionDB positional roles."""
    data = fields(node)
    # SessionDB accepts (session_id, role, content); the in-memory append helper
    # accepts (messages, row). Only a literal second argument identifies this form.
    if isinstance(node, ast.Call) and leaf(node.func) == "append_message" and len(node.args) > 1:
        if isinstance(node.args[1], ast.Constant) and node.args[1].value == "user":
            data = {"role": node.args[1], **data}
            if len(node.args) > 2:
                data.setdefault("content", node.args[2])
    # A concatenated/dynamic role is not proof of a user message.
    role = data.get("role")
    if not isinstance(role, (ast.Constant, ast.Name)) or literal(role, constants) != "user":
        return None
    content = next((data[k] for k in ("content", "text", "message", "parts") if k in data), None)
    if content is None and isinstance(node, ast.Call) and leaf(node.func) == "mirror_to_session" and len(node.args) > 2:
        content = node.args[2]
    return {
        "shape": ast.unparse(node.func) if isinstance(node, ast.Call) else "dict",
        "content": describe(content, constants),
        "display_kind": describe(data["display_kind"], constants) if "display_kind" in data else None,
    }


def imported_constants(tree: ast.Module, root: Path, relative: str, cache: dict) -> dict[str, ast.AST]:
    """Resolve explicitly imported uppercase string constants, including lazy imports.

    Imports are inspected as text and restricted to the scanned source root.
    Ambiguous aliases remain expressions. No package initialization is performed.
    """
    result = assignments(tree)
    imports: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            base = relative.split("/")[: -node.level] if node.level else []
            module = "/".join([*base, *node.module.split(".")]) + ".py"
            for alias in node.names:
                name = alias.asname or alias.name
                if name.isupper():
                    imports[name].add((module, alias.name))
    for name, candidates in imports.items():
        if name not in result and len(candidates) == 1:
            module, symbol = next(iter(candidates))
            value = imported_literal(root, module, symbol, cache, ())
            if isinstance(value, str):
                result[name] = ast.Constant(value)
    return result


def imported_literal(root: Path, module: str, symbol: str, cache: dict, seen: tuple) -> Any:
    """Follow explicit source imports only; cycles and nonliteral exports stay unknown."""
    key = (module, symbol)
    path = root / module
    if key in seen or not path.is_file():
        return None
    if any(is_link(root.joinpath(*Path(module).parts[:i])) for i in range(1, len(Path(module).parts) + 1)):
        return None
    if module not in cache:
        cache[module] = syntax(path)
    tree = cache[module]
    value = literal(assignments(tree).get(symbol), assignments(tree))
    if isinstance(value, str):
        return value
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if (alias.asname or alias.name) == symbol:
                    return imported_literal(
                        root, node.module.replace(".", "/") + ".py", alias.name, cache, (*seen, key)
                    )
    return None


def observe_call(node: ast.Call, where: dict, constants: dict, keys: dict, interfaces: dict, result: dict) -> None:
    """Record literal hook dispatch and named provider/event call candidates."""
    name = leaf(node.func)
    event = literal(node.args[0], constants) if node.args else None
    if name not in {"invoke_hook", "_invoke_hook", "MessageEvent", *interfaces["provider_methods"]}:
        return
    parameters = call_parameters(node, keys)
    if name in {"invoke_hook", "_invoke_hook"} and event in interfaces["hooks"]:
        result["calls"].append({**where, "kind": "hook", "name": event, **parameters})
    elif (
        isinstance(node.func, ast.Attribute)
        and name in interfaces["provider_methods"]
        and "memory" in (where["file"] + ast.unparse(node.func.value)).lower()
    ):
        result["calls"].append(
            {
                **where,
                "kind": "provider_candidate",
                "name": name,
                "receiver": ast.unparse(node.func.value),
                **parameters,
            }
        )
    if name == "MessageEvent":
        result["message_events"].append({**where, **parameters})


def scan(root: Path, repo: Path | None = None) -> dict:
    """Return sorted observations; malformed or empty source fails the whole scan."""
    if not root.is_dir():
        raise ValueError(f"Hermes source directory does not exist: {root}")
    markers = marker_definitions(repo or ROOT)
    interfaces = interface_definitions(repo or ROOT)
    result: dict[str, Any] = {
        "format": FORMAT,
        "excluded_directories": sorted(EXCLUDED),
        "user_rows": [],
        "markers": [],
        "interfaces": interfaces,
        "calls": [],
        "message_events": [],
    }
    locations: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    count = 0
    source_hash = hashlib.sha256()
    cache: dict[str, ast.Module] = {}
    for path in sorted(source_files(root), key=lambda path: path.relative_to(root).as_posix()):
        tree = syntax(path)
        count += 1
        relative = path.relative_to(root).as_posix()
        source_hash.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
        constants, keys = imported_constants(tree, root, relative, cache), mapping_keys(tree)
        for node, scope in nodes(tree):
            where = {"file": relative, "function": scope, "line": getattr(node, "lineno", 1)}
            row = constructions(node, constants)
            if row is not None:
                result["user_rows"].append({**where, **row})
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for name, text in markers.items():
                    if text == node.value or (not text.isidentifier() and text in node.value):
                        locations[name].add((relative, scope, node.lineno))
            if isinstance(node, ast.Call):
                observe_call(node, where, constants, keys.get(scope, keys["<helpers>"]), interfaces, result)
    if not count:
        raise ValueError(f"no Python source found in {root}")
    for name, text in markers.items():
        result["markers"].append(
            {
                "name": name,
                "text": text,
                "locations": [{"file": f, "function": q, "line": line} for f, q, line in sorted(locations[name])],
            }
        )
    result["source"] = {"python_files": count, "sha256": source_hash.hexdigest()}
    for section in ("user_rows", "calls", "message_events"):
        seen: dict[str, int] = defaultdict(int)
        for item in result[section]:
            identity = {k: v for k, v in item.items() if k not in {"line", "display_kind"}}
            key = f"{item['file']}::{item['function']}::{digest(identity)}"
            seen[key] += 1
            item["id"] = f"{key}:{seen[key]}"
        result[section].sort(key=lambda item: item["id"])
    return result


def load(path: Path, *, allow_legacy: bool = False) -> dict:
    """Read validated inventories; only refresh accepts the previous format."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") not in ({1, FORMAT} if allow_legacy else {FORMAT}):
        raise ValueError(f"unsupported inventory format in {path}; use refresh to migrate the inventory")
    for section in ("user_rows", "calls", "message_events"):
        if not isinstance(value.get(section), list) or not all(isinstance(item, dict) for item in value[section]):
            raise ValueError(f"invalid {section} records in {path}")
        ids = [item["id"] for item in value[section]]
        if not all(isinstance(identity, str) for identity in ids):
            raise ValueError(f"invalid {section} IDs in {path}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {section} IDs in {path}")
    for section in ("markers", "interfaces", "excluded_directories"):
        value[section]
    return value


def semantic(value: Any) -> Any:
    """Line movement and human dispositions do not change a source observation."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {"line", "disposition", "id"}}
    if isinstance(value, list):
        return sorted((semantic(v) for v in value), key=encode)
    return value


def location(item: dict) -> str:
    """Format a source reference using a relative path and a lexical function name."""
    return f"{item['file']}:{item['line']} ({item['function']})"


def row_text(item: dict) -> str:
    """Render the content beginning and display kind for human review."""
    content = item["content"]
    text = content["prefix"] or content["expression"]
    kind = item["display_kind"]
    return f"{location(item)} content={text!r}; display_kind={kind['expression'] if kind else '<absent>'}"


def row_changes(old: list[dict], new: list[dict]) -> list[str]:
    """Match unchanged identities before pairing one unambiguous lexical edit."""
    lines = []
    before, after = {r["id"]: r for r in old}, {r["id"]: r for r in new}
    for key in sorted(before.keys() & after.keys()):
        if semantic(before[key]) != semantic(after[key]):
            lines.append(f"USER CHANGED: {row_text(before[key])} -> {row_text(after[key])}; review disposition")
    removed = [before[k] for k in sorted(before.keys() - after.keys())]
    added = [after[k] for k in sorted(after.keys() - before.keys())]
    # Exact identities are matched first. Pair only a single remaining row in
    # one lexical/shape group; ambiguous edits remain explicit removals/additions.
    for item in removed[:]:

        def group(row):
            return row["file"], row["function"], row["shape"]

        matches = [r for r in added if group(r) == group(item)]
        peers = [r for r in removed if group(r) == group(item)]
        if len(matches) == len(peers) == 1:
            other = matches[0]
            lines.append(f"USER CHANGED: {row_text(item)} -> {row_text(other)}; no disposition for changed content")
            removed.remove(item)
            added.remove(other)
    lines.extend(f"USER REMOVED: {row_text(r)}" for r in removed)
    lines.extend(f"USER ADDED (no disposition): {row_text(r)}" for r in added)
    return lines


def marker_changes(recorded: list[dict], current: list[dict]) -> tuple[list[str], list[str]]:
    """Separate absent dependencies from movements of surviving marker text."""
    required, reference = [], []
    old, new = {m["name"]: m for m in recorded}, {m["name"]: m for m in current}

    def definitions(markers):
        return {name: item["text"] for name, item in markers.items()}

    if definitions(old) != definitions(new):
        changed = sorted(
            name for name in old.keys() | new.keys() if old.get(name, {}).get("text") != new.get(name, {}).get("text")
        )
        required.append(f"INVENTORY STALE: boundary marker definitions differ: {changed}")
    for name, marker in sorted(new.items()):
        if not marker["locations"]:
            required.append(f"MARKER MISSING: {name}={marker['text']!r}; no matching Hermes string constant")
        elif name in old and old[name]["text"] == marker["text"]:
            previous = {(p["file"], p["function"]): p for p in old[name]["locations"]}
            places = {(p["file"], p["function"]): p for p in marker["locations"]}
            if previous.keys() != places.keys():
                gone = ", ".join(location(previous[k]) for k in sorted(previous.keys() - places.keys())) or "none"
                added = ", ".join(location(places[k]) for k in sorted(places.keys() - previous.keys())) or "none"
                reference.append(f"MARKER LOCATIONS CHANGED: {name}={marker['text']!r}; removed {gone}; added {added}")
    return required, reference


def keyword_unions(calls: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Union all statically observed keywords and possible spread keys per interface."""
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for call in calls:
        keys = result[call["kind"], call["name"]]
        keys.update(call["keywords"])
        for spread in call["spreads"]:
            keys.update(spread["possible_keys"])
    return result


def missing_calls(recorded: dict, current: dict) -> list[str]:
    """Report loss of previously observed hook calls and interface keywords."""
    old, new = keyword_unions(recorded["calls"]), keyword_unions(current["calls"])
    required = []
    for name in current["interfaces"]["hooks"]:
        if ("hook", name) in old and ("hook", name) not in new:
            required.append(f"HOOK MISSING: {name}; no Hermes dispatch call found")
    for (kind, name), keywords in sorted(old.items()):
        lost = sorted(keywords - new.get((kind, name), set()))
        if lost:
            required.append(f"KEYWORDS MISSING: {kind} {name}; no remaining call supplies {lost}")
    return required


def difference_groups(recorded: dict, current: dict) -> dict[str, list[str]]:
    """Only missing dependencies and stale boundary/hook definitions require action."""
    required, reference = marker_changes(recorded["markers"], current["markers"])
    if set(recorded["interfaces"]["hooks"]) != set(current["interfaces"]["hooks"]):
        required.append(
            f"INVENTORY STALE: registered hooks {recorded['interfaces']['hooks']} -> {current['interfaces']['hooks']}"
        )
    required.extend(missing_calls(recorded, current))
    reference.extend(row_changes(recorded["user_rows"], current["user_rows"]))
    for row in recorded["user_rows"]:
        if row["display_kind"] is None and not row.get("disposition"):
            reference.append(f"USER UNREVIEWED (no disposition): {row_text(row)}")
    if recorded["interfaces"]["provider_methods"] != current["interfaces"]["provider_methods"]:
        reference.append("PROVIDER METHODS CHANGED: " + encode(current["interfaces"]["provider_methods"]).strip())
    for section in ("calls", "message_events", "excluded_directories"):
        if semantic(recorded[section]) != semantic(current[section]):
            reference.extend(surface_changes(section, recorded[section], current[section]))
    return {"required": required, "reference": reference}


def render(groups: dict[str, list[str]], refresh_command: str = REFRESH_USAGE) -> str:
    """Render both levels and an explicit refresh remedy for stale inventories."""
    if not any(groups.values()):
        return ""
    required, reference = groups["required"], groups["reference"]
    sections = [f"Action required ({len(required)}):", *required]
    if any(line.startswith("INVENTORY STALE:") for line in required):
        sections.extend(["Refresh the inventory first:", f"  {refresh_command}"])
    sections.extend(["", f"For reference ({len(reference)}):", *reference])
    return "\n".join(sections) + "\n"


def diff(recorded: dict, current: dict) -> str:
    """Return human-readable review work, or an empty string for matching surfaces."""
    return render(difference_groups(recorded, current))


def refresh(recorded: dict, current: dict) -> str:
    """Attach only proven matching dispositions to a fresh scan, keeping its source.

    Repeated identities use occurrence suffixes. When a group's size changes,
    those suffixes cannot prove which identical row survived, so none is carried.
    The supplied current inventory is mutated; the old inventory is never edited.
    """
    old = {row["id"]: row for row in recorded["user_rows"]}

    def stem(row):
        return row["id"].rsplit(":", 1)[0]

    before = Counter(stem(row) for row in old.values())
    after = Counter(stem(row) for row in current["user_rows"])
    carried, missing = set(), []
    for row in current["user_rows"]:
        row.pop("disposition", None)
        prior = old.get(row["id"])
        unchanged = prior is not None and all(prior[key] == row[key] for key in ("content", "display_kind"))
        certain = unchanged and before[stem(row)] == after[stem(row)]
        if certain and prior is not None and "disposition" in prior:
            row["disposition"] = prior["disposition"]
            carried.add(row["id"])
        elif not certain or row["display_kind"] is None:
            missing.append(row)
    dropped = [row for key, row in sorted(old.items()) if "disposition" in row and key not in carried]
    lines = [
        f"Rows without disposition ({len(missing)}):",
        *(row_text(row) for row in missing),
        "",
        f"Discarded dispositions ({len(dropped)}):",
        *(row_text(row) for row in dropped),
    ]
    return "\n".join(lines) + "\n"


def parameter_changes(section: str, old: dict, new: dict) -> list[str]:
    """Show keyword additions/removals when a single lexical call changed."""

    def group(item):
        return tuple(item.get(k) for k in ("file", "function", "kind", "name", "receiver"))

    lines = []
    for key, item in sorted(list(old.items())):
        peers = [k for k, r in old.items() if group(r) == group(item)]
        matches = [k for k, r in new.items() if group(r) == group(item)]
        if len(peers) != 1 or len(matches) != 1:
            continue
        other = new.pop(matches[0])
        old.pop(key)
        removed = sorted(set(item["keywords"]) - set(other["keywords"]))
        added = sorted(set(other["keywords"]) - set(item["keywords"]))
        spreads = ""
        if item["spreads"] != other["spreads"]:
            spreads = f"; spreads {json.dumps(item['spreads'])} -> {json.dumps(other['spreads'])}"
        lines.append(
            f"{section.upper()} PARAMETERS CHANGED: {location(other)} "
            f"{other.get('name', 'MessageEvent')} keywords +{added} -{removed}{spreads}"
        )
    return lines


def surface_changes(section: str, before: Any, after: Any) -> list[str]:
    """Use complete records for parameter changes, including unresolved spreads."""
    if not isinstance(before, list) or (before and not isinstance(before[0], dict)):
        return [f"{section.upper()} CHANGED: {encode(before).strip()} -> {encode(after).strip()}"]
    old = {item["id"]: item for item in before}
    new = {item["id"]: item for item in after}
    gone = {k: old[k] for k in old.keys() - new.keys()}
    added = {k: new[k] for k in new.keys() - old.keys()}
    lines = parameter_changes(section, gone, added) if section in {"calls", "message_events"} else []
    old, new = gone, added
    for label, values, keys in (("REMOVED", old, old.keys()), ("ADDED", new, new.keys())):
        for key in sorted(keys):
            item = values[key]
            lines.append(
                f"{section.upper()} {label}: {location(item)} {json.dumps(semantic(item), ensure_ascii=False)}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run scan/refresh/diff; status 1 means action or strict drift, 2 an input error."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("scan")
    collect.add_argument("root", type=Path)
    collect.add_argument("--output", type=Path)
    compare = commands.add_parser("diff")
    compare.add_argument("recorded", type=Path)
    compare.add_argument("root", type=Path)
    compare.add_argument("--strict", action="store_true", help="fail for any reported difference")
    update = commands.add_parser("refresh")
    update.add_argument("recorded", type=Path)
    update.add_argument("root", type=Path)
    update.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        recorded = load(args.recorded, allow_legacy=args.command == "refresh") if args.command != "scan" else {}
        current = scan(args.root)
        if args.command == "scan":
            output = encode(current)
            if args.output:
                args.output.write_text(output, encoding="utf-8", newline="\n")
            else:
                sys.stdout.write(output)
            return 0
        if args.command == "refresh":
            output = refresh(recorded, current)
            args.output.write_text(encode(current), encoding="utf-8", newline="\n")
            sys.stdout.write(output)
            return 0
        groups = difference_groups(recorded, current)
        command = shlex.join(
            [
                "python",
                "scripts/hermes_surface.py",
                "refresh",
                args.recorded.as_posix(),
                args.root.as_posix(),
                "--output",
                args.recorded.with_suffix(".refreshed.json").as_posix(),
            ]
        )
        output = render(groups, command)
        sys.stdout.write(output or "No Hermes surface changes.\n")
        return int(bool(groups["required"]) or (args.strict and bool(groups["reference"])))
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
        print(f"Hermes surface check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
