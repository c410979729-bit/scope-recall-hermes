"""Byte-bounded Hermes snapshots with recoverable loss accounting in the shared spool."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from scope_recall.core.file_lock import advisory_file_lock

MAX_BYTES = 64 * 1024 * 1024


def encode(value: dict) -> bytes:
    """Use the same compact UTF-8 representation for the queue and decoded HTTP envelope."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _folder(config: dict) -> Path:
    return config["state_dir"] / "spool"


@contextmanager
def sending(config: dict):
    """Serialize senders without holding the queue-edit lock across network I/O."""
    config["state_dir"].mkdir(parents=True, exist_ok=True)
    with advisory_file_lock(config["state_dir"] / "hermes-send.lock", timeout_seconds=0):
        yield


def _write(path: Path, value: dict) -> None:
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _state_path(config: dict) -> Path:
    return config["state_dir"] / "hermes-queue-loss.json"


def _state(config: dict) -> dict:
    path = _state_path(config)
    if not path.exists():
        return {"dropped_count": 0, "first_dropped_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _recover(config: dict, state: dict) -> None:
    for pending in state.get("pending", []):
        name = pending["name"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("invalid Hermes loss journal")
        path = _folder(config) / name
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))["payload"]
            if (current.get("queue_key"), current.get("revision")) == (pending["key"], pending["revision"]):
                path.unlink()
    if state.get("pending"):
        state.pop("pending")
        _write(_state_path(config), state)


@contextmanager
def _editing(config: dict):
    folder = _folder(config)
    folder.mkdir(parents=True, exist_ok=True)
    with advisory_file_lock(folder / "hermes-queue.lock", timeout_seconds=1):
        _recover(config, _state(config))
        yield folder


def _discard(config: dict, items: list[Path], at: str, *, unstored: bool = False) -> None:
    state = _state(config)
    pending = []
    for path in items:
        payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
        pending.append({"name": path.name, "key": payload["queue_key"], "revision": payload.get("revision")})
    state.update(
        dropped_count=state["dropped_count"] + len(pending) + int(unstored),
        first_dropped_at=state.get("first_dropped_at") or at,
        pending=pending,
    )
    # Publish the loss and exact revisions before deleting anything. Recovery never increments the count again.
    _write(_state_path(config), state)
    _recover(config, state)


def put(config: dict, payload: dict, at: str) -> Path:
    """Coalesce a turn's snapshot and journal capacity losses before removing their exact revisions."""
    key = payload["queue_key"]
    if not isinstance(key, str) or len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
        raise ValueError("invalid Hermes queue identity")
    with _editing(config) as folder:
        old = next(iter(folder.glob(f"*-{key}.json")), None)
        path = old or folder / f"{time.time_ns():020d}-{key}.json"
        if old is not None:
            previous = json.loads(old.read_text(encoding="utf-8"))
            if previous["payload"].get("revision", 0) > payload.get("revision", 0):
                return path
        _write(path, {"payload": payload, "observed_at": at})
        items = sorted(folder.glob("*.json"))
        total = sum(item.stat().st_size for item in items)
        discarded = []
        for item in items:
            if total <= MAX_BYTES:
                break
            total -= item.stat().st_size
            discarded.append(item)
        if discarded:
            _discard(config, discarded, at)
        return path


def _matches(path: Path, payload: dict) -> bool:
    if not path.exists():
        return False
    current = json.loads(path.read_text(encoding="utf-8"))["payload"]
    return (current.get("queue_key"), current.get("revision")) == (payload.get("queue_key"), payload.get("revision"))


def acknowledge(config: dict, path: Path, payload: dict) -> None:
    """Remove only the sent revision; a concurrent newer snapshot remains queued."""
    with _editing(config):
        if _matches(path, payload):
            path.unlink()


def discard(config: dict, path: Path | None, payload: dict, at: str) -> None:
    """Account for an unsendable snapshot before deletion, including one rejected before retention."""
    with _editing(config):
        if path is None or _matches(path, payload):
            _discard(config, [] if path is None else [path], at, unstored=path is None)


def status(config: dict) -> dict:
    """Finish any interrupted loss deletion and report durable counts without conversation content."""
    with _editing(config) as folder:
        items = list(folder.glob("*.json"))
        loss = _state(config)
        return {
            "queued_turns": len(items),
            "queued_bytes": sum(item.stat().st_size for item in items),
            "capacity_bytes": MAX_BYTES,
            "dropped_count": loss.get("dropped_count", 0),
            "first_dropped_at": loss.get("first_dropped_at"),
        }
