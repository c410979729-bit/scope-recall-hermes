"""Resident recall target and handler for Hermes shared-store entries.

Runs inside the worker-owned loopback resident process so automatic prefetch and explicit tool recalls
execute outside the Hermes gateway's Python GIL while re-verifying the entry's manifest and exact session
audience on every request. Imports stay strictly within shared and client layers.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import tzinfo
from functools import partial
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from scope_recall.contracts import (
    ContractError,
    InstanceBinding,
    Origin,
    PrincipalKind,
    RecallRequest,
    TrustedContext,
    TrustedSourcePrincipal,
    utc_timestamp,
    validate_model_request,
)
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.entry_control import account_pin, local_account_matches
from scope_recall.core.private import normalize_project_path, outbound_blocked
from scope_recall.core.vector_failure import using_vector_trace
from scope_recall.runtime.remote_clock import stored_session_id as session_identity
from scope_recall.runtime.worker_launch import RUNTIME_CONFIG_FILENAME

from ..attachment_paths import UnverifiedAttachmentPath, local_attachment_stat, same_local_path
from ..audiences import LOCAL_PLATFORMS, LOCAL_USER_ID, approved_local_platforms
from ..host_identity import (
    HermesRuntimeScope,
    context_ref,
    entry_owner,
    owner_subjects,
    principal_ref,
    resolve_runtime_audience,
)
from ..installation import HermesIdentityError, InstallationManifest, bounded_text
from ..runtime_wiring import GAP_UNCONFIGURED, TrustedHostRuntime, attach_trusted_host_runtime
from ..shared_entries import attachment_path, read_attachment, read_shared_payload, shared_entry_manifest
from ..tool_common import FENCED_RECALL, fence_epoch
from .bounded_call import BoundedCall
from .config import REMOTE_SERVER_CONFIG_FILENAME, CodexConfigError, shared_clients
from .hook_answer import HookDiagnostics, error_detail, recall_incomplete, recall_without_vectors

_UNATTESTED_HUMAN_PLATFORMS = frozenset({"a2a", "webhook", "msgraph_webhook"})
_ALLOWED_CONTEXTS = frozenset({"primary", "subagent", "cron", "flush"})
_RUNTIME_ATTACH_MIN_S = 0.2
_SCOPE_FIELDS = (
    "platform",
    "user_id",
    "chat_type",
    "chat_id",
    "thread_id",
    "agent_identity",
    "agent_workspace",
    "agent_context",
    "gateway_session_key",
)


@dataclass(frozen=True)
class HermesResidentTarget:
    """Verified local Hermes entry attached to a shared store."""

    home: Path
    manifest: InstallationManifest
    host: str = "hermes"

    @property
    def installation_id(self) -> str:
        return self.manifest.installation_id

    @property
    def agent_id(self) -> str:
        return self.manifest.agent_id

    @property
    def data_directory(self) -> Path:
        return self.manifest.data_directory

    @property
    def scope_ids(self) -> frozenset[str]:
        return self.manifest.scope_ids

    @property
    def test_mode(self) -> bool:
        return self.manifest.test_mode

    @property
    def entry_id(self) -> str:
        return str(self.manifest.entry_id or "")

    @property
    def entry_name(self) -> str:
        return self.manifest.entry_name

    @property
    def runtime_config_path(self) -> Path:
        """The entry's own runtime config beside its pointer; the store's belongs to the worker."""
        return attachment_path(self.home).parent / RUNTIME_CONFIG_FILENAME

    def to_binding(self) -> InstanceBinding:
        return self.manifest.to_binding()


def load_hermes_resident_target(home: Path | str, *, expected_store: Path | None = None) -> HermesResidentTarget:
    """Bind a local Hermes shared-store entry; remote-forwarded and cross-account entries are refused."""
    path = Path(home).expanduser()
    if not path.is_absolute():
        raise CodexConfigError("home must be absolute")
    if expected_store is None:
        path = path.resolve()
    try:
        attachment = read_attachment(path, local_only=expected_store is not None)
        if attachment is None:
            raise CodexConfigError("home is not attached to a shared store")
        if expected_store is not None and not same_local_path(attachment.root, expected_store):
            raise CodexConfigError("home points to another shared store")
        if attachment.host != "hermes":
            raise CodexConfigError(f"home is attached as a {attachment.host} entry, not hermes")
        remote_cfg = attachment_path(path).with_name(REMOTE_SERVER_CONFIG_FILENAME)
        has_remote = (
            local_attachment_stat(remote_cfg) is not None if expected_store is not None else remote_cfg.is_file()
        )
        if has_remote:
            raise CodexConfigError("remote server entries do not use local resident recall")
        manifest = shared_entry_manifest(expected_store or attachment.root, attachment.entry_id, hermes_home=path)
    except (HermesIdentityError, UnverifiedAttachmentPath) as exc:
        raise CodexConfigError(str(exc)) from exc
    if manifest.entry_host != "hermes" or not manifest.scope_ids:
        raise CodexConfigError("home is not attached as a valid Hermes entry")
    if account_pin(manifest.local_account_id) is not None and not local_account_matches(manifest.local_account_id):
        raise CodexConfigError("entry is pinned to another local OS account")
    if not (manifest.data_directory / "memory.sqlite3").is_file():
        raise CodexConfigError("verified core database is required")
    return HermesResidentTarget(home=path, manifest=manifest)


def shared_resident_entries(binding: InstanceBinding):
    """Yield hook clients first, followed by local Hermes entries attached to the same shared store."""
    yield from shared_clients(binding)
    for entry in read_shared_payload(binding.data_directory)["entries"]:
        if entry.get("host") != "hermes":
            continue
        try:
            target = load_hermes_resident_target(Path(entry["home"]), expected_store=binding.data_directory)
            if (
                target.installation_id == binding.installation_id
                and target.agent_id == binding.agent_id
                and target.data_directory == binding.data_directory
                and target.test_mode == binding.test_mode
                and target.scope_ids <= binding.scope_ids
            ):
                yield target, entry["attached_at"]
        except (ContractError, OSError, RuntimeError, ValueError):
            continue


def _parse_zone(raw: object) -> tzinfo | None:
    if raw is None:
        return None
    if type(raw) is not str or not 1 <= len(raw) <= 120:
        raise ValueError("zone")
    try:
        return ZoneInfo(raw)
    except ZoneInfoNotFoundError:
        return None


def _parse_scope(raw: object, *, manifest: InstallationManifest) -> HermesRuntimeScope:
    if type(raw) is not dict:
        raise ValueError("scope")
    values: dict[str, str] = {}
    for name in _SCOPE_FIELDS:
        item = raw.get(name, "")
        if type(item) is not str or len(item) > 240:
            raise ValueError(name)
        values[name] = item
    expected_identity = manifest.agent_identity or manifest.agent_id
    if (
        not values["platform"]
        or not values["user_id"]
        or not values["agent_workspace"]
        or values["agent_identity"] != expected_identity
        or values["agent_context"] not in _ALLOWED_CONTEXTS
    ):
        raise ValueError("scope_invalid")
    if (
        values["platform"] in LOCAL_PLATFORMS
        and values["user_id"] == LOCAL_USER_ID
        and values["platform"] not in approved_local_platforms(manifest.owner_principals)
    ):
        raise ValueError("local_surface_unapproved")
    return HermesRuntimeScope(**values, entry_id=str(manifest.entry_id or ""))


def _source_principal(
    manifest: InstallationManifest,
    scope: HermesRuntimeScope,
    origin: Origin,
    *,
    user_name: str,
    local_account_principal: str | None,
) -> TrustedSourcePrincipal:
    if origin == "human_direct":
        if scope.platform in {"cli", *LOCAL_PLATFORMS} and (
            local_account_principal is not None or not local_account_matches(manifest.local_account_id)
        ):
            return TrustedSourcePrincipal("human", "unresolved")
        if scope.platform in _UNATTESTED_HUMAN_PLATFORMS:
            return TrustedSourcePrincipal("human", "unresolved")
        return TrustedSourcePrincipal(
            "human",
            "verified",
            principal_ref("human", manifest.installation_id, scope.platform, scope.user_id),
            display_name=user_name or None,
        )
    kinds = {
        "tool_observation": "tool",
        "external_document": "document",
        "origin_unknown": "unknown",
        "memory_reinjection": "unknown",
        "imported": "unknown",
    }
    return TrustedSourcePrincipal(cast(PrincipalKind, kinds.get(origin, "unknown")), "unresolved")


def _build_context(
    manifest: InstallationManifest,
    scope: HermesRuntimeScope,
    allowed_scope_ids: frozenset[str],
    payload: dict[str, Any],
) -> TrustedContext:
    session_id = bounded_text(payload.get("session_id"), field="session_id", error=ValueError)
    stored_session = session_identity(session_id, entry_id=manifest.entry_id)
    if len(stored_session) > 240:
        raise ValueError("session_id")
    user_name = payload.get("user_name", "")
    if type(user_name) is not str or len(user_name) > 240:
        raise ValueError("user_name")
    local_principal = payload.get("local_account_principal")
    if local_principal is not None and (type(local_principal) is not str or len(local_principal) > 240):
        raise ValueError("local_account_principal")
    raw_recent = payload.get("recent_messages") or []
    if (
        type(raw_recent) is not list
        or len(raw_recent) > 8
        or any(type(m) is not str or len(m) > 16384 for m in raw_recent)
    ):
        raise ValueError("recent_messages")
    origin: Origin = "origin_unknown" if scope.platform in _UNATTESTED_HUMAN_PLATFORMS else "human_direct"
    try:
        workspace_root = normalize_project_path(scope.agent_workspace)
    except ContractError:
        workspace_root = None
    binding = manifest.to_binding()
    context = TrustedContext(
        binding,
        stored_session,
        allowed_scope_ids,
        origin,
        task_anchor=context_ref(
            "hermes-task",
            manifest.installation_id,
            scope.platform,
            scope.chat_type,
            scope.chat_id,
            scope.thread_id,
            scope.agent_workspace,
            stored_session,
        ),
        recent_messages=tuple(raw_recent),
        source_principal=_source_principal(
            manifest, scope, origin, user_name=user_name, local_account_principal=local_principal
        ),
        entry_id=manifest.entry_id,
        project_root=workspace_root,
        owner_resolver=partial(entry_owner, binding),
        subject_resolver=partial(owner_subjects, binding, manifest.entry_id or ""),
    )
    compacted_at = payload.get("compacted_at")
    compaction_seq = payload.get("compaction_seq")
    compaction_epoch = payload.get("compaction_epoch")
    if compacted_at is not None:
        compacted_at = utc_timestamp(compacted_at)
    if (
        compacted_at is not None
        and type(compaction_seq) is int
        and 1 <= compaction_seq < 2**63
        and type(compaction_epoch) is str
        and 1 <= len(compaction_epoch) <= 64
    ):
        context = replace(
            context,
            compacted_at=compacted_at,
            compaction_seq=compaction_seq,
            compaction_epoch=compaction_epoch,
        )
    return context


class HermesResidentHandler:
    """Read-only recall handler kept inside the resident process for a Hermes entry."""

    def __init__(self, target: HermesResidentTarget) -> None:
        self.target = target
        self.host = "hermes"
        self.core = MemoryCore(CoreConfig(target.to_binding()))
        self._host_runtime: TrustedHostRuntime | None = None
        self._attaching: BoundedCall | None = None
        self._runtime_attach_attempted = False
        self.diagnostics = HookDiagnostics(capability_gaps=(GAP_UNCONFIGURED,))

    @classmethod
    def from_home(cls, home: str | Path) -> HermesResidentHandler:
        return cls(load_hermes_resident_target(home))

    @property
    def runtime_ready(self) -> bool:
        return self._host_runtime is not None and bool(getattr(self._host_runtime, "configured", False))

    def _remaining(self, deadline: float) -> float:
        return max(0.0, deadline - self.core.clock.monotonic())

    def ensure_runtime(self, *, deadline: float | None = None) -> None:
        if self._host_runtime is not None:
            return
        attaching = self._attaching
        if attaching is None:
            if self._runtime_attach_attempted:
                return
            if deadline is not None and self._remaining(deadline) < _RUNTIME_ATTACH_MIN_S:
                return
            self._runtime_attach_attempted = True
            attaching = self._attaching = BoundedCall(
                self._attach,
                name="scope-recall-hermes-runtime-attach",
                remaining_seconds=None if deadline is None else self._remaining(deadline),
            )
        if not attaching.wait(None if deadline is None else self._remaining(deadline)):
            self.diagnostics.last_reason = "runtime_attach_late"
            return
        self._attaching = None
        host_runtime, self.diagnostics.runtime_attach_ms = attaching.result()
        if host_runtime is None:
            self.diagnostics.last_reason = "runtime_attach_failed"
            self.diagnostics.capability_gaps = ("capability_gap:trusted_runtime_invalid",)
            return
        self._host_runtime = host_runtime
        self.core = host_runtime.core
        self.diagnostics.capability_gaps = host_runtime.capability_gaps

    def _attach(self) -> tuple[TrustedHostRuntime | None, int]:
        started = time.monotonic()
        try:
            host_runtime = attach_trusted_host_runtime(
                config_path=self.target.runtime_config_path,
                expected_binding=self.target.to_binding(),
                session_id=f"hermes-resident:{self.target.installation_id}:{self.target.entry_id}",
                allowed_scope_ids=self.target.scope_ids,
                host_adapter="hermes",
                core=self.core,
            )
        except Exception:  # noqa: BLE001 - reported as a failed attach; callers recall directly
            host_runtime = None
        return host_runtime, round((time.monotonic() - started) * 1000)

    def warm_vectors(self, seconds: float) -> None:
        self.ensure_runtime()
        warm = getattr(getattr(self._host_runtime, "_runtime", None), "warm_vector_store", None)
        if callable(warm):
            warm(seconds)

    def warm_embedding(self, seconds: float) -> None:
        context = TrustedContext(
            self.target.to_binding(),
            "warm-embedding",
            self.target.scope_ids,
            "host_generated",
            entry_id=self.target.entry_id,
        )
        if outbound_blocked(self.core.storage, self.core.clock, context):
            return
        self.ensure_runtime()
        warm = getattr(getattr(self._host_runtime, "_runtime", None), "warm_query_embedding", None)
        if callable(warm):
            warm(seconds)

    def close(self) -> None:
        attaching, self._attaching = self._attaching, None
        if attaching is not None:
            attaching.abandon(lambda made: made[0].close() if made[0] is not None else None)
        if self._host_runtime is not None:
            runtime, self._host_runtime = self._host_runtime, None
            runtime.close(detach_worker=True)

    def resident_recall_for(
        self, payload: dict[str, Any], current_refs: tuple[str, ...], gaps: tuple[str, ...], remaining: float
    ) -> dict[str, Any]:
        """Execute one read-only recall after re-checking the entry manifest and exact audience."""
        del gaps
        self.diagnostics = HookDiagnostics(
            capability_gaps=self._host_runtime.capability_gaps
            if self._host_runtime is not None
            else (GAP_UNCONFIGURED,)
        )
        deadline = self.core.clock.monotonic() + max(0.0, remaining)
        manifest = load_hermes_resident_target(self.target.home).manifest
        if manifest.to_binding() != self.target.to_binding():
            raise ValueError("entry_binding_changed")
        scope = _parse_scope(payload.get("scope"), manifest=manifest)
        audience = resolve_runtime_audience(manifest, scope)
        if not audience.allowed_scope_ids:
            self.diagnostics.last_reason = "no_audience"
            self.diagnostics.capability_gaps = audience.capability_gaps
            return {}
        request_raw = payload.get("request")
        if type(request_raw) is not dict:
            raise ValueError("request")
        prepare_render = payload.get("prepare_render") is True
        background = payload.get("background_without_evidence") is not False
        zone = _parse_zone(payload.get("zone"))
        raw_entries = payload.get("host_memory_entries") or []
        if type(raw_entries) is not list or len(raw_entries) > 256 or any(type(e) is not str for e in raw_entries):
            raise ValueError("host_memory_entries")
        self.ensure_runtime(deadline=deadline)
        context = _build_context(manifest, scope, audience.allowed_scope_ids, payload)
        validate_model_request("recall_request", request_raw, context)
        left = self._remaining(deadline)
        if left <= 0:
            self.diagnostics.last_reason = "deadline_exceeded"
            return {}
        options: dict[str, Any] = {
            "current_source_refs": current_refs,
            "deadline_seconds": left,
            "zone": zone,
            "background_without_evidence": background,
        }
        if raw_entries:
            options["host_memory_entries"] = frozenset(raw_entries)
        trace: dict[str, Any] = {}
        self.diagnostics.recall_trace = trace
        try:
            with using_vector_trace(trace):
                result = self._execute(context, cast(RecallRequest, request_raw), options, deadline, prepare_render)
        except (ContractError, OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            code = getattr(exc, "code", None)
            self.diagnostics.recall_error_detail = error_detail(
                f"{type(exc).__name__}:{code}" if isinstance(code, str) else type(exc).__name__
            )
            self.diagnostics.last_reason = "recall_exception"
            return {}
        packet = result["packet"]
        without = recall_without_vectors(packet.get("gaps") or ())
        self.diagnostics.recall_vectors = without is None
        if without is not None:
            self.diagnostics.recall_vector_gap = error_detail(without)
        incomplete = recall_incomplete(packet)
        if incomplete is not None:
            self.diagnostics.recall_vectors = False
            self.diagnostics.recall_error_detail = error_detail(incomplete)
            self.diagnostics.last_reason = "recall_incomplete"
        trace["vector_packet"] = int(bool(trace.get("vector_finished")) and incomplete is None)
        return result

    def _execute(
        self,
        context: TrustedContext,
        request: RecallRequest,
        options: dict[str, Any],
        deadline: float,
        prepare_render: bool,
    ) -> dict[str, Any]:
        packet = self.core.recall_packet(context, request, **options)
        if not prepare_render:
            original = packet
            packet = fence_epoch(
                dict(packet),
                self.core.memory_epoch(context),
                FENCED_RECALL,
                retracted=lambda since: self.core.memory_retracted_since(context, since),
            )
            packet = self.core.release_recall_packet(context, packet, original=original)
            return {"packet": packet}
        preparation = self.core.prepare_recall_render(context, packet)
        if packet.get("items") and not self.core.recall_packet_current(context, packet):
            left = self._remaining(deadline)
            if left <= 0:
                return {"packet": packet, "stale": True}
            options["deadline_seconds"] = left
            packet = self.core.recall_packet(context, request, **options)
            preparation = self.core.prepare_recall_render(context, packet)
        if packet.get("items") and not self.core.recall_packet_current(context, packet):
            return {"packet": packet, "stale": True}
        return {
            "packet": packet,
            "render_ref": preparation.render_ref,
            "canonical_text": preparation.canonical_text,
            "context": preparation.context,
        }
