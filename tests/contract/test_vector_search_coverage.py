"""Runtime search coverage bounds the exclusion of embedded assistant lexical hits."""

from dataclasses import replace

import pytest
from scope_recall.contracts import InstanceBinding
from scope_recall.runtime.auxiliary import AuxiliaryRuntimeConfig
from scope_recall.runtime.instance import build_runtime_instance
from scope_recall.runtime.instance_config import RuntimeInstanceConfig, VectorRuntimeConfig

from tests.v11_support import recall_request, source_event

QUERY = "What watering schedule did we settle on for the tomato seedlings?"


class PartitionStore:
    """An existing synthetic index whose completed partition requests can consume a fixed clock budget."""

    def __init__(self):
        self.searched = []
        self.after_search = lambda: None

    def open_existing(self):
        pass

    def search(self, vector, *, scope_id, limit):
        self.searched.append(scope_id)
        self.after_search()
        return []

    def close(self):
        pass


class QueryEmbedding:
    """Answer a query locally without contacting a model."""

    def embed_query(self, text, *, remaining_seconds):
        return (1.0, 0.0)


def runtime(tmp_path, *, embedding, scopes=("TEST-a", "TEST-z"), project=False):
    """Bind the production lazy facade to a synthetic SQLite authority and partitioned index."""
    binding = InstanceBinding("TEST-agent", "TEST-installation", tmp_path / "data", frozenset(scopes), True)
    config = RuntimeInstanceConfig(
        binding=binding,
        session_id="TEST-query",
        allowed_scope_ids=binding.scope_ids,
        project_id="TEST-project" if project else None,
        branch_id="TEST-branch" if project else None,
        auxiliary=AuxiliaryRuntimeConfig.from_mapping({"external_embedding": False, "external_consolidation": False}),
        vector=VectorRuntimeConfig(
            backend="sqlite-bruteforce",
            storage_dir=tmp_path / "vectors",
            table_name="TEST-vectors",
            dimensions=2,
            test_injection_override=True,
        ),
    )
    store = PartitionStore()
    instance = build_runtime_instance(config, vector_factory=lambda _: store)
    if embedding:
        instance.auxiliary = replace(instance.auxiliary, query_embedding=QueryEmbedding())
    instance.core.initialize()
    return instance, store


def embedded_reply(instance, scope, *, key, project=False):
    """Capture an assistant original in another session and retain its completed embedding receipt."""
    actor = replace(
        instance.config.context(),
        session_id=f"TEST-capture-{key}",
        actor_origin="assistant_visible",
        project_id="TEST-project" if project else None,
        branch_id="TEST-branch" if project else None,
    )
    saved = instance.core.record_event(
        actor,
        source_event(
            source_event_key=f"TEST-{key}",
            role="assistant",
            origin="assistant_visible",
            content="Tomato seedlings do best on a steady watering schedule. "
            + (
                "Give every pot twenty milliliters before sunrise."
                if key == "unchecked"
                else "Use the orange can to water the tray after sunset."
            ),
        ),
        scope_id=scope,
        remaining_seconds=10,
    )
    ref = saved.event_refs[0]
    with instance.core.storage.write(actor) as tx:
        tx._check(write=True).execute(
            "UPDATE work_items SET state='done' WHERE work_type='embed' AND subject_ref=?", (ref.ref,)
        )
    return ref.ref


def test_disabled_query_embedding_keeps_embedded_assistant_lexical_evidence(tmp_path):
    """A configured index with no query route has not checked any source by vector."""
    instance, store = runtime(tmp_path, embedding=False)
    try:
        ref = embedded_reply(instance, "TEST-z", key="disabled")
        packet = instance.core.recall_packet(instance.config.context(), recall_request(query=QUERY), deadline_seconds=2)
        assert ref in {item["ref"] for item in packet["items"]}, packet["gaps"]
        assert "vector_unavailable" in packet["gaps"]
        assert not any(gap.startswith("vector_error:") for gap in packet["gaps"])
        assert store.searched == []
    finally:
        instance.close()


@pytest.mark.parametrize("project", [False, True], ids=["other-scope", "partial-project-scope"])
def test_partial_runtime_search_keeps_lexical_evidence_in_unchecked_partitions(tmp_path, monkeypatch, project):
    """An unchecked scope stays available; an incomplete project scan conservatively retains its logical scope."""
    scopes = ("TEST-a",) if project else ("TEST-a", "TEST-z")
    instance, store = runtime(tmp_path, embedding=True, scopes=scopes, project=project)
    try:
        unchecked = embedded_reply(instance, scopes[-1], key="unchecked", project=project)
        checked = embedded_reply(instance, scopes[0], key="checked", project=project)
        clock = [100.0]
        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        store.after_search = lambda: clock.__setitem__(0, clock[0] + 1.1)
        packet = instance.core.recall_packet(instance.config.context(), recall_request(query=QUERY), deadline_seconds=2)
        assert len(store.searched) == 1
        refs = {item["ref"] for item in packet["items"]}
        assert unchecked in refs, packet["gaps"]
        if not project:
            assert checked not in refs
        assert "vector_search_incomplete" in packet["gaps"]
    finally:
        instance.close()


def test_complete_empty_runtime_search_can_exclude_embedded_assistant_lexical_hits(tmp_path):
    """An empty result can constrain lexical hits when every allowed partition was actually searched."""
    instance, store = runtime(tmp_path, embedding=True)
    try:
        ref = embedded_reply(instance, "TEST-z", key="complete")
        packet = instance.core.recall_packet(instance.config.context(), recall_request(query=QUERY), deadline_seconds=2)
        assert len(store.searched) == 2
        assert ref not in {item["ref"] for item in packet["items"]}
        assert "vector_unavailable" not in packet["gaps"]
        assert "vector_search_incomplete" not in packet["gaps"]
    finally:
        instance.close()
