"""Physical import latency with concurrent hooks on synthetic evidence only."""

import json
import os
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from queue import Queue
from threading import Event

import pytest
from scope_recall.contracts import ContractError
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.memory_import import IMPORT_BATCH_SIZE, ImportRecord, import_record, prefetch_new_sources
from v11_support import context, recall_request, source_event


def test_host_capture_reports_busy_before_it_has_durable_ingress(tmp_path):
    """A refused inbox write reports a typed failure and never claims durability."""
    ctx = context(tmp_path / "TEST-ingress-busy")
    core = MemoryCore(CoreConfig(ctx.binding))
    core.initialize()
    core.storage.timeout_seconds = 0
    with closing(sqlite3.connect(core.storage.path)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        with pytest.raises(ContractError) as refused:
            core.record_host_event(ctx, source_event(), scope_id="TEST-scope", host_scope=None, remaining_seconds=0.01)
        assert refused.value.code == "STORAGE_UNAVAILABLE"
        assert holder.execute("SELECT count(*) FROM capture_inbox").fetchone()[0] == 0
        holder.rollback()
    receipt = core.record_host_event(ctx, source_event(), scope_id="TEST-scope", host_scope=None)
    assert receipt.durability == "persisted"


@pytest.mark.real_store_time
def test_imports_release_writer_for_capture_and_recall(tmp_path):
    """Each import boundary admits a capture and recall before importing finishes."""
    count = int(os.environ.get("SCOPE_RECALL_IMPORT_BENCHMARK_RECORDS", "200"))
    assert count >= 200
    stride = count // 20
    ctx = context(tmp_path / "TEST-import-fairness")
    core = MemoryCore(CoreConfig(ctx.binding))
    core.initialize()
    core.record_event(ctx, source_event(content="TEST saffron palette"), scope_id="TEST-scope")
    hook = MemoryCore(CoreConfig(ctx.binding))
    hook_ctx = replace(ctx, session_id="TEST-hook")
    milestones, probes = Queue(), []
    finished_probe, stopping = Event(), Event()

    def probe(index):
        started = time.perf_counter()
        receipt = hook.record_host_event(
            hook_ctx,
            source_event(source_event_key=f"TEST-hook/{index}", content=f"TEST capture number {index}"),
            scope_id="TEST-scope",
            host_scope=None,
            remaining_seconds=30.0,
        )
        captured = time.perf_counter()
        packet = hook.recall(hook_ctx, recall_request(query="saffron", mode="current"))
        ended = time.perf_counter()
        assert receipt.durability in {"persisted", "queued"}
        assert packet.items
        return {"capture": captured - started, "recall": ended - captured}

    baseline = [probe(i) for i in range(5)]

    def import_many():
        began = time.perf_counter()
        for index in range(count):
            if index % IMPORT_BATCH_SIZE == 0:
                batch = tuple(
                    ImportRecord(f"TEST-{n}", f"TEST synthetic imported material {n}", "assistant")
                    for n in range(index, min(count, index + IMPORT_BATCH_SIZE))
                )
                hints = prefetch_new_sources(
                    core, ctx, batch, source="mem0", source_id="TEST-large", scope_id="TEST-scope"
                )
            item = batch[index % IMPORT_BATCH_SIZE]
            receipt = import_record(
                core, ctx, item, source="mem0", source_id="TEST-large", scope_id="TEST-scope", _new_sources=hints
            )
            assert receipt.durability == "persisted"
            if index < stride * 20 and index % stride == stride // 10:
                # Pause between records, outside their transactions. A hook must
                # write while the import is unfinished, even on a slow disk.
                finished_probe.clear()
                milestones.put(index)
                assert finished_probe.wait(timeout=120), "capture did not finish between imported records"
                if stopping.is_set():
                    break
        return time.perf_counter() - began

    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(import_many)
        try:
            for index in range(5, 25):
                milestones.get(timeout=120)
                assert not task.done()
                try:
                    probes.append(probe(index))
                except BaseException:
                    stopping.set()
                    raise
                finally:
                    finished_probe.set()
            elapsed = task.result(timeout=1800)
        finally:
            stopping.set()
            finished_probe.set()
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT count(*) FROM source_events WHERE origin='imported'").fetchone()[0] == count
    metrics = {"records": count, "import_seconds": round(elapsed, 4), "probes_during_import": len(probes)}
    for label, samples in (("baseline", baseline), ("during", probes)):
        for operation in ("capture", "recall"):
            values = [sample[operation] for sample in samples]
            metrics[f"{label}_{operation}_median_ms"] = round(statistics.median(values) * 1000, 3)
            metrics[f"{label}_{operation}_max_ms"] = round(max(values) * 1000, 3)
    print("IMPORT_METRICS=" + json.dumps(metrics, sort_keys=True))
