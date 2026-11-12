"""Synthetic syntax and reviewed-fixture gates for the Hermes source inventory."""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import hermes_surface as surface  # noqa: E402

BASELINE = ROOT / "fixtures/hermes/surface-0.23.0.json"


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    write(root, "adapters/hermes/boundary.py", 'HEAD = "[host]"\nHEADS = (HEAD, "[other]")\n')
    write(root, "adapters/hermes/hooks.py", '_OBSERVERS = dict(HOSTS["hermes"].hook_observers)\n')
    write(root, "adapters/host_profiles.py", 'HOSTS = {"hermes": HostProfile(hook_observers=(("pre", "observe"),))}\n')
    write(root, "adapters/hermes/provider.py", "class Provider:\n    def sync_turn(self): pass\n")
    write(root, "adapters/hermes/tool_surface.py", "class Tools:\n    def handle_tool_call(self): pass\n")
    write(root, "adapters/hermes/protocol.py", "class Protocol:\n    def initialize(self): pass\n")
    return root


@pytest.fixture
def host(tmp_path):
    root = tmp_path / "host"
    root.mkdir()
    return root


def take(host, repo, text):
    write(host, "main.py", text)
    return surface.scan(host, repo)


def reviewed(value):
    for row in value["user_rows"]:
        if row["display_kind"] is None:
            row["disposition"] = {"status": "unknown", "reason": "Synthetic fixture; no runtime provenance."}
    return value


def dependencies(host):
    """Keep unrelated dependency checks healthy while a test changes one surface."""
    write(host, "support.py", 'MARKERS = ["[host]", "[other]"]\ninvoke_hook("pre")\n')


def test_constructs_and_counterexamples(host, repo):
    result = take(
        host,
        repo,
        """
HEAD = "[host]"
ALIAS = HEAD + " notice"
LOOP = LOOP
ROLE = "user"
# {"role": "user", "content": "comment"}
def outer():
    def inner():
        return {"role": "user", "content": ALIAS}
    return dict(role=ROLE, content=f"[prefix] {payload}")
class Owner:
    async def method(self):
        append_message(role="user", content="start " + dynamic, display_kind="hidden")
        mirror_to_session("chat", "id", "mirror text", role="user")
        return {"role": "user", "content": LOOP}
noise = [{"role": "assistant", "content": "no"}, {"role": dynamic, "content": "no"},
         {"role": "us" + suffix, "content": "no"}]
""",
    )
    rows = result["user_rows"]
    assert len(rows) == 5
    assert {row["function"] for row in rows} == {"outer", "outer.inner", "Owner.method"}
    assert {row["content"]["prefix"] for row in rows} == {"[host] notice", "[prefix] ", "start ", "mirror text", ""}
    assert sum(row["display_kind"] is not None for row in rows) == 1


def test_identical_sites_remain_distinct_and_line_movement_is_ignored(host, repo):
    dependencies(host)
    source = 'def f():\n    a = dict(role="user", content="same")\n    b = dict(role="user", content="same")\n'
    first = reviewed(take(host, repo, source))
    second = take(host, repo, "\n\n" + source)
    assert len({row["id"] for row in first["user_rows"]}) == 2
    assert surface.diff(first, second) == ""
    assert surface.encode(take(host, repo, source)) == surface.encode(take(host, repo, source))


def test_markers_are_literal_substrings_and_locations_have_lexical_names(host, repo):
    result = take(host, repo, '# [other]\ndef f():\n    return f"[host] {value}"\n')
    markers = {m["name"]: m for m in result["markers"]}
    assert markers["HEAD"]["locations"][0]["function"] == "f"
    assert markers["HEADS[1]"]["locations"] == []


def test_parameters_spreads_and_events(host, repo):
    result = take(
        host,
        repo,
        """
def turn():
    kwargs = {"session_id": sid}
    kwargs["messages"] = messages
    invoke_hook("pre", user_message=text, **kwargs)
    memory_provider.sync_turn(text, session_id=sid, **kwargs)
    MessageEvent(text=text, source=source)
""",
    )
    assert result["interfaces"] == {
        "hooks": ["pre"],
        "provider_methods": ["handle_tool_call", "initialize", "sync_turn"],
    }
    assert len(result["calls"]) == 2
    hook = next(c for c in result["calls"] if c["kind"] == "hook")
    assert hook["keywords"] == ["user_message"]
    assert hook["spreads"] == [{"expression": "kwargs", "possible_keys": ["messages", "session_id"]}]
    assert result["message_events"][0]["keywords"] == ["source", "text"]


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ('x = {"role": "user", "content": "one"}', 'x = {"role": "user", "content": "two"}', "USER CHANGED"),
        (
            'x = dict(role="user", content="one")',
            'x = dict(role="user", content="one", display_kind="hidden")',
            "USER CHANGED",
        ),
        ("x = 1", 'x = dict(role="user", content="new")', "USER ADDED (no disposition)"),
        ('x = dict(role="user", content="old")', "x = 1", "USER REMOVED"),
        ('x = "[host]"', "x = 1", "MARKER MISSING"),
        ('x = "[host]"', 'def elsewhere():\n    return "[host]"', "MARKER LOCATIONS CHANGED"),
        ("memory.sync_turn(text, session_id=sid)", "memory.sync_turn(text, messages=history)", "CALLS"),
        ("MessageEvent(text=text)", "MessageEvent(text=text, source=source)", "MESSAGE_EVENTS"),
    ],
)
def test_diff_categories(host, repo, before, after, expected):
    old = reviewed(take(host, repo, before))
    output = surface.diff(old, take(host, repo, after))
    assert expected in output
    assert surface.diff(old, take(host, repo, after)) == output


def test_diff_ignores_added_comments_and_does_not_pair_ambiguous_edits(host, repo):
    old = reviewed(take(host, repo, 'a = dict(role="user", content="a")\nb = dict(role="user", content="b")\n'))
    new = take(host, repo, 'a = dict(role="user", content="c")\nb = dict(role="user", content="d")\n')
    output = surface.diff(old, new)
    assert output.count("USER REMOVED") == 2
    assert output.count("USER ADDED") == 2
    assert "USER CHANGED" not in output


def test_changed_boundary_or_registration_invalidates_recorded_definitions(host, repo):
    old = take(host, repo, 'x = "[host]"')
    write(repo, "adapters/hermes/boundary.py", 'HEAD = "[replacement]"\n')
    write(
        repo, "adapters/host_profiles.py", 'HOSTS = {"hermes": HostProfile(hook_observers=(("other", "observe"),))}\n'
    )
    output = surface.diff(old, take(host, repo, 'x = "[host]"'))
    assert "INVENTORY STALE: boundary marker definitions" in output
    assert "INVENTORY STALE: registered hooks" in output
    assert "Refresh the inventory first:" in output
    assert "python scripts/hermes_surface.py refresh" in output


def test_unreviewed_rows_input_errors_and_no_source_execution(host, repo, monkeypatch, capsys):
    dependencies(host)
    source = 'raise RuntimeError("never execute")\nx = dict(role="user", content="text")\n'
    result = take(host, repo, source)
    assert "USER UNREVIEWED" in surface.diff(result, result)
    monkeypatch.setattr(surface, "ROOT", repo)
    output = host / "recorded.json"
    assert surface.main(["scan", str(host), "--output", str(output)]) == 0
    loaded = surface.load(output)
    reviewed(loaded)
    output.write_text(surface.encode(loaded), encoding="utf-8")
    assert surface.main(["diff", str(output), str(host)]) == 0
    write(host, "main.py", source + 'y = dict(role="user", content="extra")\n')
    assert surface.main(["diff", str(output), str(host)]) == 0
    assert surface.main(["diff", str(output), str(host), "--strict"]) == 1
    assert surface.main(["scan", str(host / "missing")]) == 2
    assert "does not exist" in capsys.readouterr().err
    write(host, "main.py", "this is invalid Python !")
    with pytest.raises(SyntaxError):
        surface.scan(host, repo)


def test_environment_and_test_sources_are_not_scanned(host, repo):
    write(host, "main.py", "pass")
    for folder in ("tests", "evals", ".git", "venv", "node_modules"):
        write(host, f"{folder}/bad.py", "invalid !")
    assert surface.scan(host, repo)["user_rows"] == []


def test_recorded_rows_are_reviewed_and_marker_definitions_are_current():
    recorded = surface.load(BASELINE)
    for row in recorded["user_rows"]:
        if row["display_kind"] is None:
            assert row.get("disposition", {}).get("reason"), f"Review {row['id']} and record its disposition"
            assert row["disposition"]["status"] in {"recognized", "outside_history", "unknown", "unverified"}
    expected = {m["name"]: m["text"] for m in recorded["markers"]}
    assert expected == surface.marker_definitions(), (
        "Boundary markers changed. Run python scripts/hermes_surface.py refresh <old-inventory> <Hermes source> "
        "--output <new-inventory>, then review the diff and fill the listed dispositions."
    )
    assert recorded["interfaces"] == surface.interface_definitions(), "Adapter interfaces changed; rescan and review."
    assert len(recorded["user_rows"]) == 110
    dispositions = [row["disposition"] for row in recorded["user_rows"] if row["display_kind"] is None]
    assert len(dispositions) == 99
    assert Counter(item["status"] for item in dispositions) == {
        "outside_history": 51,
        "unverified": 34,
        "recognized": 10,
        "unknown": 4,
    }


def test_tool_and_inventory_do_not_ship():
    allowlist = json.loads((ROOT / "packaging/v11-module-allowlist.json").read_text(encoding="utf-8"))
    assert "scripts/hermes_surface.py" not in allowlist["python_modules"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "exclude scripts/hermes_surface.py" in manifest
    assert "prune fixtures/hermes" in manifest
    tree = ast.parse((ROOT / "scripts/hermes_surface.py").read_text(encoding="utf-8"))
    imports = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imports.update(n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
    assert imports <= sys.stdlib_module_names


def test_positional_database_role_and_lazy_imported_module_constant(host, repo):
    write(host, "prompts.py", 'NOTICE = "[host] imported"\n')
    result = take(
        host,
        repo,
        """
from prompts import NOTICE as NOTE
class Store:
    def write(self):
        db.append_message(session, "user", NOTE, display_kind="hidden")
        db.append_message(session, dynamic_role, "ignored")
        return {"role": "user", "content": NOTE}
""",
    )
    assert len(result["user_rows"]) == 2
    assert all(row["content"]["prefix"] == "[host] imported" for row in result["user_rows"])
    assert sum(row["display_kind"] is not None for row in result["user_rows"]) == 1


def test_long_constant_changes_are_visible_beyond_display_prefix(host, repo):
    old = reviewed(take(host, repo, 'x = dict(role="user", content=' + repr("a" * 400 + "old") + ")"))
    new = take(host, repo, 'x = dict(role="user", content=' + repr("a" * 400 + "new") + ")")
    assert old["user_rows"][0]["content"]["prefix"] == new["user_rows"][0]["content"]["prefix"]
    assert "USER CHANGED" in surface.diff(old, new)


def test_inventory_loader_rejects_duplicate_ids_and_incompatible_format(tmp_path):
    data = surface.load(BASELINE)
    data["user_rows"].append(data["user_rows"][0])
    path = tmp_path / "bad.json"
    path.write_text(surface.encode(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate user_rows"):
        surface.load(path)
    path.write_text('{"format": -1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported inventory format"):
        surface.load(path)


def test_spread_helper_fields_and_scope_isolation(host, repo):
    result = take(
        host,
        repo,
        """
class CallIds:
    session_id: str
    conversation_id: str
    def keywords(self):
        return {k: v or '' for k, v in asdict(self).items()}
def first():
    kwargs = {'first': value}
    invoke_hook('pre', **kwargs, **CallIds().keywords())
def second():
    kwargs = {'second': value}
    invoke_hook('pre', **kwargs)
""",
    )
    first = next(c for c in result["calls"] if c["function"] == "first")
    second = next(c for c in result["calls"] if c["function"] == "second")
    assert first["spreads"][0]["possible_keys"] == ["first"]
    assert first["spreads"][1]["possible_keys"] == ["conversation_id", "session_id"]
    assert second["spreads"][0]["possible_keys"] == ["second"]


def test_identical_new_calls_are_not_hidden_and_keyword_deltas_are_explicit(host, repo):
    old = take(host, repo, "memory.sync_turn(text, old=value)")
    new = take(host, repo, "memory.sync_turn(text, new=value)")
    output = surface.diff(old, new)
    assert "keywords +['new'] -['old']" in output
    new = take(host, repo, "memory.sync_turn(text, old=value)\nmemory.sync_turn(text, old=value)")
    assert "CALLS ADDED" in surface.diff(old, new)


def test_identifier_markers_match_complete_literals(host, repo):
    write(repo, "adapters/hermes/boundary.py", 'KIND = "steer"\n')
    result = take(host, repo, 'text = "documentation about steers"\n')
    assert result["markers"][0]["locations"] == []
    result = take(host, repo, 'text = "steer"\n')
    assert result["markers"][0]["locations"]


def test_direct_hook_registration_is_discovered_and_unknown_loops_fail_closed(repo):
    path = repo / "adapters/hermes/hooks.py"
    original = path.read_text(encoding="utf-8")
    path.write_text(original + "ctx.register_hook('extra', callback)\n", encoding="utf-8")
    assert surface.interface_definitions(repo)["hooks"] == ["extra", "pre"]
    path.write_text(original + "_SUPPORTED_HOOKS = dynamically_selected()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="supported hook shape changed"):
        surface.interface_definitions(repo)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (
            'anchors = ["[host]", "[other]"]\ninvoke_hook("pre")',
            'anchors = ["[other]"]\ninvoke_hook("pre")',
            "MARKER MISSING",
        ),
        ('anchors = ["[host]", "[other]"]\ninvoke_hook("pre")', 'anchors = ["[host]", "[other]"]', "HOOK MISSING"),
        ('invoke_hook("pre", removed=value)', 'invoke_hook("pre")', "KEYWORDS MISSING: hook pre"),
        (
            "memory.sync_turn(text, removed=value)",
            "memory.sync_turn(text)",
            "KEYWORDS MISSING: provider_candidate sync_turn",
        ),
        ('invoke_hook("pre", **{"removed": value})', 'invoke_hook("pre")', "KEYWORDS MISSING: hook pre"),
    ],
)
def test_required_dependency_losses(host, repo, before, after, expected):
    if "anchors" not in before:
        dependencies(host)
    old = take(host, repo, before)
    groups = surface.difference_groups(old, take(host, repo, after))
    assert any(expected in line for line in groups["required"])
    output = surface.render(groups)
    assert output.startswith(f"Action required ({len(groups['required'])}):\n")
    assert f"For reference ({len(groups['reference'])}):" in output


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ('x = dict(role="user", content="old")', 'x = dict(role="user", content="new")', "USER CHANGED"),
        (
            'x = dict(role="user", content="same")',
            'x = dict(role="user", content="same", display_kind="hidden")',
            "USER CHANGED",
        ),
        ("pass", 'x = dict(role="user", content="new")', "USER ADDED"),
        ('x = dict(role="user", content="old")', "pass", "USER REMOVED"),
        ('marker = "[host]"', 'def moved():\n    return "[host]"', "MARKER LOCATIONS CHANGED"),
        ('invoke_hook("pre")', 'invoke_hook("pre", extra=value)', "CALLS PARAMETERS CHANGED"),
        ("memory.sync_turn(text)", "memory.sync_turn(text, extra=value)", "CALLS PARAMETERS CHANGED"),
        ("MessageEvent(text=text)", "MessageEvent(text=text, source=source)", "MESSAGE_EVENTS PARAMETERS CHANGED"),
    ],
)
def test_reference_only_changes(host, repo, before, after, expected):
    dependencies(host)
    old = reviewed(take(host, repo, before))
    groups = surface.difference_groups(old, take(host, repo, after))
    assert groups["required"] == []
    assert any(expected in line for line in groups["reference"])
    assert surface.render(groups).startswith("Action required (0):\n")


@pytest.mark.parametrize("kind", ["hook", "provider"])
def test_keywords_are_unioned_across_sites_and_spreads(host, repo, kind):
    dependencies(host)
    call = 'invoke_hook("pre"' if kind == "hook" else "memory.sync_turn(text"
    old = take(host, repo, call + ", retained=value)\n" + call + ', **{"retained": other})')
    new = take(host, repo, call + ")\n" + call + ', **{"retained": other})')
    assert surface.difference_groups(old, new)["required"] == []
    new = take(host, repo, call + ")\n" + call + ")")
    assert any("retained" in line for line in surface.difference_groups(old, new)["required"])


def test_an_unobserved_hook_is_not_reported_as_a_new_loss(host, repo):
    source = 'anchors = ["[host]", "[other]"]\n'
    absent = take(host, repo, source)
    assert surface.difference_groups(absent, take(host, repo, source))["required"] == []
    present = take(host, repo, source + 'invoke_hook("pre")\n')
    assert surface.difference_groups(absent, present)["required"] == []
    assert any("HOOK MISSING" in line for line in surface.difference_groups(present, absent)["required"])


def test_provider_receivers_and_paths_use_the_memory_heuristic(host, repo):
    write(
        repo,
        "adapters/hermes/provider.py",
        "class Provider:\n    def sync_turn(self): pass\n    def shutdown(self): pass\n",
    )
    write(
        host,
        "main.py",
        """
server.initialize()
server.shutdown()
sock.shutdown()
p.sync_turn(text, invisible=value)
self._MeMoRy.sync_turn(text, visible=value)
""",
    )
    write(host, "Memory/providers.py", "p.sync_turn(text, from_file=value)\n")
    result = surface.scan(host, repo)
    calls = [call for call in result["calls"] if call["kind"] == "provider_candidate"]
    assert {(call["receiver"], call["file"]) for call in calls} == {
        ("self._MeMoRy", "main.py"),
        ("p", "Memory/providers.py"),
    }


def test_cli_levels_strict_stale_remedy_and_no_change(host, repo, tmp_path, monkeypatch, capsys):
    dependencies(host)
    monkeypatch.setattr(surface, "ROOT", repo)
    old = tmp_path / "reviewed.json"
    old.write_text(surface.encode(take(host, repo, "pass")), encoding="utf-8")
    assert surface.main(["diff", str(old), str(host), "--strict"]) == 0
    assert capsys.readouterr().out == "No Hermes surface changes.\n"
    write(host, "main.py", 'x = dict(role="user", content="new")')
    assert surface.main(["diff", str(old), str(host)]) == 0
    assert "For reference (1):" in capsys.readouterr().out
    assert surface.main(["diff", str(old), str(host), "--strict"]) == 1
    write(repo, "adapters/hermes/boundary.py", 'HEAD = "[other]"\n')
    assert surface.main(["diff", str(old), str(host)]) == 1
    output = capsys.readouterr().out
    assert "INVENTORY STALE" in output and "Refresh the inventory first:" in output
    assert str(old).replace("\\", "/") in output
    assert "refresh" in output and "--output" in output
    assert surface.main(["diff", str(old), str(host), "--strict"]) == 1


def test_refresh_carries_unchanged_and_reports_new_changed_and_discarded(host, repo):
    old = reviewed(
        take(
            host,
            repo,
            """
keep = dict(role="user", content="keep")
change = dict(role="user", content="before")
remove = dict(role="user", content="remove")
""",
        )
    )
    old["source"] = {"old": "not reused"}
    current = take(
        host,
        repo,
        """

keep = dict(role="user", content="keep")
change = dict(role="user", content="after")
added = dict(role="user", content="added")
""",
    )
    source = current["source"].copy()
    prior = {row["content"]["prefix"]: row for row in old["user_rows"]}
    output = surface.refresh(old, current)
    rows = {row["content"]["prefix"]: row for row in current["user_rows"]}
    assert rows["keep"]["disposition"] == prior["keep"]["disposition"]
    assert "disposition" not in rows["after"] and "disposition" not in rows["added"]
    assert "Rows without disposition (2):" in output
    assert "Discarded dispositions (2):" in output
    assert "before" in output and "remove" in output
    assert current["source"] == source != old["source"]


@pytest.mark.parametrize("change", ["id", "content", "display_kind"])
def test_refresh_checks_every_identity_component(host, repo, change):
    old = reviewed(take(host, repo, 'x = dict(role="user", content="same")'))
    current = take(host, repo, 'x = dict(role="user", content="same")')
    row = current["user_rows"][0]
    if change == "id":
        row["id"] += ":different"
    else:
        row[change] = {"expression": "different", "prefix": "different", "digest": "different"}
    output = surface.refresh(old, current)
    assert "disposition" not in row
    assert "Rows without disposition (1):" in output and "Discarded dispositions (1):" in output


def test_refresh_is_conservative_when_identical_site_count_changes(host, repo):
    source = 'a = dict(role="user", content="same")\n'
    old = reviewed(take(host, repo, source * 2))
    current = take(host, repo, source * 2)
    assert "Rows without disposition (0):" in surface.refresh(old, current)
    current = take(host, repo, source * 3)
    output = surface.refresh(old, current)
    assert not any("disposition" in row for row in current["user_rows"])
    assert "Rows without disposition (3):" in output and "Discarded dispositions (2):" in output


def test_cli_refresh_migrates_legacy_format_and_rejects_bad_inputs(host, repo, tmp_path, monkeypatch, capsys):
    dependencies(host)
    monkeypatch.setattr(surface, "ROOT", repo)
    old = reviewed(take(host, repo, 'x = dict(role="user", content="same")'))
    old["format"] = surface.FORMAT - 1
    path, output = tmp_path / "old.json", tmp_path / "new.json"
    path.write_text(surface.encode(old), encoding="utf-8")
    assert surface.main(["diff", str(path), str(host)]) == 2
    assert surface.main(["refresh", str(path), str(host), "--output", str(output)]) == 0
    refreshed = surface.load(output)
    assert refreshed["format"] == surface.FORMAT
    assert refreshed["user_rows"][0]["disposition"] == old["user_rows"][0]["disposition"]
    assert "Rows without disposition (0):" in capsys.readouterr().out
    path.write_text("invalid JSON", encoding="utf-8")
    assert surface.main(["refresh", str(path), str(host), "--output", str(output)]) == 2
    assert surface.main(["diff", str(output), str(host / "missing"), "--strict"]) == 2
