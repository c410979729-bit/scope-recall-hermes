"""Automatic injection needs channel evidence and a prompt that needs memory."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest
from scope_recall.core.auto_recall import automatic_query_topic, automatic_recall_skip_reason
from scope_recall.core.recall_policy import RecallPolicy, numbered_link_conflict, numbered_links, rrf_score
from scope_recall.core.retrieval import CandidateRef, SearchContext

from tests.contract.test_autonomous_context import app, task
from tests.contract.test_recall_admission import _app, _candidate, _capture, recall
from tests.contract.test_recall_dating_and_order import _say
from tests.v11_support import recall_request, source_event


def test_explicit_recall_never_sends_inline_private_text_to_vector_port(tmp_path):
    core, ctx, vectors = _app(tmp_path)
    _capture(core, ctx, "TEST-seed-private-query", "TEST visible topic.")
    recall(core, ctx, query="TEST visible <private>TEST secret sentinel</private> topic", mode="current")
    assert vectors.calls
    assert "TEST secret sentinel" not in vectors.calls[0]["query"]


def test_one_common_identifier_is_below_floor_but_explicit_recall_keeps_it(tmp_path):
    core, ctx, _ = _app(tmp_path)
    source = _capture(core, ctx, "one-word", "2a invoice archive")
    result = core.recall_packet(ctx, recall_request(query="2a telescope lens"))
    assert result["items"] == []
    assert "auto_relevance_below_floor:lexical:1" in result["gaps"]
    explicit = recall(core, ctx, query="2a telescope lens", mode="current")
    assert source.ref in {item.ref for item in explicit.items}
    explicit_auto = core.recall_packet(
        ctx, recall_request(query="2a telescope lens"), background_without_evidence=False
    )
    assert source.ref in {item["ref"] for item in explicit_auto["items"]}
    assert not any(gap.startswith("auto_relevance_below_floor:") for gap in explicit_auto["gaps"])


def test_explicit_auto_lookup_can_search_a_mid_session_continuation(tmp_path):
    core, ctx, vectors = _app(tmp_path)
    source = _capture(core, ctx, "earlier-task", "continue the synthetic poster")
    assert not core.recall_packet(ctx, recall_request(query="continue"))["items"]
    assert not vectors.calls
    explicit = core.recall_packet(ctx, recall_request(query="continue"), background_without_evidence=False)
    assert source.ref in {item["ref"] for item in explicit["items"]}
    assert vectors.calls


def test_strong_channel_and_exact_reference_can_qualify_a_weak_lexical_hit(tmp_path, monkeypatch):
    core, ctx, vectors = _app(tmp_path)
    source = _capture(core, ctx, "weak-seed", "2a invoice archive")
    vectors.candidates = (_candidate(source),)
    assert source.ref in {item.ref for item in recall(core, ctx, query="2a telescope lens").items}
    vectors.candidates = ()
    monkeypatch.setattr(
        core.recall_pipeline.storage_reader, "related", lambda *a, **k: pytest.fail("weak seed expanded")
    )
    assert not recall(core, ctx, query="2a telescope lens").items
    monkeypatch.undo()
    exact = recall(core, ctx, query="2a telescope lens", focus_refs=(f"{source.ref}@1",))
    assert source.ref in {item.ref for item in exact.items}


@pytest.mark.parametrize(
    "query,content",
    [("window", "window pigeon"), ("window bird", "window bird pigeon"), ("我家窗外有什么", "我窗外有只鸽子")],
)
def test_short_questions_retain_matching_terms(tmp_path, query, content):
    core, ctx, _ = _app(tmp_path)
    source = _capture(core, ctx, "short", content)
    result = recall(core, ctx, query=query)
    assert source.ref in {item.ref for item in result.items}


def test_each_lexical_answer_qualifies_independently_of_an_echo(tmp_path):
    core, ctx, _ = _app(tmp_path)
    query = "What is the orchard sprinkler pressure?"
    answer = _capture(core, ctx, "pressure-answer", "The sprinkler pressure is forty psi.")
    assert answer.ref in {item["ref"] for item in core.recall_packet(ctx, recall_request(query=query))["items"]}
    echo = _capture(core, replace(ctx, session_id="TEST-earlier-question"), "pressure-echo", query)
    result = core.recall_packet(ctx, recall_request(query=query))
    assert answer.ref in {item["ref"] for item in result["items"]}
    assert echo.ref not in {item["ref"] for item in result["items"]}


def test_a_weak_long_message_cannot_lift_its_reply_above_a_strong_answer(tmp_path, monkeypatch):
    core, ctx, vectors = _app(tmp_path)
    document = _say(
        core,
        ctx,
        "orchard sprinkler " + "合成文档内容。" * 1000,
        origin="human_direct",
        role="user",
        when="2026-09-01T09:00:00Z",
        key="TEST-manual",
    )
    reply = _say(
        core,
        ctx,
        "First I check where each rule file is.",
        origin="assistant_visible",
        role="assistant",
        when="2026-09-01T09:00:12Z",
        key="TEST-manual-reply",
    )
    answer = _capture(core, replace(ctx, session_id="TEST-pressure-answer"), "strong-answer", "Set it to forty psi.")
    seed = CandidateRef(
        "event", document.ref, 1, "lexical", rank=11, lexical_score=2.0, matched_query_terms=("orchard", "sprinkler")
    )
    monkeypatch.setattr(core.recall_pipeline.storage_reader, "lexical", lambda *a, **k: (seed,))
    leading = [
        _capture(
            core,
            replace(ctx, session_id=f"TEST-leading-{index}"),
            f"leading-{index}",
            f"Synthetic direct evidence {index}.",
        )
        for index in range(4)
    ]
    vectors.candidates = (*(_candidate(source) for source in leading), _candidate(answer, score=0.84))
    reader = replace(ctx, session_id="TEST-pressure-reader")
    result = core.recall_packet(reader, recall_request(query="orchard sprinkler operating pressure", max_items=5))
    refs = {item["ref"] for item in result["items"]}
    assert answer.ref in refs, result
    assert reply.ref not in refs, result


@pytest.mark.parametrize("path", ["ordinary", "echo", "shared_echo"])
def test_turn_reply_score_inherits_only_an_ordinary_seed_bound(tmp_path, monkeypatch, path):
    core, ctx, _ = _app(tmp_path)
    question = _say(
        core,
        ctx,
        "synthetic orchard sprinkler manual",
        origin="human_direct",
        role="user",
        when="2026-09-01T09:00:00Z",
        key="TEST-score-question",
    )
    reply = _say(
        core,
        ctx,
        "The valve is beside the gate.",
        origin="assistant_visible",
        role="assistant",
        when="2026-09-01T09:00:12Z",
        key="TEST-score-reply",
    )
    seed = CandidateRef("event", question.ref, 1, "lexical", rank=11, fusion_score=rrf_score((11,)))
    seeds, echoes = ((seed,), ()) if path == "ordinary" else ((), (seed,))
    if path == "shared_echo":
        seeds = (replace(seed, ref="TEST-ordinary-seed", rank=3, fusion_score=rrf_score((3,))),)
        monkeypatch.setattr(
            core.recall_pipeline.storage_reader,
            "turn_replies",
            lambda *args: (CandidateRef("event", reply.ref, 1, "recent_raw", rank=1),),
        )
    search = SearchContext.from_request(
        recall_request(query="orchard sprinkler operating pressure"), ctx, now=core.clock.utc_now(), deadline=102.0
    )
    with core.storage.read(ctx) as tx:
        expanded = core.recall_pipeline._expand(tx, search, seeds, [], echoes=echoes)
    candidate = next(candidate for candidate in expanded if candidate.ref == reply.ref)
    assert candidate.fusion_score == (seed.fusion_score if path == "ordinary" else rrf_score((2,)))


def test_multi_topic_question_retains_each_lexical_answer(tmp_path):
    core, ctx, _ = _app(tmp_path)
    queue = _capture(core, ctx, "queue", "Onyx 自研的队列换成了托管队列。")
    audit = _capture(core, ctx, "audit", "Onyx 的管理接口开启了审计日志。")
    result = recall(core, ctx, query="替换 Onyx 自研的队列和给 Onyx 管理接口开审计日志，我们先做的是哪个？")
    assert {queue.ref, audit.ref} <= {item.ref for item in result.items}


def test_attachment_terms_do_not_dilute_an_independently_matched_question(tmp_path):
    core, ctx, _ = _app(tmp_path)
    source = _capture(core, ctx, "project-owner", "TEST 白鹭项目的负责人是 KZ-42。")
    attachment = "".join(chr(0x4E00 + index) for index in range(180))
    result = recall(core, ctx, query=f"白鹭项目的负责人 KZ-42 是谁？下面是附件：\n{attachment}")
    assert source.ref in {item.ref for item in result.items}


def test_vector_floor_is_independent_of_lexical_coverage_and_admission(tmp_path):
    core, ctx, vectors = _app(tmp_path)
    source = _capture(core, ctx, "semantic", "quiet aviary")
    vectors.candidates = (_candidate(source, score=0.95),)
    assert source.ref in {item.ref for item in recall(core, ctx, query="violet lantern").items}
    core.recall_pipeline.policy = RecallPolicy(vector_threshold=0.8, auto_vector_threshold=0.9)
    vectors.candidates = (_candidate(source, score=0.85),)
    result = core.recall_packet(ctx, recall_request(query="violet lantern"))
    assert result["items"] == []
    assert "auto_relevance_below_floor:vector:1" in result["gaps"]
    explicit_auto = core.recall_packet(ctx, recall_request(query="violet lantern"), background_without_evidence=False)
    assert source.ref in {item["ref"] for item in explicit_auto["items"]}
    assert source.ref in {item.ref for item in recall(core, ctx, query="violet lantern", mode="current").items}


@pytest.mark.parametrize(
    "query",
    [
        "继续",
        "好的",
        "按你推荐的办",
        "删除了吗",
        "codex在开始跑了吗",
        "太棒了！",
        "Thanks!",
        "go ahead",
        "Is it done?",
        "Great job, continue!",
        "Continue, thanks",
        "继续，谢谢",
        "Thanks, continue",
        "继续，谢谢，进度如何？",
        "可以，按照你说的设置",
        "好，按你建议的来",
        "按你推荐的改",
        "按照你刚才说的弄吧",
        "删除了吗\n",
        "删除了吗\r\n  ",
    ],
)
def test_mid_session_conversation_starts_no_embedding_or_search(tmp_path, monkeypatch, query):
    core, ctx, vectors = _app(tmp_path)
    _capture(core, ctx, "earlier", "Please prepare a synthetic poster.")
    current = _capture(core, ctx, "current", query)
    monkeypatch.setattr(vectors, "prefetch_query", lambda *a, **k: pytest.fail("embedding requested"), raising=False)
    monkeypatch.setattr(core.recall_pipeline.storage_reader, "lexical", lambda *a, **k: pytest.fail("search requested"))
    result = core.recall_packet(ctx, recall_request(query=query), current_source_refs=(f"{current.ref}@1",))
    assert result["items"] == []
    assert any(gap.startswith("auto_recall_skipped:") for gap in result["gaps"])
    assert vectors.calls == []


@pytest.mark.parametrize("query", ["继续", "还有什么没做完的吗？", "这个如何修"])
def test_first_prompt_continues_grounded_task_after_current_prompt_capture(app, query):
    core, ctx = app
    ctx = replace(ctx, task_anchor="TEST-active-task")
    episode = task(core, ctx, "TEST 海报排版还未完成")
    fresh = replace(ctx, session_id="TEST-new-session", task_anchor=None)
    _capture(core, fresh, "notice", "A synthetic assistant notice.", origin="assistant_visible")
    current = _capture(core, fresh, "first", query)
    result = core.recall_packet(fresh, recall_request(query=query), current_source_refs=(f"{current.ref}@1",))
    assert episode.ref in {item["ref"] for item in result["items"]}
    assert not any(gap.startswith("auto_recall_skipped:") for gap in result["gaps"])


def test_first_unfinished_question_requires_one_grounded_task_in_its_project(app):
    core, ctx = app
    query = "还有什么没做完的吗？"
    reader = replace(ctx, session_id="TEST-new-session")
    for project in (None, "TEST-other-project"):
        task(core, replace(ctx, project_id=project, task_anchor="TEST-foreign"), "TEST unrelated translation")
    assert not core.recall_packet(reader, recall_request(query=query))["items"]
    first = task(core, replace(ctx, task_anchor="TEST-layout"), "TEST layout still needs approval")
    packet = core.recall_packet(reader, recall_request(query=query))
    assert [item["ref"] for item in packet["items"]] == [first.ref]
    assert not core.recall_packet(reader, recall_request(query="测试，在吗？"))["items"]
    task(core, replace(ctx, task_anchor="TEST-colour"), "TEST colour still needs approval")
    assert not core.recall_packet(reader, recall_request(query=query))["items"]


def test_exact_task_anchor_survives_a_crowded_project(app):
    core, ctx = app
    target = replace(ctx, task_anchor="TEST-layout")
    episode = task(core, target, "TEST verify the poster layout")
    for index in range(24):
        task(core, replace(ctx, task_anchor=f"TEST-other-{index}"), f"TEST inspect pump {index}")
    reader = replace(target, session_id="TEST-new-session")
    query = recall_request(query="还有什么没做完的吗？")
    assert [item["ref"] for item in core.recall_packet(reader, query)["items"]] == [episode.ref]
    assert not core.recall_packet(replace(reader, task_anchor="TEST-new-anchor"), query)["items"]


def test_two_grounded_segments_for_the_same_anchor_are_ambiguous(app, monkeypatch):
    from scope_recall.core.episode_storage import Episodes

    core, ctx = app
    anchored = replace(ctx, task_anchor="TEST-layout")
    task(core, anchored, "TEST verify the poster dimensions")
    # Advance to the next real segment without capturing 200 unrelated events.
    monkeypatch.setattr(Episodes, "_segment_index", lambda self, series: 1)
    task(core, anchored, "TEST prepare the print proof")
    reader = replace(anchored, session_id="TEST-new-session")
    assert not core.recall_packet(reader, recall_request(query="继续"))["items"]


@pytest.mark.parametrize("prompt", ["还有什么没做完的吗？", "继续"])
def test_first_followup_uses_the_host_directory_when_each_session_has_its_own_task_anchor(app, prompt):
    core, ctx = app
    old = replace(ctx, project_id=None, task_anchor="TEST-old-session-task", project_root="/test/orchard")
    reader = replace(old, session_id="TEST-new-session", task_anchor="TEST-new-session-task")
    episode = task(core, old, "TEST verify the poster dimensions")
    task(core, replace(old, project_root="/test/orchard-two", task_anchor="TEST-other"), "TEST replace pump")
    query = recall_request(query=prompt)
    assert [item["ref"] for item in core.recall_packet(reader, query)["items"]] == [episode.ref]
    assert not core.recall_packet(replace(reader, project_identity="unverified"), query)["items"]
    assert not core.recall_packet(replace(reader, project_identity="remote"), query)["items"]


@pytest.mark.parametrize(
    "query",
    [
        "这个如何修",
        "那怎么修这个问题？",
        "How do we fix this?",
        "还有什么没做完的吗？",
        "测试，在吗？",
        "全部停止！然后升级",
    ],
)
def test_unanchored_prompt_does_not_inject_another_tasks_strong_match(tmp_path, query):
    core, ctx, vectors = _app(tmp_path)
    unrelated = _capture(core, ctx, "past-task", "旧任务：修理水泵以后，升级温室控制器。")
    vectors.candidates = (_candidate(unrelated, score=0.99),)
    for reader in (ctx, replace(ctx, session_id="fresh")):
        packet = core.recall_packet(reader, recall_request(query=query))
        assert not packet["items"], packet["items"]
        assert "auto_recall_skipped:unanchored_followup" in packet["gaps"]
    assert not vectors.calls
    explicit = core.recall_packet(ctx, recall_request(query=query), background_without_evidence=False)
    assert unrelated.ref in {item["ref"] for item in explicit["items"]}


@pytest.mark.parametrize("status_first", [True, False])
def test_mixed_prompt_searches_its_topic_and_keeps_a_cross_session_answer(tmp_path, status_first):
    core, ctx, vectors = _app(tmp_path)
    status, topic = "还有什么没做完的吗？", "白鹭温室的喷头压力是多少？"
    query = status + topic if status_first else topic + status
    wrong = _capture(core, ctx, "unrelated-status", "还有什么没做完的吗？旧画册的插图没做完。")
    echo = _capture(core, ctx, "old-question", query)
    answer = _capture(core, ctx, "greenhouse", "白鹭温室的喷头压力是四十千帕。")
    packet = core.recall_packet(replace(ctx, session_id="new-reader"), recall_request(query=query))
    assert vectors.calls[0]["query"] == "白鹭温室的喷头压力是多少？"
    refs = {item["ref"] for item in packet["items"]}
    assert answer.ref in refs
    assert refs.isdisjoint({wrong.ref, echo.ref})
    vectors.calls.clear()
    core.recall_packet(ctx, recall_request(query=query), background_without_evidence=False)
    assert vectors.calls[0]["query"] == query


@pytest.mark.parametrize(
    "query,topic",
    [
        ("这个如何修复 Atlas？", "这个如何修复 Atlas？"),
        ("怎么修这个问题，别动 core/recall.py？", "怎么修这个问题，别动 core/recall.py？"),
        ("这个怎么修？只看 https://board.example/tickets/17", "这个怎么修？只看 https://board.example/tickets/17"),
        ("白鹭温室用什么系统？然后升级。", "白鹭温室用什么系统？然后升级。"),
        ("上次温室排水是怎么修的？", "上次温室排水是怎么修的？"),
        ("测试环境的端口是多少？", "测试环境的端口是多少？"),
        ("白鹭温室漏水了。怎么修这个问题？", "白鹭温室漏水了。怎么修这个问题？"),
        ("Great.Basin is the project name", "Great.Basin is the project name"),
        ("Thanks. .env.example is missing", ".env.example is missing"),
        ("好的。/opt/orchard 如何安装？", "好的。/opt/orchard 如何安装？"),
        ("OK, see https://board.example/t/17?a=1&b=2#Here", "see https://board.example/t/17?a=1&b=2#Here"),
    ],
)
def test_conversational_prefix_preserves_the_literal_topic(query, topic):
    assert automatic_query_topic(query) == topic
    assert automatic_recall_skip_reason(query) is None


@pytest.mark.parametrize("ending", ["", "!"])
def test_numbered_links_do_not_cross_to_another_record_in_the_same_collection(tmp_path, ending):
    core, ctx, vectors = _app(tmp_path)
    wanted = "https://board.example/orchard/tickets/42017"
    other = "https://board.example/orchard/tickets/42018"
    query = f"查一下这个工单的解决办法 {wanted}{ending}"
    wrong = _capture(core, ctx, "wrong-ticket", f"这个工单的解决办法是换皮带。{other}")
    right = _capture(core, replace(ctx, session_id="right"), "right-ticket", f"这个工单要换轴承。{wanted}?view=full")
    answer = _capture(core, replace(ctx, session_id="notes"), "maintenance", "换轴承后，喷头的压力恢复到四十千帕。")
    vectors.candidates = (_candidate(wrong, score=0.99), _candidate(right, score=0.95), _candidate(answer))
    packet = core.recall_packet(replace(ctx, session_id="reader"), recall_request(query=query))
    assert {right.ref, answer.ref} <= {item["ref"] for item in packet["items"]}
    assert all(other not in item["content"] for item in packet["items"]), packet["items"]
    explicit = core.recall_packet(ctx, recall_request(query=query), background_without_evidence=False)
    assert wrong.ref in {item["ref"] for item in explicit["items"]}


def test_numbered_link_veto_accepts_either_comparison_target_and_supporting_prose():
    first, second = "https://board.example/Atlas/issues/17", "https://board.example/Atlas/issues/18"
    search = numbered_links(f"Compare {first} and {second}")
    assert not numbered_link_conflict(search, second + "#reply")
    assert not numbered_link_conflict(search, "The pump bearing needs replacement.")
    assert not numbered_link_conflict(search, "See the manual at https://docs.example/chapters/42")
    assert not numbered_link_conflict(search, "https://board.example/atlas/issues/17")
    assert not numbered_link_conflict(search, "https://board.example/atlas/issues/19")
    assert numbered_link_conflict(
        numbered_links("https://board.example/Atlas!/issues/17!"), "https://board.example/Atlas!/issues/19"
    )
    assert numbered_link_conflict(search, "https://board.example/Atlas/issues/19的维修记录。")
    assert numbered_link_conflict(search, "https://board.example/Atlas/issues/19/files")


@pytest.mark.parametrize(
    "candidate,conflicts",
    [
        ("https://github.com/CherryHQ/cherry-studio/issues/14250", False),
        ("https://github.com/orchard/another-repo/issues/6", False),
        ("https://github.com/another-owner/scope-recall/issues/6", False),
        ("https://mirror.example/orchard/scope-recall/issues/6", False),
        ("issues/6", False),
        ("https://github.com/Orchard/Scope-Recall/issues/6/files", True),
    ],
)
def test_issue_conflict_requires_the_same_host_owner_and_repository(candidate, conflicts):
    search = numbered_links("https://github.com/orchard/scope-recall/issues/5")
    assert numbered_link_conflict(search, candidate) is conflicts


def test_repository_case_does_not_change_a_records_identity():
    mixed = "https://github.com/CherryHQ/cherry-studio/issues/14250"
    alias = "https://github.com/cherryhq/Cherry-Studio/issues/14250#reply"
    assert numbered_links(mixed) == numbered_links(alias)
    assert not numbered_link_conflict(numbered_links(mixed), alias)


def test_chinese_text_ends_numbered_links_without_a_space():
    first = "https://github.com/orchard/scope-recall/issues/5"
    second = "https://github.com/CherryHQ/cherry-studio/issues/14250"
    assert numbered_links(f"{first}和{second}里的讨论") == numbered_links(first) | numbered_links(second)
    assert numbered_links(f"{first}more") == set()


@pytest.mark.parametrize("nominate", [True, False])
def test_status_links_bind_the_post_across_handles_hosts_and_media_views(tmp_path, nominate):
    core, ctx, vectors = _app(tmp_path)
    query = "https://x.com/orchard42/status/42017里说的是什么？"
    wrong = _capture(core, ctx, "wrong-post", "旧帖说要换皮带 https://twitter.com/pump/status/42018/photo/1")
    right = _capture(core, ctx, "right-post", "帖子说要换轴承 https://twitter.com/i/web/status/42017")
    if nominate:
        vectors.candidates = (_candidate(wrong, score=0.99), _candidate(right, score=0.95))
    packet = core.recall_packet(replace(ctx, session_id="reader"), recall_request(query=query))
    assert right.ref in {item["ref"] for item in packet["items"]}
    assert all("42018" not in item["content"] for item in packet["items"])


@pytest.mark.parametrize(
    "query",
    [
        "我是谁",
        "我以前删除了吗",
        "我喜欢什么",
        "Who am I?",
        "What did we delete?",
        "继续查找我的偏好",
        "Is Orchard running?",
        "Thanks, what is my preferred editor?",
        "Continue, thanks, and check the orchard pressure",
        "继续，谢谢，检查喷头压力",
        "按你说的设置喷头压力",
        "可以，按你建议的改配置文件",
    ],
)
def test_owner_past_and_named_topic_questions_are_not_skipped(tmp_path, query):
    core, ctx, _ = _app(tmp_path)
    _capture(core, ctx, "earlier", "synthetic session context")
    with core.storage.read(ctx) as tx:
        assert automatic_recall_skip_reason(query, tx=tx, context=ctx) is None


@pytest.mark.parametrize("query,reason", [("", "blank_prompt"), ("  ", "blank_prompt"), (" /help", "slash_command")])
def test_input_only_skips_need_no_session(query, reason):
    assert automatic_recall_skip_reason(query) == reason


@pytest.mark.parametrize("indexed", [True, False], ids=["session-index", "before-index-initialization"])
def test_session_probe_finds_the_prior_turn_despite_another_sessions_traffic(tmp_path, indexed):
    from scope_recall.core.schema import SOURCE_SESSION_INDEX

    _, ctx, _ = _app(tmp_path)
    with sqlite3.connect(":memory:") as conn:
        conn.execute("""CREATE TABLE source_events(scope_id, occurred_at, event_id, source_revision,
                     session_id, entry_id, role, source_original_origin, origin,
                     project_id, branch_id, read_blocked, content_sha256)""")
        conn.execute("""CREATE INDEX source_scope_time
                     ON source_events(scope_id,occurred_at,event_id,source_revision)""")
        if indexed:
            conn.execute(SOURCE_SESSION_INDEX)
        conn.execute("CREATE INDEX source_content ON source_events(scope_id,role,content_sha256)")
        insert = """INSERT INTO source_events(scope_id,occurred_at,event_id,source_revision,session_id,entry_id,
            role,source_original_origin,origin,project_id,branch_id,read_blocked) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"""
        conn.executemany(
            insert,
            (
                (
                    "TEST-scope",
                    str(index).zfill(8),
                    str(index),
                    1,
                    "TEST-other-session",
                    "local",
                    "user",
                    None,
                    "human_direct",
                    None,
                    None,
                    0,
                )
                for index in range(10000)
            ),
        )
        steps = []
        conn.set_progress_handler(lambda: steps.append(1) or 0, 100)
        tx = SimpleNamespace(_check=lambda: conn)
        assert automatic_recall_skip_reason("继续", tx=tx, context=ctx) is None
        if indexed:
            assert len(steps) < 150, "a first prompt must not scan every earlier human message in its scope"
        conn.execute(
            insert,
            ("TEST-scope", "", "TEST-earlier", 1, ctx.session_id, "local", "user", None, "human_direct", None, None, 0),
        )
        steps.clear()
        assert automatic_recall_skip_reason("还有什么没做完的吗？", tx=tx, context=ctx) == "unanchored_followup"
        if indexed:
            assert len(steps) < 150


def test_session_probe_obeys_the_deadline_and_releases_its_handler(tmp_path):
    from scope_recall.contracts import ContractError

    core, ctx, _ = _app(tmp_path)
    _capture(core, ctx, "prior", "Prepare a synthetic poster.")
    with core.storage.read(ctx) as tx:
        with pytest.raises(ContractError, match="DEADLINE_EXCEEDED"):
            automatic_recall_skip_reason("继续", tx=tx, context=ctx, deadline=99.0, clock=core.clock)
        assert tx._check().execute("SELECT 1").fetchone()[0] == 1


def test_session_probe_interrupts_an_inflight_statement(tmp_path):
    from scope_recall.contracts import ContractError

    _, ctx, _ = _app(tmp_path)
    ticks = iter(range(100))
    with sqlite3.connect(":memory:") as conn:

        class SlowProbe:
            set_progress_handler = conn.set_progress_handler

            def execute(self, *args):
                return conn.execute("""WITH RECURSIVE rows(n) AS (VALUES(1) UNION ALL
                                    SELECT n+1 FROM rows WHERE n<100000) SELECT sum(n) FROM rows""")

        tx = SimpleNamespace(_check=lambda: SlowProbe())
        with pytest.raises(ContractError, match="DEADLINE_EXCEEDED"):
            automatic_recall_skip_reason(
                "继续", tx=tx, context=ctx, deadline=3.0, clock=SimpleNamespace(monotonic=lambda: next(ticks))
            )
        assert SlowProbe().execute().fetchone()[0] == 5000050000


def _said_as(core, ctx, key, content, *, role):
    origin = "assistant_visible" if role == "assistant" else "human_direct"
    event = source_event(source_event_key=key, origin=origin, role=role, content=content)
    saved = core.record_event(
        replace(ctx, actor_origin=origin), event, scope_id=next(iter(ctx.allowed_scope_ids)), remaining_seconds=10
    )
    assert saved.durability == "persisted"
    return core.source(ctx, saved.event_refs[0].ref, 1)


def _embedded(core, *sources):
    """Finish the sources' embedding work, as the worker does once their vectors are written."""
    with closing(sqlite3.connect(core.storage.path)) as conn:
        conn.executemany(
            "UPDATE work_items SET state='done' WHERE work_type='embed' AND subject_ref=?",
            [(source.ref,) for source in sources],
        )
        conn.commit()


SEEDLINGS = "What watering schedule did we settle on for the tomato seedlings?"


def _passed_over(tmp_path, *, role="assistant", embedded=True, note=""):
    """A message whose words match ``SEEDLINGS`` and a vector search that ranked only something else, below the
    threshold."""
    core, ctx, vectors = _app(tmp_path)
    content = f"Tomato seedlings do best on a steady watering schedule.{note}"
    said = _said_as(core, ctx, "TEST-seedlings", content, role=role)
    other = _capture(core, ctx, "TEST-unrelated", "TEST an unrelated note about invoices.")
    if embedded:
        _embedded(core, said, other)
    vectors.candidates = (_candidate(other, score=0.3),)
    return core, ctx, vectors, said


def test_an_embedded_reply_the_vector_search_passed_over_stays_out_of_automatic_injection(tmp_path):
    core, ctx, _vectors, reply = _passed_over(tmp_path)

    automatic = core.recall_packet(ctx, recall_request(query=SEEDLINGS))
    explicit = recall(core, ctx, query=SEEDLINGS, mode="current")

    assert reply.ref not in {item["ref"] for item in automatic["items"]}
    assert reply.ref in {item.ref for item in explicit.items}


@pytest.mark.parametrize("case", ["not_embedded", "owner_message", "vector_unavailable", "identifier"])
def test_a_passed_over_lexical_hit_stays_when_meaning_could_not_judge_it(tmp_path, case):
    """A reply said a moment ago has no vector yet, the owner's own messages answer aggregate questions, a failed
    vector search ranked nothing, and an identifier is found by its words."""
    core, ctx, vectors, said = _passed_over(
        tmp_path,
        role="user" if case == "owner_message" else "assistant",
        embedded=case != "not_embedded",
        note=" SR4127 tracks it." if case == "identifier" else "",
    )
    query = SEEDLINGS
    if case == "vector_unavailable":

        def unavailable(*_args, **_kwargs):
            raise TimeoutError("TEST vector search timed out")

        vectors.search = unavailable
    if case == "identifier":
        query = "What watering schedule did SR4127 set for the tomato seedlings?"

    automatic = core.recall_packet(ctx, recall_request(query=query))

    assert said.ref in {item["ref"] for item in automatic["items"]}, automatic["gaps"]
