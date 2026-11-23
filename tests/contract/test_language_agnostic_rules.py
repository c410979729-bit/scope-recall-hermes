"""Synthetic contracts for corpus IDF recall admission, 7-key semantic verdicts, and alias cleanup."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from scope_recall.contracts import ClaimProposal, TrustedSourcePrincipal
from scope_recall.core.claims import Qualification, qualify
from scope_recall.core.entity_names import eligible_entity_name
from scope_recall.core.entity_resolution import (
    approve_entity_alias,
    entity_name_evidenced,
    publish_entity_merge,
)
from scope_recall.core.evidence_support import SupportVerdict, using_support
from scope_recall.core.recall_channels import _matched_statement_terms
from scope_recall.core.recall_policy import (
    RecallPolicy,
    meaningful_query_terms,
    query_has_specific_anchor,
    query_term_information,
    ubiquity_ceiling,
    word_document_frequencies,
)
from scope_recall.core.retrieval import CandidateRef, RetrievedObject
from scope_recall.core.review_policy import set_policy
from scope_recall.core.semantic_proposals import prepare_publication, prepare_support, validate_judgement
from v11_support import recall_request

from tests.contract.test_claims import accept, app, capture, draft  # noqa: F401


def _owner_ctx(ctx):
    return replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-owner"))


def _positioned_draft(source, quote: str, value_text: str, **fields) -> ClaimProposal:
    content = source.event["content"]
    start = content.index(quote)
    proposal = draft(
        source,
        value_text,
        kind=fields.pop("kind", "preference"),
        statement_kind=fields.pop("statement_kind", "assertion"),
        subject=fields.pop("subject", "current_user"),
        predicate=fields.pop("predicate", "饮品偏好"),
        conditions=fields.pop("conditions", []),
        **fields,
    )
    proposal["evidence_spans"][0]["quote"] = quote
    proposal["evidence_spans"][0]["start"] = start
    proposal["evidence_spans"][0]["end"] = start + len(quote)
    return proposal


def _envelope(proposal: ClaimProposal) -> dict:
    refs = list(dict.fromkeys(f"{s['source_ref']}@{s['source_revision']}" for s in proposal["evidence_spans"]))
    return dict(
        protocol_version="1.1",
        source_refs=refs,
        claim_proposals=[proposal],
        resume_proposals=[],
        reference_proposals=[],
    )


def _prepared_verdicts(
    proposal: ClaimProposal,
    source,
    *,
    state="supported",
    action="new",
    attribution="author",
    polarity="affirmative",
    scope="durable",
):
    reply = json.dumps(
        {
            "support": state,
            "action": action,
            "target_ref": None,
            "target_revision": None,
            "attribution": attribution,
            "polarity": polarity,
            "scope": scope,
        }
    )
    return prepare_support(_envelope(proposal), (source,), lambda messages, **kwargs: reply)


def _latest_version(core, ctx, ref: str):
    with core.storage.read(ctx) as tx:
        return tx.claims.versions(ref)[-1]


def test_corpus_idf_excludes_unseen_boundary_bigrams_and_blocks_pure_ubiquitous_matches():
    policy = RecallPolicy()
    query = "TEST-项目甲的偏好是什么"
    terms = meaningful_query_terms(query)
    assert "好是" in terms

    freqs = {"偏好": 80, "什么": 120}
    for term in terms:
        if "项目甲" in term or "test" in term:
            freqs[term] = 3

    specific_hit = CandidateRef(
        kind="claim",
        ref="claim-TEST-1",
        revision=1,
        source="lexical",
        matched_query_terms=tuple(t for t in terms if freqs.get(t) in {3, 80}),
    )
    assert policy.automatic_admission(specific_hit, query, term_frequencies=freqs, corpus_size=500) is True

    filler_only_hit = CandidateRef(
        kind="claim",
        ref="claim-TEST-2",
        revision=1,
        source="lexical",
        matched_query_terms=("偏好", "什么", "的偏"),
    )
    freqs_with_filler = {**freqs, "的偏": 90}
    assert (
        policy.automatic_admission(filler_only_hit, query, term_frequencies=freqs_with_filler, corpus_size=500) is False
    )

    ubiquitous_query = "继续检查一下刚才那个代码文件"
    u_terms = meaningful_query_terms(ubiquitous_query)
    u_freqs = {term: 95 for term in u_terms}
    u_partial_hit = CandidateRef(
        kind="event",
        ref="event-TEST-u",
        revision=1,
        source="lexical",
        matched_query_terms=u_terms[:5],
    )
    assert (
        policy.automatic_admission(u_partial_hit, ubiquitous_query, term_frequencies=u_freqs, corpus_size=500) is False
    )

    attachment_query = "继续检查一下刚才那个代码文件，下面是今天导出的构建日志信息详情"
    a_terms = meaningful_query_terms(attachment_query)
    a_freqs = {term: (95 if term in u_terms else 4) for term in a_terms}
    assert (
        policy.automatic_admission(u_partial_hit, attachment_query, term_frequencies=a_freqs, corpus_size=500) is False
    )

    segmented_doc_hit = CandidateRef(
        kind="event",
        ref="event-TEST-doc",
        revision=1,
        source="lexical",
        matched_query_terms=("blue", "paint"),
    )
    assert (
        policy.automatic_admission(
            segmented_doc_hit, "blue paint", term_frequencies={"blue": 125, "paint": 125}, corpus_size=125
        )
        is True
    )


def test_validate_judgement_accepts_legacy_four_keys_and_structured_seven_keys():
    legacy = {"support": "supported", "action": "new", "target_ref": None, "target_revision": None}
    assert validate_judgement(legacy, ())["support"] == "supported"

    structured = {
        **legacy,
        "attribution": "author",
        "polarity": "affirmative",
        "scope": "durable",
    }
    assert validate_judgement(structured, ())["attribution"] == "author"

    with pytest.raises(Exception, match="support_verdict|attribution"):
        validate_judgement({**legacy, "attribution": "author"}, ())

    with pytest.raises(Exception, match="attribution"):
        validate_judgement({**structured, "attribution": "invented_role"}, ())


@pytest.mark.parametrize(
    "utterance,quote,predicate,value_text",
    [
        ("TEST 吾喜饮清茶。", "吾喜饮清茶", "饮茶偏好", "清茶"),
        ("TEST 私は緑茶が好きです。", "私は緑茶が好きです", "tea_preference", "緑茶"),
        ("TEST Je préfère le thé vert.", "Je préfère le thé vert", "tea_preference", "thé vert"),
        ("TEST 界面主色统一采用深岩灰。", "界面主色统一采用深岩灰", "界面主色", "深岩灰"),
    ],
)
def test_multilingual_classical_and_omitted_self_statements_bind_via_structured_verdict(
    app, utterance, quote, predicate, value_text
):
    core, raw_ctx = app
    ctx = _owner_ctx(raw_ctx)
    source = capture(core, ctx, utterance)
    proposal = _positioned_draft(source, quote, value_text, predicate=predicate)
    verdicts = _prepared_verdicts(proposal, source, attribution="author", polarity="affirmative", scope="durable")
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    stored = core.current_claim(ctx, receipt.items[0].ref)
    assert stored is not None and stored.state == "active"
    assert stored.payload["subject"] == "principal:TEST-owner"


@pytest.mark.parametrize(
    "utterance,quote,attribution,scope,expected_reason",
    [
        ("TEST 客服说下个月套餐会涨价。", "客服说下个月套餐会涨价", "third_party", "durable", "other_speaker"),
        (
            "TEST My doctor said avoid espresso.",
            "My doctor said avoid espresso",
            "third_party",
            "durable",
            "other_speaker",
        ),
        (
            "TEST 这一轮先用浅灰背景试一下。",
            "这一轮先用浅灰背景试一下",
            "author",
            "ephemeral_turn",
            "relative_scope_not_preserved",
        ),
        (
            "TEST 要是改用云端方案，预算大概要翻倍。",
            "要是改用云端方案，预算大概要翻倍",
            "author",
            "unasserted",
            "hypothetical_or_undecided",
        ),
    ],
)
def test_third_party_ephemeral_and_unasserted_verdicts_stay_proposed_and_quarantined_from_auto_recall(
    app, utterance, quote, attribution, scope, expected_reason
):
    core, raw_ctx = app
    ctx = _owner_ctx(raw_ctx)
    source = capture(core, ctx, utterance)
    subject = "current_user" if attribution == "third_party" else "TEST-ProjectAlpha"
    proposal = _positioned_draft(source, utterance, quote[:4], subject=subject, predicate="TEST-pred", kind="fact")
    verdicts = _prepared_verdicts(proposal, source, attribution=attribution, polarity="affirmative", scope=scope)
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert receipt.items[0].state == "proposed"
    stored = _latest_version(core, ctx, receipt.items[0].ref)
    assert stored.state == "proposed"
    assert stored.reason == expected_reason

    follow = capture(core, ctx, f"查询 {utterance}")
    auto_packet = core.recall(
        ctx,
        recall_request(query=utterance, mode="auto"),
        current_source_refs=(f"{follow.ref}@{follow.revision}",),
    )
    assert all(item.ref != stored.ref for item in auto_packet.items)


def test_negative_normalized_claims_and_lexical_non_negations_reconcile_cleanly(app):
    core, raw_ctx = app
    ctx = _owner_ctx(raw_ctx)
    norm_source = capture(core, ctx, "I prefer deep blue.")
    norm_proposal = _positioned_draft(norm_source, "I prefer deep blue.", "navy", subject="user", predicate="color")
    norm_verdicts = _prepared_verdicts(
        norm_proposal, norm_source, attribution="author", polarity="affirmative", scope="durable"
    )
    with using_support(norm_verdicts):
        norm_receipt = accept(core, ctx, norm_proposal)
    assert norm_receipt.items[0].state == "active"

    neg_source = capture(core, ctx, "TEST 吾平素不食香菜。")
    neg_proposal = _positioned_draft(neg_source, "吾平素不食香菜", "香菜", predicate="忌口")
    neg_verdicts = _prepared_verdicts(
        neg_proposal, neg_source, attribution="author", polarity="negative", scope="durable"
    )
    with using_support(neg_verdicts):
        neg_receipt = accept(core, ctx, neg_proposal)
    assert neg_receipt.items[0].state == "active"

    dropped_neg = _positioned_draft(neg_source, "吾平素不食香菜", "香菜", predicate="喜好")
    bad_verdicts = _prepared_verdicts(
        dropped_neg, neg_source, state="unsupported", attribution="author", polarity="negative", scope="durable"
    )
    with core.storage.read(ctx) as tx:
        roots = tx.claims.roots((f"{neg_source.ref}@{neg_source.revision}",))
    with using_support(bad_verdicts):
        assert qualify(dropped_neg, roots).state == "proposed"

    lex_source = capture(core, ctx, "TEST 我平时在无线网络下使用深色模式。")
    lex_proposal = _positioned_draft(lex_source, "我平时在无线网络下使用深色模式", "深色模式", predicate="界面模式")
    lex_verdicts = _prepared_verdicts(
        lex_proposal, lex_source, attribution="author", polarity="affirmative", scope="durable"
    )
    with using_support(lex_verdicts):
        lex_receipt = accept(core, ctx, lex_proposal)
    assert lex_receipt.items[0].state == "active"


def test_third_party_update_verdict_cannot_overwrite_owner_principal_slot(app):
    core, raw_ctx = app
    ctx = _owner_ctx(raw_ctx)
    first = capture(core, ctx, "TEST 我喜欢喝绿茶。")
    owner_proposal = _positioned_draft(first, "我喜欢喝绿茶", "绿茶", predicate="饮茶偏好")
    first_verdicts = _prepared_verdicts(
        owner_proposal, first, attribution="author", polarity="affirmative", scope="durable"
    )
    with using_support(first_verdicts):
        owner_ref = accept(core, ctx, owner_proposal).items[0].ref

    second = capture(core, ctx, "TEST 客服说饮茶偏好是乌龙茶。")
    intruder = _positioned_draft(second, "客服说饮茶偏好是乌龙茶", "乌龙茶", predicate="饮茶偏好")

    def rogue_update_judge(messages, **kwargs):
        return json.dumps(
            {
                "support": "supported",
                "action": "update",
                "target_ref": owner_ref,
                "target_revision": 1,
                "attribution": "third_party",
                "polarity": "affirmative",
                "scope": "durable",
            }
        )

    verdicts = prepare_publication(
        core.storage, ctx, _envelope(intruder), (second,), rogue_update_judge, clock=core.clock
    )
    with using_support(verdicts):
        receipt = accept(core, ctx, intruder)
    assert receipt.items[0].ref != owner_ref
    assert receipt.items[0].state == "proposed"
    owner_current = core.current_claim(ctx, owner_ref)
    assert owner_current is not None and owner_current.revision == 1
    assert owner_current.payload["value_text"] == "绿茶"
    assert len(core.claim_history(ctx, owner_ref)) == 1


@pytest.mark.parametrize(
    "bad_label",
    [
        "",
        "x",
        "--dry-run",
        "-v",
        "FOO_BAR=1",
        "v4.0.6",
        "6b01878e0e07c205",
        "principal:cli:user",
        "current_user",
        "me",
        "my",
        "myself",
    ],
)
def test_eligible_entity_name_rejects_structural_non_entities(bad_label):
    assert eligible_entity_name(bad_label) is False


def test_entity_name_evidenced_requires_subject_in_quote_neighbourhood(app):
    core, ctx = app
    text = (
        "TEST-Atlas runs our primary search service;\n"
        "Two sentences later we checked routine metrics;\n"
        "Separately, TEST-UnrelatedTool finished its nightly job!"
    )
    raw = capture(core, ctx, text)
    proposal = _positioned_draft(
        raw,
        "Separately, TEST-UnrelatedTool finished its nightly job!",
        "nightly job",
        kind="fact",
        subject="TEST-Atlas",
        predicate="status",
        entity_names=["TEST-UnrelatedTool"],
    )
    first_quote = "TEST-Atlas runs our primary search service;"
    proposal["evidence_spans"].insert(
        0,
        {
            "source_ref": raw.ref,
            "source_revision": raw.revision,
            "quote": first_quote,
            "start": 0,
            "end": len(first_quote),
        },
    )
    verdicts = _prepared_verdicts(proposal, raw, attribution="author", polarity="affirmative", scope="durable")
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    stored = core.current_claim(ctx, receipt.items[0].ref)
    assert stored is not None
    with core.storage.read(ctx) as tx:
        sources = tx.claims.live_sources((f"{raw.ref}@{raw.revision}",), stored.scope_id)
    assert entity_name_evidenced(stored, "TEST-UnrelatedTool", sources=sources) is False

    cased_raw = capture(core, ctx, "test-atlas routes traffic through TEST-GatewayNode.")
    cased_prop = _positioned_draft(
        cased_raw,
        "routes traffic through TEST-GatewayNode.",
        "TEST-GatewayNode",
        kind="fact",
        subject="TEST-Atlas",
        predicate="route",
        entity_names=["TEST-GatewayNode"],
    )
    cased_verdicts = _prepared_verdicts(
        cased_prop, cased_raw, attribution="author", polarity="affirmative", scope="durable"
    )
    with using_support(cased_verdicts):
        cased_stored = core.current_claim(ctx, accept(core, ctx, cased_prop).items[0].ref)
    assert cased_stored is not None
    with core.storage.read(ctx) as tx:
        cased_sources = tx.claims.live_sources((f"{cased_raw.ref}@{cased_raw.revision}",), cased_stored.scope_id)
    assert entity_name_evidenced(cased_stored, "TEST-GatewayNode", sources=cased_sources) is True


def test_retire_invalid_aliases_previews_read_only_and_preserves_owner_approved_aliases(app):
    core, ctx = app
    s_target = capture(core, ctx, "TEST-NorthGate role edge-gateway.")
    c_target = core.current_claim(
        ctx,
        accept(
            core,
            ctx,
            draft(
                s_target,
                "edge-gateway",
                kind="fact",
                statement_kind="assertion",
                subject="TEST-NorthGate",
                predicate="role",
            ),
        )
        .items[0]
        .ref,
    )
    s_source = capture(
        core,
        ctx,
        "TEST-EdgeProxy routing TEST-NorthGate. Later sentence. Unrelated mentions TEST-DistantNoun.",
    )
    c_source = core.current_claim(
        ctx,
        accept(
            core,
            ctx,
            draft(
                s_source,
                "TEST-NorthGate",
                kind="fact",
                statement_kind="assertion",
                subject="TEST-EdgeProxy",
                predicate="routing",
                entity_names=["TEST-DistantNoun"],
            ),
        )
        .items[0]
        .ref,
    )
    assert c_target is not None and c_source is not None
    decision = SupportVerdict("", "supported", c_target.ref, c_target.revision, "duplicate")
    with core.storage.write(ctx) as tx:
        set_policy(tx, core.clock, c_source.scope_id, True, 0)
        pending = publish_entity_merge(tx, c_source, c_target, decision, now=core.clock.utc_now())
        approved = approve_entity_alias(tx, core.clock, pending, pending)
        set_policy(tx, core.clock, c_source.scope_id, False, 1)

        bad_proposal = json.loads(json.dumps(approved.payload))
        bad_proposal["predicate"] = "entity_identity:legacybad"
        bad_proposal["value_text"] = "--bad-flag"
        bad_proposal["alias"]["name"] = "--bad-flag"
        bad_saved = tx.claims.append(
            c_source.scope_id,
            bad_proposal,
            Qualification("active", "derived_summary", "entity_identity_supported"),
            recorded_at=core.clock.utc_now(),
        )

    s_mgmt = capture(core, ctx, "TEST-WestEdge routing TEST-NorthGate.")
    c_mgmt = core.current_claim(
        ctx,
        accept(
            core,
            ctx,
            draft(
                s_mgmt,
                "TEST-NorthGate",
                kind="fact",
                statement_kind="assertion",
                subject="TEST-WestEdge",
                predicate="routing",
            ),
        )
        .items[0]
        .ref,
    )
    assert c_mgmt is not None
    from tests.contract.test_memory_admin import management

    mgmt_merged = management(core, ctx).health_merge_entity(
        c_mgmt.ref, c_mgmt.revision, c_target.ref, c_target.revision, name="TEST-WestEdge"
    )["ref"]

    preview = core.operations.retire_invalid_aliases(ctx, limit=16, dry_run=True)
    assert preview["applied"] is False
    assert [item["ref"] for item in preview["changed"]] == [bad_saved.ref]
    assert any(item["ref"] == approved.ref and item["reason"] == "owner_preserved" for item in preview["skipped"])
    assert any(item["ref"] == mgmt_merged and item["reason"] == "owner_preserved" for item in preview["skipped"])
    assert core.current_claim(ctx, bad_saved.ref) is not None

    applied = core.operations.retire_invalid_aliases(ctx, limit=16, dry_run=False)
    assert applied["applied"] is True
    assert [item["ref"] for item in applied["changed"]] == [bad_saved.ref]
    assert core.current_claim(ctx, bad_saved.ref) is None
    assert _latest_version(core, ctx, bad_saved.ref).state == "retracted"
    assert core.current_claim(ctx, approved.ref) is not None
    assert core.current_claim(ctx, mgmt_merged) is not None


def test_cjk_boundary_monotonicity_and_sublinear_ubiquity_separate_noise_without_wordlists():
    corpus_size = 400_000
    ceiling = ubiquity_ceiling(corpus_size)
    assert 3_000 <= ceiling <= 3_500

    steering = "目前什么进度了，下一步应该做什么？"
    raw = {
        "目前": 5600,
        "前什": 420,
        "什么": 12000,
        "么进": 95,
        "进度": 4200,
        "度了": 310,
        "下一": 6000,
        "一步": 5500,
        "步应": 170,
        "应该": 3800,
        "该做": 490,
        "做什": 1080,
    }
    adjusted = word_document_frequencies(steering, raw)
    assert adjusted["步应"] >= 3800
    assert query_has_specific_anchor(steering, raw, corpus_size) is False

    compound = "项目甲的偏好是什么？"
    compound_raw = {
        "项目": 20000,
        "目甲": 3,
        "甲的": 2,
        "的偏": 50,
        "偏好": 4000,
        "好是": 100,
        "是什": 2000,
        "什么": 12000,
    }
    _cpresent, cspecific, canchors = query_term_information(
        meaningful_query_terms(compound), compound, compound_raw, corpus_size
    )
    assert "目甲" in cspecific
    assert "目甲" in canchors
    assert query_has_specific_anchor(compound, compound_raw, corpus_size) is True
    policy = RecallPolicy(vector_threshold=0.70)
    compound_claim = CandidateRef(
        "claim", "claim-TEST-cmp", 1, "claim_lexical", matched_query_terms=("项目", "目甲", "偏好")
    )
    assert policy.automatic_admission(compound_claim, compound, term_frequencies=compound_raw, corpus_size=corpus_size)

    assert query_has_specific_anchor("What is Quasar991?", {"quasar991": 1}, 600) is True
    assert query_has_specific_anchor("デプロイ先はどこですか？", {}, 400_000) is True

    dated_cand = CandidateRef("event", "event-TEST-dated", 1, "dated", matched_query_terms=("测试", "结果"))
    assert (
        policy.automatic_admission(
            dated_cand, "测试结果是什么？", term_frequencies={"测试": 20000, "结果": 15000}, corpus_size=corpus_size
        )
        is True
    )

    proc_obj = RetrievedObject(
        kind="procedure",
        ref="claim-TEST-proc",
        revision=1,
        content=json.dumps(
            {
                "kind": "procedure",
                "subject": "ScopeRecall",
                "predicate": "部署步骤",
                "value_text": "先备份再平滑重启",
                "conditions": [],
                "entity_names": ["天璇网关"],
                "procedure": {"steps": ["backup"], "verification_basis": "doctor"},
            },
            ensure_ascii=False,
        ),
        origin="human_direct",
        temporal_status="current",
        applicability="trusted scope",
        evidence_refs=("event-TEST-1@1",),
        basis="direct_report",
        expandable=True,
    )
    assert "天璇" in _matched_statement_terms("天璇网关的部署步骤 verification_basis", proc_obj)
    assert "verification_basis" not in _matched_statement_terms("天璇网关的部署步骤 verification_basis", proc_obj)

    obs_payload = {
        "kind": "observation",
        "subject": "天枢实例",
        "predicate": "端口规律",
        "value_text": "监听 8652",
        "conditions": [],
        "entity_names": ["Port8652"],
    }
    obs_obj = RetrievedObject(
        kind="claim",
        ref="claim-TEST-obs",
        revision=1,
        content="Inductive observation (not the owner's words): 监听 8652",
        origin="inductive_observation",
        temporal_status="current",
        applicability="trusted scope",
        evidence_refs=("event-TEST-2@1",),
        basis="observed",
        expandable=False,
        metadata=(("payload_json", json.dumps(obs_payload, ensure_ascii=False)),),
    )
    assert "天枢" in _matched_statement_terms("天枢实例 Port8652 监听什么", obs_obj)
    assert "port8652" in _matched_statement_terms("天枢实例 Port8652 监听什么", obs_obj)

    target = "所有私钥和凭据应该存放在哪里"
    target_raw = {
        "所有": 7800,
        "有私": 80,
        "私钥": 850,
        "钥和": 9,
        "和凭": 140,
        "凭据": 5800,
        "据应": 35,
        "应该": 3800,
        "该存": 28,
        "存放": 2400,
        "放在": 2400,
        "在哪": 2100,
        "哪里": 1280,
    }
    _tpresent, tspecific, tanchors = query_term_information(
        meaningful_query_terms(target), target, target_raw, corpus_size
    )
    assert "私钥" in tspecific
    assert "私钥" in tanchors
    assert query_has_specific_anchor(target, target_raw, corpus_size) is True


@pytest.mark.parametrize(
    "prompt,freqs",
    [
        ("开始吧", {"开始": 6400}),
        ("善，依汝所言", {"依汝": 0, "汝所": 0, "所言": 4100}),
        ("Proceed as you see fit", {"proceed": 3800, "as": 25000, "you": 11000, "see": 8400, "fit": 4200}),
        ("今の進捗はどう？", {"進捗": 0}),
    ],
)
def test_multilingual_unanchored_prompts_fail_mature_anchor_and_lexical_admission(prompt, freqs):
    policy = RecallPolicy(vector_threshold=0.70)
    corpus_size = 400_000
    assert query_has_specific_anchor(prompt, freqs, corpus_size) is False
    terms = meaningful_query_terms(prompt)
    candidate = CandidateRef("event", "event-TEST-steer", 1, "lexical", matched_query_terms=terms)
    assert policy.automatic_admission(candidate, prompt, term_frequencies=freqs, corpus_size=corpus_size) is False


def test_mature_marginal_vector_requires_primary_anchor_while_strong_paraphrase_passes():
    policy = RecallPolicy(vector_threshold=0.70)
    corpus_size = 400_000
    query = "帮我写一个二分查找的边界测试用例"
    freqs = {
        "帮我": 2480,
        "我写": 389,
        "写一": 1218,
        "一个": 19400,
        "个二": 110,
        "二分": 32,
        "分查": 19,
        "查找": 417,
        "找的": 104,
        "的边": 698,
        "边界": 8800,
        "界测": 108,
        "测试": 22500,
        "试用": 681,
        "用例": 1200,
    }
    marginal_noise = CandidateRef(
        "event",
        "event-TEST-boundary",
        1,
        "vector",
        vector_id="vec-1",
        embedding_space=policy.embedding_space_id,
        vector_score=0.747,
        matched_query_terms=("边界", "测试", "用例"),
    )
    assert policy.automatic_admission(marginal_noise, query, term_frequencies=freqs, corpus_size=corpus_size) is False

    marginal_anchored = replace(marginal_noise, matched_query_terms=("二分", "查找", "边界"))
    assert policy.automatic_admission(marginal_anchored, query, term_frequencies=freqs, corpus_size=corpus_size) is True

    strong_paraphrase = replace(marginal_noise, vector_score=0.78, matched_query_terms=())
    assert policy.automatic_admission(strong_paraphrase, query, term_frequencies=freqs, corpus_size=corpus_size) is True


def test_consolidated_ungrounded_tool_event_stays_out_of_auto_recall_while_fresh_and_explicit_pass(app):
    core, ctx = app
    tool_ctx = replace(ctx, actor_origin="tool_observation")
    fresh_tool = capture(
        core,
        tool_ctx,
        "TEST-Quasar991 diagnostic output confirmed gateway port 19442 active.",
        origin="tool_observation",
    )
    probe = capture(core, ctx, "What port did TEST-Quasar991 report?")
    before = core.recall(
        ctx,
        recall_request(query="TEST-Quasar991 gateway port", mode="auto"),
        current_source_refs=(f"{probe.ref}@{probe.revision}",),
    )
    assert any(item.ref == fresh_tool.ref for item in before.items)

    with core.storage.write(ctx) as tx:
        conn = tx._check()
        conn.execute(
            "UPDATE work_items SET state='done' WHERE work_type IN ('consolidate','embed') "
            "AND subject_ref=? AND subject_revision=?",
            (fresh_tool.ref, fresh_tool.revision),
        )

    after_auto = core.recall(
        ctx,
        recall_request(query="TEST-Quasar991 gateway port", mode="auto"),
        current_source_refs=(f"{probe.ref}@{probe.revision}",),
    )
    assert all(item.ref != fresh_tool.ref for item in after_auto.items)

    after_explicit = core.recall(
        ctx,
        recall_request(query="TEST-Quasar991 gateway port", mode="current"),
        current_source_refs=(f"{probe.ref}@{probe.revision}",),
    )
    assert any(item.ref == fresh_tool.ref for item in after_explicit.items)


def test_unseen_paraphrase_terms_and_edge_clitics_do_not_starve_real_anchors():
    policy = RecallPolicy(vector_threshold=0.70)
    corpus_size = 400_000
    english = "Where should you book me when I need to concentrate?"
    en_freqs = {"book": 12, "me": 15000, "when": 9000, "need": 6500, "concentrate": 0}
    present, _spec, anchors = query_term_information(meaningful_query_terms(english), english, en_freqs, corpus_size)
    assert "concentrate" not in present
    assert "book" in anchors
    hit = CandidateRef("claim", "claim-TEST-book", 1, "lexical", matched_query_terms=("book", "me"))
    assert policy.automatic_admission(hit, english, term_frequencies=en_freqs, corpus_size=corpus_size) is True

    edge = "Hermes 升级 wheel 时怎么平滑切换"
    edge_freqs = {
        "hermes": 850,
        "升级": 1400,
        "wheel": 620,
        "时怎": 18,
        "怎么": 9200,
        "么平": 2,
        "平滑": 340,
        "滑切": 190,
        "切换": 980,
    }
    _ep, _espec, edge_anchors = query_term_information(meaningful_query_terms(edge), edge, edge_freqs, corpus_size)
    assert "时怎" not in edge_anchors
    assert "滑切" in edge_anchors
