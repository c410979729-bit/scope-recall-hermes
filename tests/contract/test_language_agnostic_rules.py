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
from scope_recall.core.recall_policy import RecallPolicy, meaningful_query_terms
from scope_recall.core.retrieval import CandidateRef
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
