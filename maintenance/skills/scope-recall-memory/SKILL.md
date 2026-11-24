---
name: scope-recall-memory
description: Use when the user asks what Scope Recall remembers about them, where a memory came from, whether it still holds, or wants one corrected, muted or deleted ("你记了我什么", "这条是我说的还是你总结的", "现在还有效吗", "改掉", "别再主动提", "删掉"). Says what a deletion takes with it before doing it.
---

The user asks in plain words and gets a plain answer. Do the lookups yourself; do
not hand them refs or JSON to read. Never say something is remembered, changed or
gone unless a tool said so. Tools: `profile`, `entity`, `recall`, `inspect`,
`trace`, `status` (plus `revise`, `forget`, and `propose_memory` where the host
exposes verified request-message binding).

When the user states a new value in conversation, no tool call is needed. Follow
the new statement in your reply. Explicit `revise` and `forget` require a
host-verified binding to the exact latest complete user message in this model
request (`request_message_binding`, supported on Hermes and local Codex shared
entries). On MCP hosts without conversation-message attestation (such as Claude
Code, WorkBuddy, dsh, Antigravity, or Cherry Studio), or whenever
`request_message_binding` is returned, use the local memory management page or
`scope-recall memory` commands for explicit corrections, withdrawal, suppression
and deletion.

## What do you remember about me?

1. `profile` with `subject` set to the name the user goes by here (try the other
   names they use if the first finds nothing). It returns facts that are confirmed
   or in dispute, never unconfirmed ones.
2. `recall` with `mode: "current"` and a query in their words, for what `profile`
   does not list. An item whose `claim_state` is `proposed` is not confirmed yet;
   say so rather than presenting it as known.
3. Answer in sentences, grouped by topic. `no_match` means nothing is remembered
   in Scope Recall (`fallback_to_transcript_search_for_verbatim_history` in
   `unmet_needs` marks when verbatim transcript search is the next fallback for
   past conversation wording; for "what do you remember about me?", report that
   nothing is stored rather than inventing memories from chat fragments).

## Did I say this, or did you work it out?

Read the item's `basis`: `direct_report` was said by someone (and `origin:
human_direct` means the user typed it), `observed` was seen in a tool result,
`derived_summary` is the plugin's own summary, `inferred_suggestion` is a guess.
To show where it came from, `inspect` each of its `evidence_refs` and quote the
line. If there is no evidence to show, say that.

## Which agent heard it?

When several agents share one memory, there is one store, not one per agent:
each agent runs its own copy of this plugin, and all of them write to and read
from that store. What each can read follows its own chats: what the owner said
to any of the agents in the owner's own chats is common to all of them, while a
group or a conversation only one agent took part in stays with that agent.

In a shared memory an item carries `entries`: the agent (id and name) each
piece of its evidence came in through. An item from another agent is that
agent's experience, not yours. When you use it, say which agent it came from,
and never describe it as something you did or were told yourself. An item
without `entries` comes from a memory only you use.

## Is it still true?

`temporal_status` (`current`, `historical`, `disputed`, `unknown`) and
`claim_state` (`active`, `proposed`, `superseded`, `disputed`, `retracted`) answer
it. For what changed and when, `recall` with `mode: "history"`; say what replaced
the old value.

## Change it, stop bringing it up, or delete it

Use `revise` or `forget` when available and bound to the user's current request
message; when a host returns `request_message_binding` or omits those tools,
explain the effects and direct the person to local memory management (`scope-recall ui`
or `scope-recall memory`). The trusted core's source-backed mutation contract
enforces the following rules:

- **Correct** (`revise`): writes a new version; the old one stays as history.
  Their message must contain the new value, a correcting word (改为, 改成, 换成,
  更正, 纠正; correct, replace, change to, switch to) and what is being corrected
  (its subject or its old value).
- **Withdraw** (`revise` with `new_value: null`): "this is no longer so", with
  nothing to put in its place (撤回, 撤销, 作废, 不再使用, 停止使用; retract, withdraw,
  stop using).
- **Stop bringing it up** (`forget` with `mode: "suppress"`): it is no longer used
  unasked, and is still found when they ask for it. Their message must say so
  (不要主动提, 别再主动, 不再主动提; do not mention) and name the item. There is no
  tool that undoes this. Say that before doing it.
- **Delete** (`forget` with `mode: "delete"`): erases the memory, **the whole
  message it came from, and every other fact taken from that same message**, in
  every version. It cannot be undone, and the message that asks for it is erased
  too. It does not reach the chat app's own history, the host's transcripts, text
  already shown in this conversation, or a backup somebody made.
  1. `inspect` the item and its `evidence_refs`, and tell the user what else came
     from that message. If you cannot tell, say that other facts from the same
     message will go with it.
  2. The core source-backed contract recognizes explicit deleting words (删除,
     删掉, 忘掉, 清除; delete, erase, forget) and the named item. A batch names every ref.
  3. Call `forget(mode="delete", ...)` with current versions when bound, or use local memory management otherwise.
  4. Report the resulting receipt's `affected_objects` and cleanup layer states.

In the trusted core contract, `forget_not_authorized` means the stored message did not ask
plainly; `target_not_bound` means it did not name the item; `explicit_batch_required`
means several items need every ref spelled out; `VERSION_CONFLICT` means the item
changed. The model tools return `request_message_binding` before these target
checks, including when retrying a deletion receipt.

## "Don't record this"

Use the local memory management page for capture privacy controls and deletion
of existing material. Scope Recall's controls apply to its own stored memory.
