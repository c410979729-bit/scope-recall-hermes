# Host capture and identity

Hermes, Codex, Claude Code, WorkBuddy and dsh share immutable installation bindings and exact audience grants.
Session ids distinguish conversations; they do not attest a person's identity. Local Hermes surfaces require the
owner's approved grant. A shared coding-client entry uses the route approved when it was attached. An unknown route
never borrows another route's scopes.

The host declarations in `host_registry.py` specify session and turn fields, lifecycle events, application
messages and local surfaces. The shared interpreter removes complete application envelopes before treating visible
words as evidence. Codex rollout recovery reads visible response messages and excludes analysis, tool traffic and
application suggestion requests. Claude Code reminders and task notices, and WorkBuddy wrappers, meta messages and
subagent records retain their host-specific filtering.

## Projects and linked worktrees

A local installation recognizes a linked worktree only when all four checks agree: its `.git` file names an
administrative directory below a registered repository's `.git/worktrees`, `commondir` resolves to that repository's
common directory, the administrative `gitdir` points back to the same worktree's `.git` file, and that common directory
belongs to a registered project root. Nested directories inherit that project's scopes. Invalid pointers remain
unrecognized. No Git process or remote URL grants access. A shared entry continues to use its attached audience.

## Tool inputs

The SDK's internal argument model remains the validation authority. The advertised input schema uses typed JSON Schema
unions for nullable values and enums for literal values, without changing required fields or defaults. Optional
`request_id`, `as_of`, `focus_refs`, `target`, `predicate`, `evidence_refs` and `reason` accept omission and null as their
existing unset meaning. Required arguments still require presence. The required `new_value` accepts any JSON value;
null withdraws the fact. Required `valid_from` accepts null and preserves it in the mutation request. Integer
arguments remain strict integers; booleans and numeric strings cannot substitute for them. Extra identity, scope,
path and session arguments remain forbidden.

The `recall` tool description names its usage contexts directly so every host's tool list carries the routing rule
without relying on an already-loaded skill: questions about earlier user decisions, preferences, standing conventions,
how something was done (`mode="method"`), whether a fact still holds (`mode="current"` or `"history"`), or where a
statement came from go to `recall` first, leaving raw transcript search for verbatim wording or when `recall` returns
`status="no_match"` (`fallback_to_transcript_search_for_verbatim_history` in `unmet_needs`). Host integrations should
rely on automatic turn recall and semantic tool selection rather than maintaining keyword lists, which break under
paraphrasing.

## Native Hermes request binding

Hermes model mutations require an executor-owned request identity and the latest complete user's durable message
UID. The recorded occurrence must match the actual provider input and the captured source. Execution middleware
binds the witness only around its real tool callback and revokes it on return. Known Chat Completions tool calls
and completed Responses function-call outputs are recognized before host normalization. Failed or incomplete
Responses outputs and unknown or ambiguous layouts grant no mutation authority; ordinary capture and reads retain
their existing paths.

## Recovery and privacy

The next end hook and the existing background worker reconcile session ledgers with captured sources. A timed-out or
killed hook does not have to return to make its host's recorded messages recoverable. Checkpoints are disposable;
repeated reads deduplicate against storage. Current grants, private intervals at the original message time and
inline `<private>...</private>` filtering apply on every replay. Recorded working directories and context-only
checkpoints preserve project privacy across resumed sessions. Missing or unverifiable directory provenance uses the
existing unverified-project policy rather than borrowing the host home as a verified project. Remote transport and its spool also strip inline
private words before making a copy. The native dsh spool calls the same local capture filter before writing tagged
text, even for a remote entry; an unavailable filter withholds that message. Ordinary messages need no filter process.
Worker and doctor expose bounded replay progress and recovery actions in
[Background work](background-work.md).

## Stdio lifetime

MCP stdio services observe both input EOF and the launching process's lifetime. On Windows a virtual-environment
redirector is distinguished from its launcher. Disconnect initiates cancellation and allows at most three seconds
for service cleanup. Hermes uses the host's native memory-provider lifecycle rather than a separate MCP stdio process.

## Resident recall

The background worker starts one recall process for a shared library or a standalone local Hermes store, discovered
through a directory private to this OS user. Every entry of that store uses the same store lock and signed loopback
transport. Requests name their entry and host; the service reloads the attachment or local manifest before choosing its
audience-specific handler. Entries cannot borrow one another's grants, including after a grant or attachment changes.
Capture remains synchronous in the invoking hook.

Hooks, MCP services, and Hermes adapters publish activity but never launch or keep a separate recall
process. Worker passes do not extend idle time. The process exits after the longest enabled entry idle allowance,
when all entries disable it, when its installed package is replaced, or when recall remains stuck. A missing runtime
config starts no service; a transient unreadable config receives a second check before shutdown. The default
allowance is 120 minutes for local Hermes installations and coding-client entries. An explicit `resident_recall_minutes: 0`
remains a way to disable residence when no other entry has it enabled. The worker's explicit zero stops the library
service.

Existing hook and MCP launch commands and their installed timeouts remain unchanged. Refresh the native dsh plugin
with the existing `apply-install` workflow before restarting dsh; remote entries use the remote client installation
command. A package-only upgrade leaves that copied JavaScript file unchanged. Doctor checks running package code,
but does not compare this plugin copy against the package. The background worker must be configured and running to
provide resident recall. Without it hooks retain their bounded direct recall and ledger recovery, with any unavailable
or late channel recorded in diagnostics.

## What Codex attests

The Codex MCP server binds a call to a conversation through Codex's reserved MCP thread metadata only; the model
cannot supply it as a tool argument. Without it, reads still answer and report the gap
`mcp_session_is_not_codex_conversation_id`, and `propose_memory`, `revise` and `forget` are refused.

Local Codex shared entries also require `PreModelRequest` and an actual `PreToolUse` witness. The MCP call id,
parent item and sampled history watermark must identify the latest complete retained user occurrence. Its source
must already exist in SQLite with the same verified person and pinned local account. Thread or turn metadata alone
does not authorize writes. Independent installations with unresolved human identities, remote clients, unsupported
transcripts, and clients without these hooks keep reads and ordinary capture but refuse model mutations.

`propose_memory` uses the same verified request source and adds its reference to the proposal's evidence. Its text
remains an `assistant_visible` source with a capture receipt, neither a published fact nor the owner's approval.
Facts come from consolidation: a separately configured background model reads stored sources, and every fact it
proposes cites a source reference, its revision and an exact quotation, all checked before acceptance. A proposal
without the native source binding refuses instead of storing an unassociated candidate.

A prompt that an application delivers into Codex's user box reaches the hooks as an ordinary user prompt, and nothing
in that signal tells typing from automation. It is stored as the owner's direct input and carries the owner's
authority in that session, including for a correction or deletion it asks for. The entry's `verified` principal
attests the approved account, not who pressed the keys, and there is no delegated-input origin. Keep work between
agents out of the user box: `handoff_write` records an external agent report (see Task handoffs), and pausing
recording keeps a session's input out of the store ([Private mode](private-mode.md)).

## Task handoffs

`handoff_write`, `handoff_list`, `handoff_accept` and `handoff_cancel` are available
on the Codex, Claude Code, WorkBuddy and dsh MCP servers and the Hermes provider.
They use the authenticated entry's read and write scopes. Handoffs are agent
reports attached to task episodes; they do not become facts or consolidation input.

To write a report, pass `protocol_version: "1.1"` and a `report` object:

```json
{
  "task_ref": "episode-reference",
  "progress": "Layout is ready",
  "changed_files": ["src/layout.py"],
  "uncommitted_files": ["src/layout.py"],
  "next_step": "Check the narrow viewport",
  "owner_decisions": [],
  "evidence_refs": ["source-reference@1"],
  "session_label": "sender-conversation-label",
  "ttl_days": 7
}
```

Use a visible task episode and current source references belonging to it. File
paths and progress are reports; the server does not read those files or verify
Git status. `session_label` is optional and always labelled `unverified`; it never
authorizes a write or identifies a unique handoff. The separate `sender_session` comes only from host-bound context, such as Hermes identity, hook or remote ingress context, captured session-end metadata, or reserved MCP thread metadata. Without that attestation it is null, and report-body privacy uses project and store rules rather than the display label. A store-generated reference
identifies each report. A trusted project id, or a host-supplied normalized working
directory when project ids are absent, confines delivery to the same project.
An unverified or missing project directory cannot authorize a directory-based report.

`ttl_days` accepts integers from 1 to 365 and defaults to 7. `handoff_list` returns
at most 30 records by default, up to 100 with `limit`; continue with `next_before`
as `before`. Set `include_closed: true` to include accepted, cancelled and expired
records. Every record includes its confirmation and safety state. Explicit reads
label held reports; they never release a held source's text.

Pass `ref` and `expected_revision` to accept or cancel. Only one entry can accept;
a second attempt reports `already_accepted` and the receiving entry. A stale version
is refused. Cancellation belongs to the sending entry or the local installation
owner through the management page. The owner page uses the existing preview and
confirmation operation. Reports keep their lifecycle audit in SQLite.

At session end, existing task resumes produce unconfirmed drafts. A resume
published after its session end also produces a draft. The sending entry confirms
one through `handoff_write` with `report: {"draft_ref": "handoff-reference",
"expected_revision": 1}`. Confirmation publishes the exact stored snapshot;
changed evidence requires a new report. No additional model call is made.

The session briefing includes at most three confirmed pending handoffs within
`session_brief_tokens`. Each appears as escaped JSON labelled as an external
agent report. Accepted, cancelled, expired, quarantined and pending-review reports
are excluded. Lifecycle, expiry, source visibility and injection policy are checked
again before delivery. Reads apply expiry immediately; the existing briefing
worker records expiry transitions in bounded writes. The final single-injection and session token caps include handoff text. Handoffs do not create correction-notice targets; withdrawn evidence instead prevents the report from being delivered. Inline private spans in report text are removed by the shared capture filter before storage. Turning off
`session_brief_enabled` also turns off automatic handoff delivery. No host memory
file needs to be edited.

The automatic schema upgrade preserves existing facts and keeps the upgrade
coordinator's rollback snapshot. To return to an older release, restore its
pre-upgrade snapshot using the documented restore workflow; older code does not
write the newer schema. Handoffs are local coordination records and are not
included in the portable memory interchange format.

## Data and host boundaries

SQLite truth lives in the verified Core data directory. Vector indexes are companions and may be rebuilt only through an explicit configured space. The Hermes and Codex adapters bind host identity, installation identity, scopes, and project roots before opening Core. After that binding is verified, an omitted runtime-config argument checks only `<data_directory>/runtime-config.json`; a missing file stays basic plus an explicit capability gap. The file is never generated or discovered from the current directory, parent directories, or credential locations. Optional host wiring must report a capability gap instead of creating or repairing a database.

The `codex` extra adds the MCP SDK. NumPy provides mapped retrieval and exact full-vector
scoring without SQLite extensions. The `lancedb` extra is only a reader for existing
LanceDB embeddings. Legacy `vector.backend` values map to the SQLite companion automatically.
The worker imports existing vectors without embedding calls and retains the original files.
Until all visible identities are verified, recall uses lexical channels and reports an import gap.
Missing or unreadable input remains a reported gap. Changing the embedding model still requires
an explicit re-embedding run; opening a store never authorizes paid reconstruction.
