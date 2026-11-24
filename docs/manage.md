# Manage local memories

For auxiliary model usage, diagnosis and explicit release of uncertain reservations, see
[Auxiliary model budgets](auxiliary-budget.md).

`scope-recall memory` is the local store owner's interface. The operating system account that can access the
installation can use it. It does not call a model or a network service. Agent tools keep their existing scope and
conversation-authorization checks; they do not expose this installation capability.

Select a shared store with `--root /path/to/store`. For an installation with its own store, use
`--host hermes|codex|claude-code|workbuddy|dsh|antigravity --instance-root /path/to/home`. An attached entry directs you to the shared
store root so management sees the whole installation. Examples below use a shared root.

## Local account approval

Local host owner routes require the OS account approved at installation or attachment.
The account comes from the process token on Windows or the OS UID elsewhere, never a username environment variable.
Its pin is stored once in the installation configuration, or in the selected shared entry's record.
Existing grants remain unchanged. A missing pin, unreadable account or different account grants no local owner audience.
Loading a configuration, capturing a prompt and calling an MCP tool never enroll an account.
Account enforcement requires the updated adapter; older adapter code does not enforce the optional pin.

An older unbound installation needs explicit approval from the owner's local account:

```sh
scope-recall approve-local-account --host hermes --instance-root /path/to/home --yes
scope-recall approve-local-account --host codex --instance-root /path/to/client-home --yes
```

For another attached client, select its host and home instead. This command preserves scopes and identities,
backs up the configuration under its receipts directory, and changes only the selected entry's pin in a shared store.
It refuses to replace a pin belonging to a different account. The whole-store management capability described above
remains an explicit local operator capability; it is not exposed through an agent tool.

## Correction retries

`MemoryManagement.revise` and `scope-recall memory revise --request-id ID` accept an explicit occurrence ID.
Reuse it only to resend the exact same correction with the same expected revision and value, under the same
operator account, session, principal and scopes. An exact retry returns the original action, item, source and epoch
from SQLite, including after reopening Core, and writes no additional version or source. Reusing the ID with a
different command is refused. A later target revision, deletion, private interval or withdrawn capability also
prevents replay. Each new user command needs a new ID; omitting it retains ordinary version checks.
The receipt stores the command's digest and content-free result alongside its truthful local-management audit.
It does not store another copy of the correction text.

This receipt applies to the explicit management API and CLI. The management page does not supply occurrence IDs.
Core's model revision path has no durable revision receipt; Hermes and MCP model `revise` and `forget` continue to
refuse requests without verified latest-message binding.

## Entry controls

The installation owner can list entries, inspect recent activity and change an entry's state. A local store uses
`local` as its entry ID. Shared stores use the IDs registered for their host and remote entries.

```sh
scope-recall memory entries --root /path/to/store
scope-recall memory entry ENTRY --root /path/to/store
scope-recall memory entry-state ENTRY read-only --actor OWNER --root /path/to/store
scope-recall memory entry-state ENTRY paused --actor OWNER --root /path/to/store
scope-recall memory entry-state ENTRY normal --actor OWNER --root /path/to/store
```

Read-only blocks new captures while allowing recall. Paused blocks new captures and memory reads; restoring normal
allows both again. Changes are recorded with the operator and time. Existing memories are retained. The management
page provides the same controls and doctor reports each entry's state.

## Recall history

**Recall history** in the management page lists each attempted recall by session or request ID, including empty
results. It shows delivered references and versions, overflow excerpt rows, channel scores and ranks, and fixed
reasons for candidates that did not reach the packet. Open an available item to use the existing correction,
mute and delete controls; those controls keep their normal revision and owner checks. A delivered packet records
what the core released, not an acknowledgement that the host displayed or adopted it.

```sh
scope-recall memory recalls --root /path/to/store --session SESSION --json
scope-recall memory recalls --root /path/to/store --request-id REQUEST --json
scope-recall doctor --host codex --instance-root /path/to/home --recall-request-id REQUEST
```

The `inspect` tool accepts `recall-request:REQUEST` within the asking session and entry, or the packet's
`recall-diag:` reference. The management page and local commands use the installation owner's capability.
References are checked again against current scope, project, source versions, deletion, mute, privacy and
injection holds. A reference that fails those checks has no content or action link. Each item keeps its object kind and retrieval-channel ranks. `source_type` additionally distinguishes
`session_summary`, `task_summary`, `adopted_decision` and `observation` from ordinary claims. `summary_covered` means the
summary was omitted because the final packet already displays its quotes in originals or a same-source decision;
exploratory packet fits do not create that reason. `pending_review` distinguishes incomplete injection screening
from `quarantine`, and overflow rows retain the original source reference, revision and retrieval channel.
History does not copy
conversation or fact text. A captured prompt is represented by its source references; otherwise only its length is stored. Neither full nor read-only private mode records attempts.

`recall.sqlite3` is a disposable sibling of `memory.sqlite3`, outside the truth schema and memory exports. The
bounded asynchronous writer drops a record when its queue is full or the log cannot be written, while recall
continues. Doctor reports cumulative dropped and failed writes; a completely full disk can also prevent the
health receipt from being saved. A crash or expired retention can therefore leave a gap in diagnostic history.

The UTC calendar retention defaults to 30 days, with a total log ceiling of 32 MiB and 128 queued attempts.
Runtime settings `recall_log_days` (1-365), `recall_log_bytes` (131072-1073741824) and `recall_log_queue` (1-4096)
set the initial limits. Each log persists one shared retention and capacity policy; another process cannot replace
it by writing with different settings. To change an existing log's policy, stop its writers, archive that disposable
log and start a fresh one with the new settings; memory truth is unaffected. The minimum is 128 KiB because WAL,
shared-memory blocks and health metadata count toward the ceiling. Explicit smaller settings must be increased.
The database reserves a complete WAL image and shared-memory blocks, checkpoints before accepting a new batch,
and drops a write when a reader prevents that checkpoint. A bounded shutdown stops new submissions, drains or
discards the queue and interrupts active SQL; WAL recovery preserves a consistent database after process loss.
Writes and history/doctor access expire old days and remove the oldest records first when full. An installation
with no running process performs expiry when next accessed. An older log too large to leave the WAL reserve
remains readable but refuses new writes; archiving that disposable log starts a fresh bounded stream.
Known unbound host sessions use the same core log without entering retrieval. Blank input and source-fence
refusals also receive records. Privacy metadata reads on these exits are capped at 50 ms; unreadable privacy
state leaves no record. Adapters translate trusted identity metadata and keep no parallel log.
Existing stores need no truth migration or historical backfill: earlier unrecorded recalls cannot be recovered.

An empty result distinguishes an unbound scope, no retrievable evidence, threshold rejection and delivery limits.
Capture references and processing counts are diagnostic evidence: pending work or a paused budget does not
by itself prove it caused a retrieval miss. Queue counts come from the last bounded worker-published snapshot, including its observation time; recall never
scans the work queue. Missing or malformed snapshots make those fields unavailable. A leased item does not prove
that a worker is running: liveness remains unknown unless the runtime probe supplies an observation. The writer
samples that probe outside the recall path. Doctor and command-line history never upgrade the truth store;
an unfinished upgrade makes history temporarily unavailable.

## Browse and inspect

```sh
scope-recall memory list --root /path/to/store
scope-recall memory search "release checklist" --root /path/to/store --kind claim
scope-recall memory show claim-REF --root /path/to/store
scope-recall memory history claim-REF --root /path/to/store --limit 20
```

Replace the object reference with one returned by `list`. Kinds are `event` (original source), `claim` (derived fact),
`episode`, `artifact` and `reference`. List and search accept `--kind`, `--scope`, `--agent`, `--session`, `--since` and
`--until`. The agent filter matches an entry ID, its display name or host; a store of its own also accepts its bound
agent ID. Time filters use source occurrence time, or recording time when occurrence time is unknown, and include
both endpoints. Supply an ISO 8601 timestamp with a timezone, for example `2026-01-01T00:00:00Z`.

Search is a literal substring match, case-insensitive for the rendered text. It sends nothing to an embedding
provider. Each call continues until it finds `--limit` matches, examines the remaining inventory or reaches a
30-second time budget. The budget bounds scanning after the inventory is ordered and is checked between hydration
batches. Inventory ordering includes first-revision lookups, timestamp normalization and sorting; it runs before
the first budget check, repeats for every continuation and takes longer as the number of objects grows. Each call
completes at least one small hydration batch before a budget stop, even if ordering has used the budget.
`--limit` accepts 1 to 100 items. List uses the same scan and order: newest first, by an event's first stored readable
revision's occurrence time (recording time when unknown), or by the recording time of another object's first revision.
Object references break timestamp ties. Later revisions keep their position.

Tables and JSON report how many object identities this call examined, whether the scan is complete, and whether it
stopped at the match limit or time budget. An empty result with more remaining says so and displays a continuation
token. JSON exposes `examined`, `scan_complete`, `stop_reason` and `next_cursor`. The count
includes identities rejected by source filters or visibility checks; it is not a count of text matches. Supply the
exact next cursor with `--cursor` and the same command, target and filters. The opaque token uses URL-safe characters
and needs no shell quoting. Continue until the cursor is absent to exhaust a search.

Listings cover every project and branch in the owner's scopes. Each call reads one SQLite snapshot. Captures during
a call or between pages do not invalidate the cursor. Objects
stored after a listing began may be missing from later pages; start again without a cursor to include them. Deletions
remove objects between pages. Blocking an event's earliest readable revision can move that event once; any remaining
readable revision stays listable. Read-blocked objects are excluded in each call's snapshot. Muted objects remain
visible to the owner. A cursor for another installation, owner scope or filter set is refused. History uses `--before` with the returned next
revision instead. `show --revision N` displays a particular version.

Default listings are tables. `--json` emits `scope-recall-management/1`, defined by
[`memory_management.schema.json`](../contracts/memory_management.schema.json). Success has `status: "ok"` and `data`;
a refused operation has `status: "error"`, `code` and `field`, and exits with code 2.

## Find stored credentials

Capture masks detected credential values with `[REDACTED_SECRET]` before storage, indexing or splitting a long
message. Text outside a masked span stays as it was. What counts as a credential is decided by the same scanner
that withholds recall results and checks model requests. An unquoted labelled value is masked through the end of
its line or its structure item (`,` `;` `)` `]` `}`), including any prose after it. A quoted value is masked up
to its closing quote, which stays, when the rest of the line starts with a delimiter, or holds no other quote and
starts with a word set off by a space or with a sentence's end (`。`, `、`, or `.`, `!` or `?` before a space and
a word, or before the line's end). Otherwise (another literal, `+` or `.` concatenation, a markup mark, a word
joined to the quote) the mask runs through the line's end, closing quote included. A literal opened with three
quotes is masked across lines from its first quote to its last and reads as one quoted marker. Key-shaped tokens
are masked over their matched spans. Original Unicode characters are covered by the normalized scan's exact map.

`masked_secret_count` in the prepared capture and capture receipt counts merged replacement spans across all
stored text fields of the original envelope, once before segmentation. An attempt prepared afresh reports its
own count; one that reuses an inbox row, and inbox replay, report that row's count. Hook diagnostics report the
count across the hook's captures, and a remote server's hook log shows `masked <n>`; serialized capture results
also include the count.
No diagnostic includes matched text. Masked captures carry `secret_values_masked` in their gaps and a complete
capture becomes partial.

Capture refuses with `plaintext_secret_rejected` when identity or reference fields contain a secret, including
event keys and session or scope identifiers; those fields are never rewritten. It also refuses when a text field
would become only the marker, the masked result scans as a secret, a private-key block is detected, a literal
opened with three quotes has no closing three, or an inexact character map or a value that may continue on the
next line requires withholding the whole text. Corrections refuse secret values so the caller can rephrase them.

```sh
scope-recall memory secrets --root /path/to/store
scope-recall memory secrets --root /path/to/store --json
```

This read-only command scans readable stored versions, including historical and muted objects, across the owner's
scopes. It lists `ref`, `kind`, `revision`, `origin`, `entry`, `session`, `time` and the names of matched patterns.
Chinese credential labels, camelCase names and Markdown English labels share one value rule: after trailing
punctuation is removed, a printable ASCII value needs at least four characters with a digit or an inner symbol,
or, after a password label (password, passwd, 密码, 口令, 授权码), at least eight letters. A value may begin with
one of `!@#$%^&*+=~?`; other punctuation right after a label belongs to the surrounding text (a list, a link, a
path, JSON). Dotted references, names with underscores and no digits, and camelCase or PascalCase names without
digits are excluded. Chinese text after a Chinese label, or after 是 or 为 behind any label but a password label,
is prose; Chinese-character passwords after Chinese labels pass this scan. Quoted values end at their closing
quote, respecting escapes.
Neither output format includes content or matched text. `rows_read` counts all versions read in SQL windows;
`candidates_scanned` counts those selected by the SQL credential-word prefilter for the full scan. Each batch ends
its read transaction before scanning content. The command uses no model or embedding provider and changes nothing.
Captures and revisions committed behind a batch's cursor require another run.

After upgrading, run this command, review the references and use `scope-recall memory forget <refs>` to remove the
findings through the normal preview and confirmation. Change any credential found, because it was already stored
and may have been sent to a model provider. Agent recall and tools withhold secret-bearing items with a
`secrets_withheld:N` gap. Profile and entity views report partial coverage when they withhold content.
The owner's local page and memory commands keep originals visible for review and deletion.

## Correct a derived fact

```sh
scope-recall memory revise claim-REF --root /path/to/store --expected-revision 3 --value "green"
scope-recall memory revise claim-REF --root /path/to/store --expected-revision 4 --retract
```

The expected revision is mandatory. A stale revision is refused without writing anything. Correction changes the
literal value and starts the new version at the correction time; the claim's subject, predicate and conditions
remain its identity. Earlier versions and original conversation text remain available in history. Original source
text cannot be edited through this interface.

A correction has a host-generated source labeled `local_management`, plus a separate audit action whose actor is
`local_user_management`. It is not recorded as a chat message from a person. The correction's text stays in ordinary
source storage, governed by the same deletion and visibility rules; the audit carries object identities and versions.

## Mute, unmute and delete

```sh
scope-recall memory mute claim-REF --root /path/to/store
scope-recall memory unmute claim-REF --root /path/to/store
scope-recall memory forget claim-REF --root /path/to/store
scope-recall memory forget-session --root /path/to/store --session SESSION-ID
scope-recall memory forget-session --root /path/to/store --since 2026-01-01T00:00:00Z --until 2026-01-02T00:00:00Z
```

These commands first display a dependency preview: the selected versions, affected objects, source records and
entries. Deleting a fact also deletes its source group and dependent facts, episodes and attachments, because leaving
the source readable could return the same content on the next recall. This is whole-source deletion, not sentence
redaction. Check the complete preview before confirming. Narrow selections larger than 2,000 affected objects.
Session selection and its preview share one read snapshot. Selection is never cut short by a time budget: it must
finish so a deletion cannot silently leave matching sources behind. A capture during selection makes application
refuse the stale preview; repeat the command to include that capture.

Repeat the command with `--yes` to apply. Include `--preview-id VALUE` to require the exact earlier preview; a change
to the store then causes a refusal. With `--json`, an applied response includes both the preview and the receipt.
Multiple references can be selected together; `--expected-revision N` additionally binds a single selected reference.

Mute stops automatic recall while keeping explicit inspection available. Unmute releases the displayed suppression
closure; it never restores deleted content. Delete immediately blocks reads and enqueues the existing physical purge.
Its receipt describes each layer. `read_blocked: true` alone does not mean physical cleanup finished; consult the
deletion operation until its active-content and history layers report completion. Host transcripts and old backup
files are outside that active-content receipt.

## Migration and rollback

The first open of a supported older store migrates it transactionally to the current schema. The added tables contain local
management action identities, actor, action, target versions, timestamp and memory epoch, content-free suppression
constraints for records that can arrive after a restore, and private intervals that hold only target, session or
project path, mode and UTC boundaries. Entity judgement receipts, source-screening policies, holds and resumable scan
checkpoints share the same upgrade transaction. Database identity binds capture authorization and scan positions to one
store lifetime; existing databases receive an identity without an inferred creation time. Original source rows are preserved; the `project_root` column is metadata
and existing rows keep NULL. Earlier executables refuse this newer schema; see
[private-mode.md](private-mode.md#migration-and-rollback) for the rollback consequence.

Before installing the new executable, stop the installation's writers and take a `scope-recall backup` SQLite
snapshot; keep the old executable and configuration alongside it. To roll back, use the snapshot restore commands
below with that snapshot and a separate backup directory. The newer executable
first records the latest deletion ledger and closes admission; the executable matching the snapshot then completes
  the restore. Do not overwrite the database file by hand: that can resurrect deleted memories. The documented
  [cold rollback exception](backup-restore.md#returning-from-40-to-316-with-extra-registered-scopes) has that risk and requires its stated precautions. Corrections made after
the snapshot are outside its recovery point. A snapshot taken while content was muted can restore that older mute
when completed by an older executable; use the management-capable executable to preserve later unmute decisions.

For a shared store, preview and then apply snapshot restore with absolute paths:

```sh
scope-recall restore --root /path/to/store --snapshot /path/to/pre-upgrade.sqlite3
scope-recall restore --root /path/to/store --snapshot /path/to/pre-upgrade.sqlite3 --backup-dir /path/to/new-rollback-backups --apply
```

For a store of its own, replace `--root` with `--host hermes --instance-root /path/to/home`, or the appropriate
host selector. A cross-schema apply reports `prepared` after saving the ledger and closing ordinary admission.
Install the release that reads the snapshot and run the same apply command to finish. Keep every host and worker
stopped between those commands. The shared-scope limitation above still applies when the finishing release is 3.16.

For management-capable snapshots, restore uses a complete suppression checkpoint, including records that inherited
a mute. For a snapshot predating management, it prepares the original deletion-ledger format that the older
executable understands. Both paths preserve deletions; only the complete checkpoint can release an older mute.


## Review held sources

Injection scanning withholds instructions before recall or extraction. A matching instruction inside an existing
quotation or code span is ambiguous: it can be a harmless example or an instruction copied from somewhere else.
Such hits become `pending_review`; unquoted blocking hits become `quarantined`. Neither state participates in
recall or background extraction. The local page's **Held content** view shows the text and matched rules, and its
Review release action binds confirmation to the displayed revision and content SHA-256.

`doctor` reports both totals and `injection_review`: pending versions counted by matched rule and verified source
origin, `oldest_scanned_at`, and `last_24h` (currently pending versions scanned in the last 24 hours, not a historical
arrival count). A source can match several rules, so rule counts can exceed the pending total. These are on-demand
diagnostics; host-side
notifications are not provided. Holds have no expiration policy: time alone neither releases them nor converts a
pending review into quarantine.

For a batch, first save and inspect the JSON preview, then explicitly apply the saved selection:

```sh
scope-recall review-sources --root /path/to/store --rule quoted_instruction --origin human_direct --before 2026-11-01T00:00:00Z --limit 100 > source-review.json
scope-recall review-sources --root /path/to/store --preview source-review.json --apply
```

The default `--state pending_review` is the only supported state. `--rule` and `--origin` may each be repeated and
match any supplied value within their respective filter. Origins use recorded provenance, including the verified
original origin for imports. `--after` and `--before` are exclusive bounds on the scan time, require an ISO 8601
timestamp with timezone, and can be combined. `--limit` selects at most 1,000 versions (100 by default), oldest
first. The preview reports the selected count, rule and origin distributions, exact versions and content hashes,
and excerpts of at most 240 characters each. `more: true` means matching versions remain beyond this batch.
The preview file is trusted local input, not a signed or tamper-evident receipt. Execution rechecks each listed
target's permissions, state and content hash, but a replacement authorized pending source with its correct hash
can also be executed; excerpts and counts do not authenticate the selection. Keep the file on the local machine,
protect it from edits and delete it after review.

To confirm continued withholding, create a preview with `--decision retain` and apply that preview. This completes
the existing rejection review and changes those pending versions to `quarantined`. The default decision is
`release`. A retain preview selects only current versions, as required by the native rejection flow; release can
also select older held versions. Applying requires `--apply --preview FILE`; selection filters cannot replace the
saved preview during
confirmation. Every version uses the existing owner-authorized release or rejection path. Changed content or a new
quarantine fails that item while the rest continue. The result counts `succeeded`, `failed` and unchanged completed
decisions `skipped`; failures include references and error codes. Repeating an unchanged completed decision is safe.
A release returns that version to normal admission and work scheduling; any other hold on its message group still
withholds the group. To work through more batches, create a fresh preview after each completed batch.

### Optional owner quote exception

The per-store `auto_release_owner_quotes` setting is off by default. Inspect it, then explicitly enable it:

```sh
scope-recall injection-policy --root /path/to/store
scope-recall injection-policy --root /path/to/store --auto-release-owner-quotes on --apply
scope-recall injection-policy --root /path/to/store --auto-release-owner-quotes off --apply
```

The setting lives in the store's existing `injection_rules` table as a reserved origin-policy record; it is not a
runtime configuration field. It applies across entries using that store. Enabling it does not revisit completed
holds; use the batch command for the existing review queue. Disabling it affects future scans and does not revoke
existing releases.

Every entry reading the store needs a scanner that understands the setting while it is enabled. Disable the
exception before reverting to an earlier scanner: it treats the unrecognized enabled policy record as an invalid
policy and withholds new content. No schema migration is required for this setting.

Only a complete successful scan consisting entirely of quoted instruction hits can use the exception. Every part
must be a recorded `human_direct` message whose verified human speaker the host's trusted owner binding identifies
as this store's owner. Another verified person, a missing speaker or owner binding, and every import stay pending;
an import's original human category does not prove local ownership. A message's claim to be the owner confers no
authority. When a host or worker cannot supply owner proof, it cannot use the exception: a store that is not
attached through a shared entry has no owner binding to check, and a background worker that finishes a scan on its
own has none either, so the same quoted message stays pending there. Assistant messages, tool
output and external documents keep their review holds. Any unquoted hit, blocking rule, invalid policy,
normalization limit or unfinished scan still withholds the source. Quote recognition is unchanged.

Automatic releases retain a `source_safety` row with state `released`, the matched rules and the
`auto_release_owner_quotes` reason, so the owner can inspect them later. They are policy decisions, without a
human review timestamp. The held-content review flow does not re-quarantine released versions; ordinary mute and
delete controls remain available. The exception trusts the owner's choice to retain their quoted examples; it
does not certify quoted content as safe instructions.

## Review observations

The Review page lists unconfirmed cross-session observations. The held-content page also presents their induction review controls.
Use **Confirm observation** to record review, or open the fact detail to edit the body or delete it. Confirmation
keeps the `derived_summary` basis and the original multi-session evidence. A source withdrawal still invalidates
an observation after confirmation. Existing `memory show`, `memory revise` and deletion commands handle these
claim references. Model-driven `revise` requires a host-verified latest-message binding (`request_message_binding`);
when unavailable, use local memory management for explicit edits and retractions.

## Correct source provenance

`memory origin` corrects the attribution of stored source versions. It previews by default; `--yes` applies
all displayed consequences in one transaction. Use `--preview-id` to require the exact earlier preview.
Bare event references select their current version; `EVENT@REVISION` selects that exact content version.

```sh
scope-recall memory origin EVENT@1 --origin host_generated --reason "Host notification" --root /path/to/store --json
scope-recall memory origin EVENT@1 --origin host_generated --reason "Host notification" --root /path/to/store --yes --preview-id HASH
scope-recall memory history EVENT --root /path/to/store --json
scope-recall memory origin EVENT@1 --origin human_direct --reason "Undo attribution correction" --root /path/to/store --yes
scope-recall memory origin-candidates --root /path/to/store --json
scope-recall memory origin-candidates --root /path/to/store --after SOURCE_ID --limit 100 --snippet
```

Targets are `host_generated` or `origin_unknown`. `human_direct` is available only for a source with a recorded
correction from human attribution. The source's content revision stays fixed: changing it would invalidate
existing evidence references, including mixed evidence that must remain intact. `origin_history` in `show` and
`history` records provenance revisions, actor, time, previous attribution and reason separately. Restoring human
attribution does not restore withdrawn facts or cancelled work; review those decisions separately.

The preview lists each affected claim and all its versions, including revision, state, validity interval and
selected versus other evidence roots. Main and auxiliary references follow derived source lineage. A claim is
automatically retracted only when its head is not already retracted, every unretracted version relies entirely
on the selected sources, and all those versions have exactly the head's validity interval. This appends one
ordinary head retraction with unchanged payload. History and earlier `known_at` knowledge remain available.
Mixed evidence is retained as `mixed_unchanged`. Differing validity intervals, sole-source versions alongside
versions supported elsewhere, legacy denial heads, and facts migrated from a legacy store, whose evidence exists
only as links, are listed as `review`; their claim histories stay intact.
Review those cases with `memory revise --retract`, `memory mute` or `memory forget`, checking each command's
preview and scope before applying. Observations and conversation summaries remain `source_view_unchanged`;
already withdrawn claims remain `historical_unchanged`.

The preview also enumerates queued extraction and candidate evaluation work to invalidate. Pending corrections
become obsolete; affected adoption bindings return to review and sole-source resolved references become
unresolved. Content, source indices and episode membership remain intact; the memory epoch advances so later
recall and injection read current attribution.
In-flight derivations check attribution again before accepting results. An injection already delivered before
the transaction cannot be recalled. Deleted or unavailable sources remain behind their existing visibility
boundaries; this command does not reconstruct deleted content or reverse completed deletion operations.

`origin-candidates` and provenance previews read in ordinary read-only transactions, including committed rows
in the WAL. They do not change database rows, initialize or migrate the store. SQLite may create or update
`-wal` and `-shm` files beside the database, as it can for any reader. To preserve a backup's files unchanged,
run these commands on its copy. With `--yes`, the command previews first, then recomputes and compares the
preview under the write lock before applying.

`origin-candidates` is paged by source ID, and omits bodies unless `--snippet` is given (120 characters).
It reuses the Hermes adapter's marker recognition and platform-ID precedence, and also recognizes the legacy
import review prompt. Historical envelopes that did not retain platform message IDs cannot distinguish a host
prefix from a person's pasted prefix. Treat matches without IDs as review candidates and check the original
conversation before correcting attribution. The command does not inspect any host installation's conversation files.
