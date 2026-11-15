# A client on another machine

Claude Code or Codex on a second machine (a work computer) can be an entry of the shared store on this
one. Its memories are the store's, recalled by every entry with the owner's grants, and each recalled item
names the entry it came in through, so "Work Claude Code" stays apart from this machine's Claude Code.
WorkBuddy can be such a client as well (`--host workbuddy` below, `"host": "workbuddy"` in `client.json`); it is
forwarded and its session record read like the `claude-code` host's, and where its setup differs is said below.
dsh can also be a remote entry (`--host dsh`, `"host": "dsh"`); its native plugin sends the turn's messages
inside the Stop payload instead of reading a transcript file.
Hermes gateway, CLI and Desktop can also use a remote entry (`"host": "hermes"`); its memory provider forwards
host observations to the ordinary Hermes adapter on the store machine. See [Hermes setup](#hermes-on-the-client-machine).

Nothing of the store moves. The client machine runs a small forwarder and holds only the entry's token:
each hook goes to the entry's server here over HTTP, and this machine's handler records it, as it does for a
local client. A Claude Code client reads its own session record on its machine and sends what the record
shows being said, never the record itself; the server opens no path a request names. The entry's MCP tools
are served over streamable HTTP. Tool calls and tool output are not recorded: the remote plugin does not
forward them, where a local Codex client records its tool calls.

The server listens on one private address this machine has on a network both machines are in (a tailnet),
never on every interface, and refuses any request without the entry's token. The token is made on the client
machine and stays there; this machine keeps its SHA-256.
Plain HTTP is accepted, including non-loopback addresses: use it only over a trusted private/encrypted network
such as a tailnet, not the public internet. Use an HTTPS endpoint when that network cannot protect the token.

Replace angle-bracketed paths, IDs and addresses with your own installation values.

## On this machine

1. Attach an entry for each client, as for a local one, with its own home and a name to tell it apart:

   ```text
   scope-recall attach --host claude-code --instance-root "<entry-home>" --root "<store>" --entry work-claude-code --display-name "Work Claude Code" --grants-like all --capture-like desk
   ```

2. Record where it is served and the digest the client printed (step 3 below):

   ```text
   python -m scope_recall.adapters.codex.remote_server configure --home "<entry-home>" --host claude-code --listen "<listen-address>" --port 18765 --token-sha256 <hex>
   ```

3. Serve it from an environment that has the `codex` extra, at logon and again when it stops. `--env-file`
   names the file holding the embedding key the runtime config declares; it is read in place:

   ```text
   python -m scope_recall.adapters.codex.remote_server serve --home "<entry-home>" --host claude-code --env-file <file>
   ```

   The server keeps one handler for its prompts' recall, with its SQLite vector companion open and its embedding
   worker connected, while each request's own handler stores the hook (as the entry's MCP server on this
   machine does for the hooks here: [shared-store.md](shared-store.md)).

   The server has no console. Each hook it handles (its event, the handler's reason, the capture's error code
   when it failed, how far a record was stored, the time taken with the shares of making its handler, the
   capture, attaching the handler's runtime and closing it (the rest is the recall), and for a prompt how its
   kept recall went: `warm recall answered`, `busy`, `slow`, `late`,
   `without_vectors:<gap>` or `failed:<reason>`), each request refused for want of the token and its own
   errors go to `<home>\scope-recall\remote-server.log`, kept to about 1 MB with two older copies.

   On Windows use a verified windowless interpreter or the service controller appropriate to the host.
   A launcher named `pythonw.exe` does not by itself prove that no console will open.

4. Let the client machine reach the port: an inbound firewall rule for that port, from the client's private
   address only.

Each entry has its own port and its own server process. A server runs the installed package, so it is
stopped with the other processes on the store for an upgrade (`package-upgrade`) and started after it.
For a WorkBuddy client, name `--host workbuddy` in `attach`, `configure` and `serve`.
For dsh, use `--host dsh` in all three; the attached home is on the server, not on the client machine.

When the server reaches 4.0 before a 3.16 client, Claude Code, Codex and dsh can still use
`scope-recall pause` to stop recording that session. To resume remotely, upgrade that client to 4.0;
the server's local management page can also resume recording. Upgrade dsh clients promptly: when the
server is unreachable, the 3.16 native plugin can leave the plaintext inside `<private>` tags in its
local pending-message files. This is client-side behavior that a server upgrade cannot change.
The pause receipt gives the upgrade instruction. While full private mode is active, a 3.16 forwarder
stops a later `resume` prompt locally and shows its generic private-mode notice, so that prompt never
reaches the server to receive the more specific refusal.

## On the client machine

1. Install the package in a virtual environment. Claude Code must supply a prompt id: a prompt without one is refused. Claude Code runs a hook through a shell, so its
   interpreter and `client.json` must be on paths of ASCII letters, digits and `._-/:` only; `install`
   refuses others.
2. Write `client.json` with absolute paths:

   ```json
   {"url": "http://<listen-address>:18765", "host": "claude-code", "token_file": "C:/Users/me/.scope-recall-remote/claude-code/token", "state_dir": "C:/Users/me/.scope-recall-remote/claude-code/state"}
   ```

3. Make the token and give its digest to this machine's step 2. Only the digest is printed:

   ```text
   python -m scope_recall.adapters.codex.remote_client token --config <client.json>
   ```

4. Write the plugin:

   ```text
   python -m scope_recall.adapters.codex.remote_client install --config <client.json> --plugin-dir <dir>
   ```

   Claude Code: `<dir>` is `~/.claude/skills/scope-recall`; it loads in the next session. Codex: `<dir>` is
   `~/plugins/scope-recall-codex`, listed in the personal marketplace (`~/.agents/plugins/marketplace.json`)
   and enabled in Codex, which then asks you to approve its hooks.

   The plugin's `.mcp.json` carries the token too, as the header the host sends to `/mcp`; like the token file
   it stays in your profile.

   WorkBuddy has no plugin: `<dir>` is its own home (`%USERPROFILE%\.workbuddy`), and `install` adds the hooks for
   `UserPromptSubmit`, `Stop` and `SessionEnd` (15, 10 and 10 s) to `settings.json` there and the server
   `scope-recall`, with the token header, to `mcp.json`. Everything else in those files stays, each file it
   changes is copied to `state_dir\backups\<time>\` first, and running it again changes nothing; a new token is
   written into the same server. It refuses beside another Scope Recall hook (a local entry's or another client's),
   a `scope-recall` server that names another address, or a file that is not plain JSON. WorkBuddy runs the hooks
   through Git Bash: keep the interpreter and `client.json` on paths of printable ASCII without `"`, `$`, `` ` ``
   or `\`. Quit WorkBuddy before `install` and start it again after, then approve the MCP server `scope-recall`
   in its MCP settings, where it waits for approval.
   To take the client out, quit WorkBuddy and delete the server `scope-recall` and the three hooks that run
   `remote_client` from those files, or put back the copies from `state_dir\backups\` if nothing else changed
   there since.

### Hermes on the client machine

A gateway, CLI or Desktop profile on another machine can use this store through a remote memory provider.
Use a separate entry home on the store machine; it needs no Hermes installation. Run both ends with the same
Scope Recall package version. On the client, the catalog plugin puts the core into the environment Hermes runs
plugins in, and the remote installer switches that plugin to this entry.

#### Store machine

1. Create the entry, sharing the existing owner's read scopes and a capture scope the existing entries read:

   ```text
   scope-recall attach --host hermes --remote-profile default --instance-root "<server-entry-home>" --root "<store>" --entry cloud-hermes --display-name "Cloud Hermes" --grants-like all --capture-like desk
   ```

   `default` must match the remote Hermes profile identity; use its name for a named profile. `desk` is the existing
   local Hermes entry ID. `--grants-like all` copies effective owner read/write scopes, and `--capture-like desk` selects a
   capture scope every effective owner route can read. Owner rows without an approved principal grant nothing and
   are excluded from copying and this check. The command refuses a capture scope any effective owner route cannot read.
   For a store with no attached entries, omit both grant options. CLI and Desktop owner routes are created by
   default. A gateway's platform, user, chat and thread need their own exact rows through
   [`scope-recall grants`](shared-store.md#changing-an-entrys-grants); private and group routes stay separate.
   To use existing query model routes, append `--runtime-config-from "<existing-entry-runtime-config>"` to attach;
   otherwise recall is lexical. The store keeps its existing shared worker.

2. After the client makes its token below, record only the printed digest:

   ```text
   python -m scope_recall.adapters.codex.remote_server configure --home "<server-entry-home>" --host hermes --listen "<private-address>" --port 18765 --token-sha256 <hex>
   ```

3. Serve the entry from an environment with the `codex` extra (add `--env-file "<file>"` if its query routes need keys):

   ```text
   python -m scope_recall.adapters.codex.remote_server serve --home "<server-entry-home>" --host hermes
   ```

#### Client machine

Use a profile with no local Scope Recall store: do not run `hermes memory setup` or the local installer in it.
For a named profile, add `-p <profile>` after `hermes` in the commands below.

1. Add the plugin from the catalog. This puts the core package into the environment Hermes runs plugins in:

   ```text
   hermes plugins install scope-recall
   ```

2. Install the same Scope Recall version in a virtual environment for the commands below, and write `client.json`
   with absolute paths belonging to this client:

   ```json
   {"url":"http://<private-address>:18765","host":"hermes","token_file":"/home/me/.scope-recall-remote/hermes/token","state_dir":"/home/me/.scope-recall-remote/hermes/state"}
   ```

3. Make the token; give only its printed digest to the store machine's configuration step:

   ```text
   python -m scope_recall.adapters.codex.remote_client token --config "<client.json>"
   ```

4. Install into the profile home:

   ```text
   python -m scope_recall.adapters.codex.remote_client install --config "<client.json>" --plugin-dir "<client-hermes-home>"
   ```

   This keeps the catalog's plugin, pins it to the installed version, writes the transport binding and sets
   `config.yaml`'s `memory.provider` to `scope-recall`. Other configuration bytes are preserved; changed files are
   backed up under `state_dir/backups/`. Ambiguous YAML is refused before writing. The pin stays in `plugin.yaml`,
   where the catalog keeps it: Hermes does not rebuild a plugin whose declaration moves to a `pyproject.toml`. No
   client database is opened or created.

5. If the pinned version differs from the one the catalog installed, turn the plugin off and on so Hermes rebuilds
   the environment it runs plugins in (enabling an enabled plugin changes nothing). Then stop the profile's running
   backend, which keeps the plugin it loaded:

   ```text
   hermes plugins disable scope-recall
   hermes plugins enable scope-recall
   hermes gateway stop
   ```

   A gateway run as a service is restarted with `hermes gateway restart` instead. Start a new session; the same home
   serves the gateway, CLI and Desktop. A development build needs its wheel directory as `UV_FIND_LINKS` during
   the rebuild.

#### Data, trust and limits

The token represents the **entire entry**, not one chat. Its holder declares the platform, account, sender, chat,
profile and observation times; the server cannot independently authenticate each declaration against the remote
OS or messaging platform. A holder can declare another route already granted to that entry. Give a token only to
clients trusted with every grant of the entry, and use separate entries for separate trust boundaries.

The client sends turn identity, spoken messages, provenance markers, platform message IDs, session switches and
observation times. The server supplies its registered home and invokes the existing Hermes identity, opener,
steer, turn capture, audience and prefetch functions. Request-supplied paths are not opened. Tool invocations and
tool output are not recorded; content-free batch markers preserve the steer boundary. Native memory-file and
delegation hooks are not forwarded.

Each inline start claims its session's preceding unclaimed pre-hook, even when the host changes its text during
redaction. A start without a pre-hook still owns an independent turn record. Snapshots retain current-turn history,
opener evidence and each pre-compression checkpoint's own necessary rows; a shorter post-compression history cannot
overwrite those rows before replay. API attempts update their observation slot rather than multiplying full histories.
Current binding facts replace the gateway's accumulated switch chain. The server rebuilds from these facts after a
restart; native source keys deduplicate replay. Recording command eligibility is parsed
from the original input before private-span filtering, and the server still applies native authorization and the
remote command replay fence. Completed conversation prefixes are not uploaded for retrospective backfill.

`recall`, `inspect`, `trace`, `profile`, `entity`, `status` and `handoff_list` use `/mcp`. Mutations (`propose_memory`,
`revise`, `forget` and handoff writes or transitions) are not advertised or executed. Hermes supplies profile
context for a tool call but no independent conversation attestation equivalent to Codex's reserved thread metadata.
Use an entry that can prove the latest human authorization for explicit changes.

Queries, arguments and hook bodies have `<private>` spans removed before transmission. Every content exit first
checks recording policy using content-free metadata, batched below the endpoint's encoded-byte limit. User text
and answers retain their first-observation times across later sync, compression and session-end copies; a delayed
callback cannot redate denied text after recording resumes. Each retained historical row is checked at its original
numeric timestamp or its earlier witnessed time. Rows without verifiable times lose their bodies when policy requires it. A command exemption
carries only the recording command and routing facts. Failed checks send no body. If the client already knows
recording rules exist, new content is also not queued while checks fail; existing pending snapshots wait. Before
any rules are known, an unreachable server allows filtered snapshots to queue locally for a later check. Treat
that queue as private data. The server also checks capture policy at admission. Client clocks are not offset
estimated for Hermes; keep both machines synchronized.

A lifecycle forwarding call has a seven-second budget, with a bounded connection attempt and the existing
one-minute outage cooldown. The turn continues without recalled context during an outage. No process or scheduled
task is added on the client. A later call or an explicit flush can replay pending snapshots:

```text
python -m scope_recall.adapters.codex.remote_client flush --config "<client.json>"
```

Hermes shares the existing spool directory but coalesces snapshots by turn. Its capacity is **64 MiB of JSON**, with
oldest snapshots evicted when capacity is exceeded. The shared send lock prevents another Hermes sender from
passing a claimed queue head; this is not a total ordering of concurrent host callbacks. HTTP bodies above 512 KiB
are gzip encoded; the wire bound is 8 MiB and the decoded envelope bound is 64 MiB. Queue storage and HTTP use the
same compact UTF-8 JSON envelope before compression. Decoded oversize returns 413; malformed JSON or gzip returns 400.
A snapshot that cannot fit an encoded/decoded bound, or whose fixed metadata alone cannot fit a privacy request,
is counted in dropped_count and removed so later turns can send. Capacity evictions are counted too. Loss records
are published before deletion; the next queue operation finishes an interrupted deletion without counting it twice.
Transient connection/storage failures and retry responses retain the snapshot; an unanswered or temporary privacy-check failure retains
previously queued snapshots until verification succeeds. Malformed requests and ungranted routes settle without
being counted as capacity or size losses.

Inspect occupancy and durable loss counters (including the first loss time):

```text
python -m scope_recall.adapters.codex.remote_client status --config "<client.json>"
```

`queued_bytes / queued_turns` estimates average retained snapshot bytes. With average size `B` and total entry traffic
`R` turns/hour, the approximate retention window is `64 * 1024 * 1024 / (B * R)` hours. At 16 KiB/turn this is about
4,096 turns: 68 hours at 60 turns/hour, or 34 hours at 120 turns/hour. API attempts do not multiply the snapshot
count. Sizes vary; this is an estimate, and the policy-check restriction above can prevent new offline retention.
`dropped_count` counts capacity evictions and permanently oversized snapshots, not successful deliveries or policy
refusals. The four coding clients
keep their existing queue behavior.

To return to local mode, stop the profile, preserve or flush pending hooks, remove its remote binding and wrapper,
and follow [local Hermes setup](install/hermes.md).

### dsh on the client machine

Use the client steps above with `"host": "dsh"`. Quit every dsh profile first, then pass its existing home
(`$DSH_HOME`, normally `~/.dsh`) as `--plugin-dir`:

```text
python -m scope_recall.adapters.codex.remote_client install --config <client.json> --plugin-dir <dsh-home>
```

The installer writes `scope-recall/dsh-plugin/index.mjs` and merges two native rows into `cordis.patch.yml`:
`scope-recall` runs the Python forwarder with `remoteConfig: <absolute client.json>`; `mcp-scope-recall` uses
`@deepseek-ai/dsh-mcp-client`, `transport: streamable-http`, `<url>/mcp` and the token's Authorization header.
The plugin's `home` is the client's `state_dir` (status and pending messages only); no local store attachment,
model key or `envFile` is needed. Without `remoteConfig` the plugin still uses the same-machine hook entry.
Recall and Stop wait a bounded time (9 and 20 s by default); a failed recall does not block the turn indefinitely.

Other rows and user overrides are preserved; changed existing files are copied under `state_dir/backups/`.
Reinstall is idempotent and refreshes the MCP token after token rotation. Another entry's managed block or an
unowned Scope Recall row is refused rather than silently replaced. As with a local install, session-log upload
to dsh's model API is switched off. Keep the patch and its backups private: the MCP header contains the token.
Restart dsh and check `dsh --profile headless --dump-config` for both rows, without sharing the token-bearing dump.
To remove it, quit dsh and remove only the `SCOPE_RECALL_DSH_START`/`END` block and installed plugin file;
leave the separate privacy block in place. Do not discard pending messages unless you intend to lose them.

A remote prompt that capture admits is durably queued before recall starts. The server then commits it at once
on a thread with its own write deadline; recall does not wait for that commit. The answer reports the receipt
the commit has reached by then: `queued` means the inbox still holds the capture (the commit is running, or met a
busy store or a conflict), `persisted` that its source is stored. A server exit leaves an unfinished capture in
the inbox for replay. A repeated message that differs only in its occurrence time returns the original source
as a duplicate. Different content under the same key stays queued as `capture_conflict_pending` for re-keying.

The server's hook log includes `capture <durability>:<disposition>` and a `vector trace` object containing only
execution flags, fixed phase codes and milliseconds. `vector_started` and `vector_finished` describe actual
index searches; `vector_packet` says a finished search reached a complete packet before the hook's deadline.
A finished search can find no matches. `helper_failure_phase` distinguishes `drain` (an earlier late reply),
`open` and `search`; `helper_<phase>_ms` records each wait. `embedding_failure_phase` distinguishes `prefetch`,
`connection_slot`, `connect` and `read`, with `embedding_ms` for that failed wait. `transport` identifies a
failure before the HTTP helper published a socket phase. These fields do not change recall's stage budgets.
A capture that masked credential values adds `masked <n>`, the number of merged replacement spans; the values
themselves never reach the log.

## When the server cannot be reached

A hook whose connection has not opened in 3 s gives up and answers with nothing; for a minute after that no
hook tries, so while the server is away a prompt is held up once a minute at most, and it has no recall.
A Claude Code session's record carries what was said to that session's next Stop that reaches the server,
and the cursor on the client moves only as far as the server stored; what a session had not sent when it
ended stays unsent, and a message the store was too busy to take when its hook came (the server log says
`not stored`) is stored from the record at the next Stop. A Codex hook is written to a spool on the client
before it is sent and removed once the server has answered for it; what stays is sent, with the moment it
happened, by a process a later hook starts once the server answers again. A hook sent twice is the same source,
not two. A hook whose message the busy store could not take stays in the spool (the server log says `not stored,
to be sent again`), and one the server refuses for good (HTTP 400 or 413: malformed, too large) is dropped
rather than kept, since it would be refused again and would stop every later flush. The spool keeps 256 hooks
and drops its oldest past that. What did not get through, what the spool sent and what it dropped is logged in
the client's `state_dir\remote-client.log`. While a hook's own request is sending a spooled item, the background
sender skips that item; a failed send leaves it for the next flush.

dsh does **not** use the Codex forwarder's spool. Its native plugin keeps unacknowledged messages in
`state_dir/scope-recall/dsh-spool`; failed/empty responses do not advance its acknowledgement cursor. The next
Stop or the plugin's bounded background sweep can replay them. The plugin's retention limit applies
(5,000 messages per session or 14 days); this is not unlimited delivery or a second remote queue. Check
`state_dir/scope-recall/dsh-plugin-status.json` and the forwarder's log when capture stays pending.

A message is dated by the client's clock, the moment its hook ran there, which is what makes a hook sent again
the same source. When a request's latest time is more than a minute ahead of this machine's, every time in it
moves back by that lead: a recall finds nothing dated after its now, so a message dated a day ahead would have stayed
hidden for a day. Every request of a client session moves back by the lead its first such request found, while its
own is within a minute of that, so the session's times stay about this machine's now and its turns keep their order
whatever each request's delay on the way; moved back each by its own lead, a reply that came with a quicker `Stop`
could be dated before its prompt. A request whose lead is further off (a clock set anew, a hook sent long after it
ran) moves back by its own; a message sent again under its key keeps the time its first copy was stored with.
When it was stored, when its work falls due and a recall's now are this machine's clock, so a client clock that
runs fast or slow does not hold back the work on its messages. That covers the hook's own time and every message a
request carries: a record's lines, and the turn's messages dsh's plugin sends inside its Stop. Keep the two
machines' clocks synchronized.

The client connects to the server itself and never through a proxy: `HTTP_PROXY` or a system proxy on the
client machine is for the internet and cannot reach the tailnet address. Claude Code's and Codex's own MCP
connection to `/mcp` follows their proxy settings, so the server's address must be in the client machine's
`NO_PROXY` as an exact address rather than relying on CIDR support. WorkBuddy follows these
variables and the system's proxy, and matches `NO_PROXY` by name or address only.

## A new client machine

The entry belongs to the store, not to the machine. On the new machine repeat the client steps with a new
token, and on this machine run `configure` again with its digest and restart the server: the old token stops
working and the entry's memories stay under its name. A firewall rule that names the old machine's address
needs the new one.

## Limits

Over a relayed path a request takes one to three round trips of the network, so a slow relay adds to every
hook and a direct path adds little. A prompt's hook waits at most 15 s, as a local one does, and the server's
work on it at most the entry's `hook_processing_seconds` (6 s unless set lower); both are ceilings, and the
prompt goes on as soon as recall is done. Codex allows SessionEnd and Interrupt at most 3 s, so on a slow path
those may time out; the hook is already in the spool then, and is sent from there.
