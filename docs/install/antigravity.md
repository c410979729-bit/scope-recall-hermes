# Antigravity

Antigravity uses the same shared store, entry grants and Core as the other clients. Local hooks and MCP run in the package's Python environment; a remote client forwards to a fixed server entry and creates no Scope Recall database on the client machine.

## Local shared store

Install a package build containing this adapter with the `codex` extra in the intended Python environment, then review setup before applying it:

```text
scope-recall setup --host antigravity --home "<absolute-gemini-home>" --store "<shared-store>" --dry-run
scope-recall setup --host antigravity --home "<absolute-gemini-home>" --store "<shared-store>"
```

The default home is `~/.gemini`; `SCOPE_RECALL_ANTIGRAVITY_HOME` is a Scope Recall discovery override, not an Antigravity setting. Use `--share-with <existing-entry>` when setup asks which existing owner's scopes to share. Local use requires the entry's approved operating-system account. File access by another account grants no authority.

The installer merges only `scope-recall` in `config/hooks.json` and `mcpServers.scope-recall` in `config/mcp_config.json`. It preserves other definitions, the hook's `enabled` switch, the MCP server's `disabled` switch, JSON key order and existing BOM/line endings (changed JSON is written with two-space indentation). A conflicting entry is refused. It backs up changes through the ordinary installation receipt. It does not alter rules, Google settings or Google credentials. Reload the host's customizations and approve the MCP server as required by Antigravity.

Installed hooks are `SessionStart`, `PreInvocation`, `PostInvocation` and `Stop`. Each command explicitly names `--event`, because the native stdin contract does not contain the event name. The named hook definition contains direct handler arrays, not a Claude-style `hooks` wrapper. MCP uses `command` and `args` with stdio.

Remove this entry's host registrations without deleting the shared memories:

```text
scope-recall plan-uninstall --instance-root "<absolute-gemini-home>" --target-plugin-dir "<absolute-gemini-home>"
scope-recall apply-uninstall --instance-root "<absolute-gemini-home>" --target-plugin-dir "<absolute-gemini-home>"
```

## Remote shared store

Follow [remote entries](../remote-entries.md) to attach, configure and serve a separate entry on the store machine, using `--host antigravity` in all three commands. The server entry's grants are fixed; the client cannot supply different scopes or borrow the server's operating-system account as human authorization. Use HTTPS or an encrypted private network, not unprotected public HTTP.

On the client, create a transport configuration with absolute paths:

```json
{
  "host": "antigravity",
  "url": "https://memory.example.net",
  "token_file": "/absolute/private/antigravity/token",
  "state_dir": "/absolute/private/antigravity/state"
}
```

Create the token with the existing token command, provide only its printed SHA-256 to the server configuration, then install the host blocks:

```text
python -m scope_recall.adapters.codex.remote_client token --config "<client.json>"
python -m scope_recall.adapters.codex.remote_client install --config "<client.json>" --plugin-dir "<absolute-gemini-home>"
```

The retained module name is the shared client entry, not a separate Codex memory system. The MCP block starts a stdio shim that forwards the actual protocol to the server's Streamable HTTP endpoint. It resolves the token file on each request; no bearer credential is embedded in host JSON. Changed files are backed up under `state_dir/backups`; the command prints the exact written and backup paths.

Remote hooks remove inline private spans and mask secret-like text with Core's shared filters before sending. They check the server's privacy state before content leaves the machine; unavailable verification sends no content. There is no Antigravity payload spool. The native transcript remains the recovery source, and its disposable byte cursor advances only after the server acknowledges capture. Another `PostInvocation` or `Stop` retries the complete uncaptured prefix when the server is available.

Uninstall only this transport's blocks, preserving other host configuration:

```text
python -m scope_recall.adapters.codex.remote_client uninstall --config "<client.json>" --plugin-dir "<absolute-gemini-home>"
```

This does not delete the token, native transcripts or shared memories, and works even if the token file is already absent. The server operator can separately detach the entry.

## Capture and authority boundaries

The adapter follows the hook's explicit transcript path for the same conversation UUID. It accepts `transcript_full.jsonl` (the complete record) and `transcript.jsonl` (the abridged record) in the conversation's `.system_generated/logs` directory. The desktop and CLI can keep their conversation data in different roots even when they share the same configuration. Only `USER_INPUT` from `USER_EXPLICIT` with `DONE` status and `PLANNER_RESPONSE` from `MODEL` with `DONE` status supply visible text. The native user envelope is stripped without recording its additional metadata. Thinking, tools, generic events, delegated runs and truncated content are excluded. Capture waits at an unfinished visible step so a later hook can read its final text.

A user's source identity is the conversation UUID plus native step index, not text alone. Two independent inputs with identical words remain two sources. Automatic recall runs at the first model invocation, not at every tool continuation. `lastUserInput` must agree with a complete native user step within `initialNumSteps`; missing native evidence does not fall back to a recent stored message. Visible model replies retain their assistant origin, not human authority. Capture follows the live hook instead of importing old ledgers whose workspace or conversation parent cannot be established.

Recall is returned as `injectSteps` containing `ephemeralMessage`. It never inserts a user message, approves a tool, forces continuation or changes termination behavior. Native MCP metadata does not establish a reliable tool-to-current-user occurrence binding here, so `propose_memory`, `revise` and `forget` are not advertised and direct calls are refused. Use the existing owner-only management interface for corrections, deletion and recording controls.

To check an installation, open a signed-in Antigravity conversation and ask the `status` tool which entry it is using. For a capture test, connect an isolated test store, send a distinctive non-sensitive fact, then ask for it from a fresh conversation. Keep the answer out of the recall question so the reply shows what the store returned.
