# Cherry Studio

Cherry Studio can look up Scope Recall memories through a local MCP server. Each connection uses its own
shared-store entry and the OS account that attached it. Cherry Studio conversations are not captured.

## Attach a read-only entry

Install the package with its `codex` extra in the Python environment Cherry Studio will launch.
For a development checkout, run `python -m pip install -U ".[codex]"` from the repository root.
Run `python -c "import sys; print(sys.executable)"` in that environment to get the executable path.

Use `scope-recall entries --root "<absolute-store>"` to find the entry whose owner may share its readable
scopes with Cherry Studio. Give Cherry Studio a separate entry directory outside the store.
Run these commands as the OS account that runs Cherry Studio:

```text
scope-recall attach --host cherry-studio --instance-root "<absolute-cherry-entry>" --root "<absolute-store>" --entry cherry-studio --display-name "Cherry Studio" --grants-like "<source-entry-id>"
scope-recall grants --instance-root "<absolute-cherry-entry>"
```

Check the returned allowed scopes. The writable list and capture target are empty. To start with an empty store,
run `scope-recall init-shared --root "<absolute-store>"`, then attach without `--grants-like`. The new entry
can search the owner's scope once another authorized host records something there.

## Connect in Cherry Studio

Open **Settings → MCP → MCP Servers → Add → Import from JSON**. Paste the configuration below, replacing both
paths with absolute paths on this machine. Windows paths can use forward slashes, including in paths with spaces.

```json
{
  "mcpServers": {
    "scope-recall": {
      "type": "stdio",
      "command": "<absolute-python-executable>",
      "args": ["-I", "-B", "-m", "scope_recall.adapters.codex.mcp_entry",
               "--host", "cherry-studio", "--home", "<absolute-cherry-entry>"]
    }
  }
}
```

Enable the server and open its tool list. Enable it for the intended Agent under **Work → Edit → MCP**.
The connection offers `status`, `recall`, `inspect`, `profile`, `entity`, `trace` and `handoff_list`.
Ask the Agent to call `status` and check the entry ID, then ask it to recall a phrase already recorded in an
approved scope. For a direct tool call, use `recall` with `{"protocol_version":"1.1","query":"your phrase","mode":"history","max_items":5}`.
See [Cherry Studio's MCP settings](https://www.cherryai.com/docs/en/advanced-basic/extensions/mcp/).

To revoke this connection, disable the server in Cherry Studio so its stdio process exits, then run
`scope-recall detach --instance-root "<absolute-cherry-entry>"`. Future connections require a new attachment.
