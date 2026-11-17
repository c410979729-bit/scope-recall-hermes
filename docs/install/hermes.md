# Hermes

## Remote store

Hermes gateway, CLI and Desktop can use a store on another machine. Follow
[the remote-entry commands](../remote-entries.md#hermes-on-the-client-machine): on the store machine,
`scope-recall attach --host hermes --remote-profile default` creates the entry without a Hermes install;
`--grants-like all --capture-like <existing-entry>` shares effective owner scopes in that command; unverified owner rows are excluded. Configure and serve
it through `remote_server` with `--host hermes`. CLI and Desktop owner routes are included; gateway users and
chats require exact audience rows through `scope-recall grants`.

On the client, add the plugin from the catalog (`hermes plugins install scope-recall`) without running setup, so the
environment Hermes runs plugins in has the core. Then, from a virtual environment with the same Scope Recall version,
make the token and run `remote_client install --config <client.json> --plugin-dir <client-home>`. Installation keeps
the catalog's plugin, pins it to that version, writes the transport binding and selects `memory.provider:
scope-recall` in `config.yaml`. Other bytes are preserved and changed files are backed up. If the pinned version
differs from the catalog's, turn the plugin off and on so Hermes rebuilds its environment. Stop the profile's running
backend (`hermes gateway stop`, or restart a gateway service) before starting a new session: a running backend keeps
the plugin it loaded. Do not run local setup or `hermes memory setup` in this remote profile. An existing local store
binding is refused. Remote mode opens and creates no client database.

The token represents the entire entry, including every granted chat, rather than independently authenticating
individual senders. Read tools and automatic recall use the existing HTTP service; mutations are unavailable because
Hermes' tool context does not independently attest the conversation and latest human authorization. Tool calls,
tool output, memory-file and delegation hooks are not recorded remotely.

Snapshots coalesce per turn in the existing spool, bounded to 64 MiB. Later lifecycle calls or `remote_client flush`
replay them without adding a background process. Pre-compression checkpoints keep their necessary rows until replay,
even when later history is shorter. `remote_client status` reports byte occupancy, capacity/size losses and the first
loss time. Permanently oversized snapshots are counted and removed; transient failures retain pending snapshots. For average snapshot size B bytes and R turns/hour, estimate the retention window as
`64 * 1024 * 1024 / (B * R)` hours. This is finite buffering. Once recording rules are known, a failed policy check
also prevents new content from being queued; no content is sent until a metadata-only check succeeds. See the
[remote limits and trust boundary](../remote-entries.md#data-trust-and-limits).

## Plugin catalog

For a published Scope Recall entry in the Hermes plugin catalog:

```sh
hermes plugins install scope-recall
hermes memory setup
```

Select `scope-recall`, then start a new session. The provider's setup hook uses the same local setup workflow as
`scope-recall setup --host hermes`: it creates `~/.scope-recall/shared` or reuses the home's existing attachment,
installs the memory skill, and selects `memory.provider`. A new entry gives the owner's memory to the CLI and to
Hermes Desktop on this machine ([local surfaces](#local-surfaces)). New installations use original-message retrieval
with model routes disabled and a zero budget. Setup preserves the other configuration bytes and leaves the
catalog-installed plugin directory under Hermes's control. Unsupported YAML layouts are refused with a reason.

## Package installation

1. Install the package in the intended Python environment: `python -m pip install -U "hermes-scope-recall[codex]"`.
2. Run `scope-recall setup --host hermes`, review its plan and confirm.
3. Start a new host session and complete any approval shown by the host.

Discovery uses `HERMES_HOME`, otherwise `~/.hermes`. A host-local venv is preferred; setup shows the exact interpreter
and whether it fell back to its own Python. Use `--home` or `--python` only to override that choice.

Setup selects memory.provider: scope-recall in config.yaml and installs the provider and memory skill.
Both the Hermes setup hook and standalone setup use an exact scalar edit so other configuration bytes, including
comments, quoting and indentation, are preserved. Hermes' configuration writers serialize the YAML document.

A scheduled session (`platform=cron`) with no user id does not participate in memory. Its hooks return without capture,
recall or background work, and diagnostics report `capability_gap:unbound_session`. The reason is logged once at debug
level, not as a failure on every hook. No installation owner or earlier session's user is substituted; other identity
and installation failures still raise their normal errors.

After an idle period, binding a session warms the query embedding route in the background. This request counts toward the auxiliary budget; private mode blocks the warm-up.

## Local surfaces

A new entry made by setup, from the catalog or from the package, gives the owner's private memory to two surfaces:
the CLI and the Hermes Desktop chat panel on this machine. The chat panel reaches the agent only through the Desktop
app's own loopback backend or the gateway's local session transport, which serves only the account the gateway runs
as. A session that carries a dashboard login is a user of its own and binds nothing until that login is approved.

`hermes --tui` is not approved, and setup asks no question about it: the browser dashboard's chat tab runs the TUI,
and a dashboard on loopback can serve that tab without a login to anything that reaches its port, a tunnel included.
If nobody else can reach it, approve the surface. On an entry of a shared store, add the owner principal
`{"platform": "tui", "user_id": "local"}` and its `owner_private` row with `scope-recall grants`
([changing an entry's grants](../shared-store.md#changing-an-entrys-grants)); on a home with its own store, run
`apply-install --local-platform tui`. Approve a dashboard login of your own the same way, with the login in place of
`local`, or on a home with its own store `apply-install --owner-login <platform>=<login>`.

Running setup again keeps an existing entry's grants, so an entry made by an earlier release does not gain Desktop
that way: add `desktop` as above. Until a surface is approved, its sessions capture and recall nothing; the host log
says so once per session, and `doctor` reports `local_session_unbound` with what approves it.

## Model credentials

Model keys belong in this Hermes profile's own `.env` or in a secret source managed by Hermes. The in-process
plugin resolves each route's `credential_env` through `agent.secret_scope.get_secret`, which honours the active
profile. It does not read `.env` or other credential files itself. The plugin adds only the declared route keys
to each background worker's selected environment. Workers retain the operating-system, Python, proxy and certificate
settings described in the [installation guide](../install.md), plus plugin settings and declared route credentials;
other ambient Hermes keys are excluded unless explicitly requested through `worker_environment_names`.
This applies to the `launch_worker` chain; direct and manual commands use their launcher's environment. Proxy
credentials, the CLI login location and plugin-prefixed variables retain the exceptions described in that guide.
Missing keys pause the affected work types; later
passes recheck missing names at most once every 60 seconds. Each missing name is logged once per process.

Hook callbacks read `plugins.hook_callback_timeout` through Hermes' public `load_config_readonly`, falling back to
`load_config` on hosts without that function: the default is 30 seconds, zero disables the deadline, and the maximum
is 600 seconds. If neither API is available or the setting is invalid, callbacks use the default. A NaN setting
also uses the default, even on hosts whose own timeout resolver returns NaN.

## Package-managed plugin environments

Some Hermes installations build a separate environment from the declarations in the home's `plugins` directory.
Updating the interpreter that runs the `hermes` command does not update that environment. Run
`scope-recall doctor --host hermes --instance-root /path/to/hermes-home` and inspect `hermes_runtime` for its
recorded environment, package version and NumPy availability. An unreadable environment remains unverified.

After stopping the gateway and workers and backing up their configuration and store, refresh the wrapper with
`plan-install` and `apply-install`, selecting the home's `plugins/scope-recall` directory. For wrappers it manages,
the local installer preserves the dependency declaration form, as the remote installer does: an existing
`pyproject.toml` keeps the pin there; a directory without one keeps the pin in `plugin.yaml`'s `pip_dependencies`.
Fresh wrappers use the manifest form, including for development builds. Do not add or remove `pyproject.toml`
to switch forms: Hermes's lock can still refer to the previous plugin member and fail with `Distribution not found`,
then disable the plugin. Catalog-managed plugin files remain under Hermes's control; update them through Hermes.

Rebuild the plugin environment by disabling the plugin and enabling it again. Enabling an already enabled plugin
does not rebuild it. If the pinned version is unavailable
from the package index, keep a directory containing its exact wheel available as `UV_FIND_LINKS` during every
rebuild, including after `hermes update`. Keep that wheel directory. For example, from a POSIX shell:

```sh
HERMES_HOME=/path/to/hermes-home hermes plugins disable scope-recall
HERMES_HOME=/path/to/hermes-home UV_FIND_LINKS=/path/to/wheels hermes plugins enable scope-recall
```

Use the intended profile's Hermes command. On Windows, set `HERMES_HOME` and `UV_FIND_LINKS` in the
invoking process and run `hermes plugins disable scope-recall`, then `hermes plugins enable scope-recall` from
PowerShell. Check that the plugin is enabled;
an environment sync reporting success alone does not prove that. Stop any remaining backend with
`hermes gateway stop` (or restart its gateway service), then start a new session and verify doctor and a real host
recall. The
[package-managed upgrade procedure](../../maintenance/AGENT_WORKFLOW.md#12-a-hermes-that-builds-its-own-plugin-environment)
gives the sync command and failure recovery steps.

[Common installation options, models, worker modes and verification](../install.md) apply to this host.
