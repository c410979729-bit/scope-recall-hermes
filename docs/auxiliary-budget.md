# Auxiliary model budgets

API requests reserve input tokens, output tokens and their configured price before sending. Successful calls
settle to valid reported usage. Unknown or invalid usage keeps the full reservation: the provider may have
executed and charged for a request even when the connection failed. Reservations count against the selected
budget period, either the installation's lifetime or its configured calendar month and time zone.

HTTP 400, 401, 402, 403, 404, 405, 413, 415, 422 and 429 explicitly refuse admission and settle to zero tokens
and zero amount. A completed HTTP worker reply proving failure during connection setup also settles to zero:
the model request has not been sent. This includes DNS, refused connections, TLS handshake failures and
connection timeouts. An incomplete exchange, a parent's timeout based on its last progress frame, HTTP 408,
any 5xx, and a successful response without valid usage remain uncertain and retain their reservations.
Call counts include refused calls and operator-released records.

## Inspect retained usage

The budget section of `scope-recall doctor` reports retained unknown reservations in the current period:
request count, input and output tokens, amount in micro USD, and fractions of the corresponding caps.
One USD equals 1,000,000 micro USD. Uncapped or zero-cap ratios are undefined. Per-model token cap shares
appear under `ledger_headroom.retained_unknown.by_model` in the JSON report.

Doctor also identifies caps that cannot admit one configured route reservation, including per-model caps,
and says that requests are being refused. Input reservations are floors: a larger serialized request can
require more headroom. Embedding requests reserve no output tokens. Retained usage is shown separately so
the operator can distinguish known usage from conservative charges.

## Explicitly release retained reservations

Use the runtime configuration for the installation that owns the auxiliary ledger. Preview first:

```sh
scope-recall release-budget-reservations --config /absolute/path/runtime-config.json --before 2026-11-09T00:00:00Z
```

The JSON receipt lists matching ledger records and totals for tokens and amount. `--before` requires an ISO
timestamp with an explicit offset; it matches request start times strictly before that instant. Selection is
limited to the current configured budget period. Preview uses a read-only connection and changes nothing.

After checking the provider's billing information, explicitly accept the possibility of provider charges:

```sh
scope-recall release-budget-reservations --config /absolute/path/runtime-config.json --before 2026-11-09T00:00:00Z --apply
```

Execution selects matching records again inside its write transaction, zeros their counted tokens and amount,
and adjusts every registered summary in that transaction. Records and original reservations remain in the
ledger with status ending in `_usage_unknown_released_by_operator`. Repeating the command does not release
them again. Successful calls, in-flight reservations, meter breaches, other periods and newer records are
excluded. A release can restore token or monetary headroom; it does not reset call caps, provider holds or
meter-breach fences. There is no automatic expiry or proportional settlement.

## CLI subscription accounting

The CLI subscription ledger uses a separate UTC daily call/token allowance and has no dollar price. Doctor
reports its retained unknown tokens and daily cap shares separately. A process that fails to spawn settles
to zero tokens. Once the process has started, an error or timeout without usage remains uncertain.

To inspect or release subscription records, add `--subscription` to either command above. This selects the
configured CLI subscription's current UTC day rather than the API budget period. It preserves call counts,
history and meter-breach fences, and does not change API records. Omit that flag to manage API billing.
