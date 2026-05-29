# Local append-only WAL is the session source of truth

Each annotation and coverage event is appended to a local JSONL **write-ahead log**
and `fsync`'d *before* anything else; the in-memory annotation list and the
neuroglancer annotation layers are **derived views**, and recovery after any crash
is simply replaying the log. Deletes are tombstone events (never in-place mutation),
and each annotation carries a `uuid` for idempotent replay/sync.

## Considered alternatives

- **In-memory only** — a kernel/browser crash or stray exception loses the whole
  session. This is the exact failure mode the WAL exists to prevent.
- **CAVE-table-first (POST every keypress)** — maximally durable/shared, but
  network- and write-permission-dependent and slow on the hot path; a blip stalls
  annotating. Kept as a *checkpoint* sync (off the hot path), not the primary store.

## Consequences

The only failure the WAL can't survive is disk loss, which the periodic CAVE sync
(deferred past v1) covers.
