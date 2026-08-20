# Axon myelination tagging tool

Fly along a neuron's axon in Neuroglancer and tag individual skeleton nodes as myelinated.
This is a trimmed copy of one tool from a larger EM proofreading project — see
[`sync.py`](sync.py) if you're curious what was cut and why.

## Setup

You need your own CAVE token with access to `minnie65_public`, at
`~/.cloudvolume/secrets/cave-secret.json`. If you don't have one, ask whoever shared this with
you, or see the [CAVE docs](https://caveconnectome.github.io/CAVEclient/) for how to generate
your own.

```bash
uv sync
uv run python -m proofreading.em.serve
```

Open **http://localhost:8000** in a browser. That's it — one process, no Node, no notebook.
It opens cell `864691136335553971` by default (57 axon branches); pass a different one with
`?root=<id>`, e.g. `http://localhost:8000/?root=864691135572530981`.

## Using it

The camera flies down one axon branch at a time. At any point:

| key | action |
|---|---|
| `space` | play / pause the camera |
| `t` | tag the nearest skeleton node **myelinated** (default: untagged = unmyelinated) |
| `d` | delete the nearest tag |
| `p` | start/stop **painting** — auto-tags every node the camera passes while flying, so you don't have to hit `t` per node on a long myelinated stretch |
| `x` | mark this branch reviewed and advance to the next |

Green dots are myelinated tags — they're real Neuroglancer annotations, so they're also
visible and deletable in Neuroglancer's own **Annotations** panel, not just through this UI.
Magenta = branch points, white = tips, shown for the whole cell so you can see where the
current branch sits in the larger arbor.

## What to expect

- **The first branch takes a couple of minutes.** Flying a branch means fetching its EM +
  segmentation-mask image chunks from Google Cloud Storage into a local cache before the
  camera can glide over it. Later visits to the same branch are instant.
- **A whole cell is ~30 min and a few GB**, at the default coarse mask resolution
  (`PROOFREAD_TGT_MIP=4` — a deliberate speed/fidelity tradeoff; see `tube.py` if you want the
  sharper, slower alternative).
- **Queue cells ahead of time** instead of waiting branch-by-branch:
  ```bash
  uv run python -m proofreading.em.warm_cells 864691136335553971 864691135572530981
  ```
  Cells build **one at a time**, in the order given, so the first is reviewable long before the
  last starts. `--status` to check on it, `--cancel <root_id>` to stop one. See the module
  docstring for the full option list.
- **Reclaim disk** once you're done with a cell (this is all re-downloadable cache, never your
  tags):
  ```bash
  uv run python -m proofreading.em.clear_cache            # see what's on disk
  uv run python -m proofreading.em.clear_cache --stale-masks -f
  ```
- **Some cells genuinely have no axon** in this dataset (e.g. `864691136420378007`,
  `864691135492640607`, `864691136927797322`) — an empty branch picker for one of those is
  correct behavior, not a bug.

## Where your data goes

Every tag you place or delete is appended to
`proofread_sessions/<datastack>__seed<supervoxel>__myelin.jsonl` — one line per action, never
rewritten in place. It's keyed by the cell's **seed supervoxel**, not its root id, so your
progress survives the root id changing underneath you (segmentation edits elsewhere do this
routinely). Re-running the tool against the same cell resumes exactly where you left off.

`proofread_sessions/tube_cache/` is the *other* thing that accumulates there — the downloaded
EM/mask chunks. That's disposable; `clear_cache.py` above only ever touches that directory
(it hard-refuses to run anywhere a `.jsonl` file is found, so your tags can't be deleted by it).

## Security note

Keep this on localhost. `proofreading.em.serve` binds to `127.0.0.1` only — don't change that
or put the port behind a public tunnel. The backend hands its CAVE token to the browser (via
`/api/cells/{id}/live-sources`) so Neuroglancer can authenticate its own requests; that's by
design for a single-user local tool, but it means anyone who can reach the port can reach your
CAVE token.

## If you want to modify it

The frontend is prebuilt (`web/dist/`) so you don't need Node to just use the tool. To change
it:

```bash
cd web
npm install
npm run build     # rebuilds web/dist/
```

Source: [`web/src/myelin.ts`](web/src/myelin.ts) (the tool itself) and
[`web/src/flykernel.ts`](web/src/flykernel.ts) (the shared Neuroglancer-embedding plumbing).
The backend is [`proofreading/em/`](proofreading/em/), a FastAPI app fronting a headless
review service — `service.py` is the core state machine, `api.py` the HTTP surface.
