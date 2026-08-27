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

### Why the camera sometimes slows down

The HUD shows a **buffer** reading, e.g. `buffer 26% -- slowed to 28%`. The camera advances in
proportion to how much *loaded* EM lies ahead of it, so it can't glide over image data that
hasn't arrived yet — an unloaded stretch of axon would otherwise look exactly like an
unmyelinated one, which is a wrong annotation waiting to happen. At the default speed on a
cached branch it stays at `buffer 100%` and never slows; push the speed slider up and you'll
see it throttle and recover as the loader catches up.

If it slows more than you'd like, run `flyCache()` in the browser devtools console. It prints
resident vs **expired** chunk counts and how full the GPU cache is. Nonzero *expired* means
chunks are being evicted (the cache cap is the problem); zero expired means the loader simply
isn't keeping up, in which case a lower speed or a coarser mask mip is the lever — raising the
cache cap would do nothing.

### If you zoom out, expect black around the edges

Only a narrow strip of image is downloaded around the axon — roughly ±1.2 µm. Zoom out past the
default and the view quickly extends beyond it, and those areas are **black because there is no
image there**, not because it is still loading. Waiting will never fill them.

The HUD says so explicitly when it happens, e.g. `36% of the centre of this view has NO image`. That
warning matters for tagging: black and unmyelinated look identical on screen, so treat any black
region as "no data", never as "unmyelinated".

Measured, so you know what to expect (fraction of the middle of the view with no image):

| zoom | default | 2× out | 4× out | 8× out |
|---|---|---|---|---|
| no image | 0% | 25% | 36% | 73% |

Zooming out also slows loading down, badly, while the camera is flying — at 8× out it took **10 s**
for the view to fill versus **2.5 s** once look-ahead loading was disabled. The tool now turns
look-ahead off automatically past 4× out for exactly this reason, so you should not have to think
about it. If you want the old behaviour for comparison, `?prefetch=0` forces it off at all zooms.

Two things that will *not* help, both measured rather than guessed: raising the number of concurrent
downloads (the browser only opens 6 connections no matter what the setting says), and waiting longer
(the missing tiles do not exist). Zooming back in is the only way to see image everywhere.

### If the page goes blank or freezes

A page that goes fully blank — HUD and all, since the HUD is ordinary HTML — means Chrome killed
the **tab's process**, almost always for using too much memory. That's different from an error
message appearing in the HUD (a JS bug, still recoverable) or a "WebGL context lost" warning
(GPU-side, also still recoverable): a real process kill happens with no warning and nothing left
in the console, so there's normally nothing to go on after it happens.

This tool leaves a breadcrumb for that case: every second it stashes a few numbers (JS heap,
chunk-cache size, GPU/system memory, how many branches you've loaded) in the browser's local
storage, which survives the tab dying because it lives in the browser process, not the tab's.
Reload after a blank-page crash and:

- The HUD shows **"previous session crashed or froze"**, and the full detail — a table of the last
  ~12 samples before it ended — is logged to the browser console (`F12` → Console).
- The same detail is appended to `proofread_sessions/crash_reports/myelin.log` on whichever
  machine is running the backend, so it's readable from a terminal even without opening
  DevTools. If you're asking for help with a crash, that file's last entry is the thing to send.

**Freezing is reported too, and it is a different fault.** A tab can also lock up solid — no
response for many seconds — while its process stays alive. That is not a crash: it recovers, and it
would leave no trace, because the recorder is frozen along with everything else. So the recorder also
measures how late each of its own samples was, and reports `previous session FROZE: main thread
blocked for N s`. Zooming far out while the camera is flying is a known way to trigger this, which is
another reason the tool now limits look-ahead loading when you zoom out.

Reading the report: heap climbing toward its limit points at a JS-side leak; a `webglcontextlost`
note right before the end points at the GPU rather than the tab; a large `stallMs` points at the main
thread being overwhelmed rather than memory.

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

**Running this on a different machine than the one collecting the results?** Then those
`*__myelin.jsonl` files are the only thing worth copying back — everything else in
`proofread_sessions/` regenerates. They're append-only, so copying one mid-session is safe, and
each cell's log is self-contained (keyed by seed supervoxel), so per-cell files move
independently.

Two machines annotating the *same* cell can even be merged by concatenating their logs in any
order: replay collects tombstones and applies them after reading the whole file, so a delete on
one machine still wins over a create on the other, and tag uuids are unique per machine. What
concatenation cannot do is *reconcile* — you get the union of both machines' tags, so if both
reviewed the same branch you'll have two tags per node to sort out. One machine per cell avoids
that entirely.

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
