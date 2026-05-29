# EM proofreading workflow — design & v1 plan

_Last updated: 2026-05-28. Vocabulary in [../CONTEXT.md](../CONTEXT.md); decision
records in [adr/](adr/). Builds on the [`proofreading/`](../proofreading/) package
(`FlyThrough`, `FlyThroughControls`)._

## Goal

Drive a skeleton fly-through of an EM cell (MICrONS minnie65) from the
**neuroglancer python API** so a human can review the whole cell and lay down
**typed annotations** marking proofreading errors. The annotations ultimately
feed chunkedgraph **edits** — but in v1 the editing is a separate, manual step.

## Inputs

- **root id** — chunkedgraph id of the cell (the "seg id").
- **client version** — the CAVEclient **materialization version**:
  `CAVEclient(datastack, version=client_version)`.
- **datastack** — `minnie65_phase3_v1` (live, proofreadable; production target).
  `minnie65_public` is the **read-only dev sandbox** (no edits, no root changes).

## The two-phase workflow (v1)

> **Annotate first, edit second.** The tool is read-only (reads segmentation,
> writes annotations); it never edits the chunkedgraph. See [ADR 0003](adr/0003-read-only-two-phase.md).

**Phase A — Annotate (read-only).**
1. Load by root id + materialization version; capture a **seed supervoxel** as the
   cell's durable identity.
2. Fetch the **L2 skeleton** (`client.skeleton.get_skeleton`) → resample to ~uniform
   arc length → build a **rotation-minimizing frame** → enumerate **branch paths**
   → build the coverage **checklist**.
3. Fly each branch path with `FlyThrough`; the proofreader drops **single tagged
   points** (one keypress per tag: `merge error / split error / extend / question`).
   Each point captures `uuid, tag, xyz, supervoxel_id, root_id,
   materialization_version, timestamp`.
4. **Merge-error special case:** terminate the branch-path fly-through early at the
   annotated vertex `v` and **prune the strictly-distal subtree** — omit `subtree(v)`
   *excluding* `v` (the trunk stays reviewed up to and including the merge point). Mark
   those branch paths / L2 ids `omitted`, keyed to the annotation uuid; reversible via
   tombstone. See the validated rule below.
5. **Coverage** = set of visited **L2 ids**; mark each branch path
   `covered` / `omitted`. Everything appended to the local **WAL**.

**Phase B — Edit (manual, outside the tool).**
The proofreader performs the actual splits/merges by hand in the neuroglancer
proofreading UI, guided by the annotations. This changes the **root id**.

**Phase C — Reconcile (read-only).**
Paste the **updated root id** → re-fetch the skeleton → recompute branch paths →
classify each by **L2-id coverage** (`covered` / `new` / `changed`) → emit the
**remaining paths to review**. Loop back to Phase A on the new root for what's left.

## Path building (skeleton → camera)

1. **Fetch** the L2 skeleton for the root id via the skeleton service
   (`get_skeleton(..., output_format='dict')` → see the verified dict shape below;
   needs `cloudvolume`).
2. **Resample** each branch path to ~uniform arc length (L2 vertices are jagged;
   this stabilizes tangents and evens out fly-through speed).
3. **Tangents = edge-based** (parent→child along the *ordered* branch path). Because
   the path is ordered, the direction is sign-consistent — no `flip_*` hacks.
4. **Rotation-minimizing frame** (T, N, B) propagated along the path — twist-free,
   never degenerates on straight runs (unlike `cross(tangent_i, tangent_{i+1})`).
5. **Orientation:** build `crossSectionOrientation` from `[N, B, T]` so that
   **panel 1 = the cross-section** (⊥ neurite, viewing direction = T). Neuroglancer's
   other two cross-section panels then show the two longitudinal planes for free.

## Viewer layers

- **EM imagery** — always on.
- **Graphene segmentation** — target cell highlighted; **other segments off by
  default, toggled on by a key** (revealed only when inspecting a merge).
- **Skeleton overlay** — current branch path + a position marker; full-cell skeleton faint.
- **Annotation layers** — per-tag, colored, always visible.
- **3D panel** — skeleton + position + annotations (skeleton-only; no mesh).

## Data model & persistence

- **Annotation** records a click **`xyz`**; its **supervoxel id** (the stable anchor)
  is derived from `xyz` via CloudVolume. **Coverage** is keyed on **L2 ids**, derived
  from visited skeleton vertices via `mesh_to_skel_map`. Both anchors are stable across
  edits; the skeleton / branch paths / checklist are transient and regenerated each
  session. See [ADR 0001](adr/0001-stable-id-anchoring.md).
- **WAL:** every annotation and coverage event is appended + `fsync`'d to a local
  JSONL file *before* anything else; the in-memory list and neuroglancer layers are
  derived views; recovery = replay the log; deletes are tombstones; `uuid` gives
  idempotency. See [ADR 0002](adr/0002-wal-first-durability.md).
- **CAVE annotation table sync** is **deferred** — local WAL only for v1.

## Interaction (v1)

- **Key per tag**, single point, committed on one keypress. We record only the click
  **`xyz`** (from neuroglancer); the **supervoxel** (the stable anchor) is *derived*
  later via CloudVolume `scattered_points` — so `xyz` is the source of truth and the
  supervoxel is a re-derivable cache (supervoxels never change).
- `FlyThrough` controls drive within-path motion (play / pause / step / reverse / speed).
- Keys also: toggle neighbor segments, mark a branch path done, (un)mark merge error.
- Free-navigation override is always available.

## Deferred (post-v1)

- Multi-seed `merge error` sets and two-anchor `split error` / `extend` (needed for
  *automated* edits and verifiable resolution).
- Resolution verification and the per-annotation status state machine (the two-phase
  split makes them unnecessary for v1).
- **Automated edits:** a `split error` / `extend` fix is a 2-point **merge** (auto-able);
  a `merge error` fix is a min-cut **split** needing multiple side-labeled seeds
  (likely human-driven even long-term).
- CAVE annotation-table sync (bound spatial points → materialization-resolved roots).

## Verified against the live API (2026-05-28, `minnie65_public` @ mat v1718)

Probed with the installed stack (caveclient 8.1.0 + cloud-volume) against root id
`864691135572530981`. Both former gating risks are retired:

**Skeleton service** — `client.skeleton.get_skeleton(root_id, skeleton_version=4,
output_format='dict')` returns a dict with:
- `vertices` (N, 3) float64 in **nm**; `edges` (N-1, 2) → a **tree**; rooted at the
  vertex index in `root` (int).
- `compartment` (N,) **per-vertex**: 1 = soma, 2 = axon, 3 = dendrite; `radius` (N,) per-vertex.
- `lvl2_ids` (M,) + `mesh_to_skel_map` (M,) with **M > N** (here 14801 vs 7445) —
  **L2 ids are per-L2-node, NOT per skeleton vertex.** The L2 ids for vertex `v` are
  `lvl2_ids[mesh_to_skel_map == v]`; coverage for a branch path (vertex set `V`) is
  `lvl2_ids[isin(mesh_to_skel_map, V)]`.
- `meta` carries `root_id` and the **soma point** (`soma_pt_x/y/z`) — handy for the seed.
- `skeleton_version` is **separate** from the materialization version (0 = NG-compat,
  -1 = latest, default 4). **`get_skeleton` requires `cloudvolume`.**

**Point → supervoxel → root** (this replaces reading the supervoxel under the cursor):
- `seg = client.info.segmentation_cloudvolume(agglomerate=False)`; mip0 resolution
  `[8, 8, 40]` nm/voxel.
- `voxel = xyz_nm / [8, 8, 40]`; `seg.scattered_points(voxels, coord_resolution=[8,8,40])`
  → `{voxel: supervoxel}` (batch — the proofreader's recorded `xyz` is all we need).
- `client.chunkedgraph.get_root_id(sv)` → current root. **Round-trip confirmed**
  (soma supervoxel resolved back to the cell root id).

**Branch-path tree + merge-error pruning** — prototyped on the same skeleton:
- The rooted tree from `edges` + `root` is a connected tree (7445 verts / 7444 edges);
  the `root` vertex equals `meta.soma_pt` — the soma-rooted assumption holds for this cell.
- 189 branch paths, 92 branch points, 98 tips — enumerated by splitting at the root,
  branch points (≥2 children), and tips.
- A `merge error` at vertex `v` omits the **strict** distal subtree (`subtree(v)` minus
  `v`; the trunk path *ending* at `v` stays reviewed). Two independent methods —
  geometric (vertex subtree → L2 ids via `mesh_to_skel_map`) and topological (descendant
  branch paths in the branch-path tree) — **agree exactly**: a merge just above the soma
  prunes 185/189 paths (98% of L2 ids); a mid-branch merge truncates the one containing
  path and prunes 0 subbranches.
- Implementation note: use the **geometric vertex-subtree (excluding `v`) as authoritative**;
  the branch-path tree is for display/checklist. Mind the shared branch-node vertex at
  path boundaries (it's where the naive `subtree(v)`-inclusive version over-counts by one).

## Open implementation questions (verify at build time)

- `new` vs `changed` L2 disambiguation in Phase C — spatial proximity vs the
  chunkedgraph lineage APIs.
- CAVE **write permissions** on `minnie65_phase3_v1` (for the later table sync).
- `client.chunkedgraph` split/merge signatures (for the future auto-edit step).

## Relationship to existing code

Reuses `proofreading.FlyThrough` / `FlyThroughControls`. New modules (planned):
a CAVE/EM client wrapper, a skeleton→path builder (resample + edge tangents + RMF),
an annotator (keypress capture + WAL), a coverage tracker, and EM viewer setup
(graphene segmentation + EM imagery layers).

EM deps are the `em` extra (`uv sync --extra em` → `caveclient` + `cloud-volume`).
A CAVE token at `~/.cloudvolume/secrets/cave-secret.json` is required for live calls.
