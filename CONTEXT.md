# EM Proofreading Workflow

The shared language for the skeleton-driven proofreading workflow: a human flies
through an EM cell along its L2 skeleton (driven from the neuroglancer python
API) and lays down typed annotations marking proofreading errors, which a human
later turns into chunkedgraph edits.

## Language

### Identity & segmentation

**Root id**:
The chunkedgraph id of a whole agglomerated cell *at a point in time*. Volatile — changes on every edit. Used as the per-session entry handle only.
_Avoid_: seg id, segment id, segment.

**Supervoxel id**:
The atomic level-0 segmentation unit beneath a point; stable across all edits. The anchor for **annotations**.
_Avoid_: base segment, sv (in prose).

**L2 id**:
A level-2 chunkedgraph node. There are *more* L2 ids than skeleton vertices; each maps to a vertex via the skeleton's `mesh_to_skel_map` (many L2 ids per vertex). Stable except in chunks an **edit** touches. The anchor for **coverage**.

**Seed supervoxel**:
The supervoxel captured at first load (on the soma / first node) that is the cell's *durable* identity across edits. The current root is always `get_roots([seed], timestamp=now)`.

### Work products

**Annotation**:
A single tagged 3D point a proofreader drops during traversal. Records a click **`xyz`** (the source of truth); its **supervoxel id** anchor is *derived* from `xyz` via CloudVolume. Carries a **tag**, the capture-time root id, the materialization version, a timestamp, and a uuid.

**Tag**:
The type of an annotation: `merge error | split error | extend | question`.

**Edit**:
A manual chunkedgraph split or merge the proofreader performs **outside this tool**, which changes the segmentation and therefore the **root id**.

### Structure & progress

**Branch path**:
One unbranched stretch of skeleton between branch points / tips. The unit of navigation *and* coverage; the set of branch paths partitions the cell.
_Avoid_: segment, branch (alone).

**Coverage**:
The set of visited **L2 ids**, which determines each branch path's state: `to-review | covered | omitted`. Survives edits because it is keyed on L2 ids, not branch paths.

**Omitted subtree**:
The distal subtree pruned from review when a `merge error` is dropped — branch paths belonging to the wrongly-attached object, marked `omitted` rather than `to-review`.

**WAL** (write-ahead log):
The local append-only JSONL file that is the session source of truth. The in-memory annotation list and the neuroglancer layers are *views* of it; recovery = replay the log. Deletes are tombstone events.

## Relationships

- A **cell**'s durable identity is a **seed supervoxel**; its **root id** is a transient per-session handle.
- A **root id** resolves (at a timestamp) to sets of **supervoxel ids** and **L2 ids**.
- An **annotation** is anchored to exactly one **supervoxel id** and carries one **tag**.
- A **branch path** is a sequence of L2-skeleton vertices, each mapping to an **L2 id**; it is `covered` once its L2 ids are visited.
- A **merge error** annotation terminates its branch-path fly-through early and prunes the distal **omitted subtree** from coverage.
- An **edit** changes the **root id** and the **L2 ids** of touched chunks; **supervoxel ids** never change.

## Example dialogue

> **Dev:** "After the proofreader edits the cell, the root id changes — don't all the annotations break?"
> **Domain expert:** "No. Annotations are anchored to **supervoxel ids**, which never change. On re-entry we re-resolve them against the new **root id**."
> **Dev:** "And how do we know which **branch paths** still need review?"
> **Domain expert:** "**Coverage** is a set of visited **L2 ids**. The chunks an **edit** touches get *new* L2 ids, so those branch paths automatically fall back to `to-review`."
> **Dev:** "What about the stuff past a **merge error**?"
> **Domain expert:** "That's the **omitted subtree** — it belongs to the wrongly-attached object, so we prune it and stop the fly-through there."

## Flagged ambiguities

- **"seg id" / "segment"** — used for both the chunkedgraph **root id** (the input) and a linear skeleton stretch. Resolved: the input is a **root id**; the linear stretch is a **branch path**; the bare word **"segment" is banned**.
- **"version"** — used for both the CAVEclient **materialization version** and the skeleton-service `skeleton_version`. Resolved: the **"client version"** input is the **materialization version** passed to `CAVEclient(datastack, version=...)`.
- **"distal"** — relies on the skeleton being **rooted at the soma**. Resolved (v1): assume the soma side is the true cell, so the subtree distal to a merge error is the omitted one. Revisit for soma-less fragments / merges near the soma.
