# Anchor durable state to stable ids, not root ids

Every chunkedgraph **edit** changes a cell's **root id**, so keying stored work on
root ids would orphan all of it after the first edit. We instead anchor
**annotations** to **supervoxel ids** and **coverage** to **L2 ids** (both stable
except in chunks an edit touches), and treat a cell's durable identity as a **seed
supervoxel** — the current root is always re-resolved via
`get_roots([seed], timestamp=now)`. The root id is only a transient per-session
handle, and the skeleton / branch paths / checklist are transient structures
regenerated each session.

## Why two *different* anchors

Annotations and coverage deliberately use different ids, for opposite stability reasons:

- **Annotations → supervoxel id** (never changes). A flag must *never vanish*; it
  re-resolves to whatever root contains it after any edit.
- **Coverage → L2 id** (changes only for chunks an edit touches). Reviewed-status must
  *expire exactly where an edit happened*. Keying coverage on supervoxels would be
  **wrong**: supervoxels never change, so an edited-in-place region would stay marked
  "reviewed" and the proofreader would skip re-checking the edit. The L2 id's volatility
  is the signal that resurfaces changed regions — it is the feature.

## Consequences

- After edits, annotations re-attach by supervoxel and coverage re-attaches by L2 id;
  chunks the edit touched get new L2 ids, so those branch paths automatically return
  to `to-review` (the mechanism behind Phase C reconciliation).
- Slightly over-conservative: a tiny edit to one chunk re-flags the whole branch path
  through it. Chosen deliberately (re-reviewing extra beats missing an edit).
- The data model is more indirect than "store everything under the root id," which is
  the obvious-but-wrong approach a future reader might otherwise expect.
