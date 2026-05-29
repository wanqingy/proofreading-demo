# Anchor durable state to stable ids, not root ids

Every chunkedgraph **edit** changes a cell's **root id**, so keying stored work on
root ids would orphan all of it after the first edit. We instead anchor
**annotations** to **supervoxel ids** and **coverage** to **L2 ids** (both stable
except in chunks an edit touches), and treat a cell's durable identity as a **seed
supervoxel** — the current root is always re-resolved via
`get_roots([seed], timestamp=now)`. The root id is only a transient per-session
handle, and the skeleton / branch paths / checklist are transient structures
regenerated each session.

## Consequences

- After edits, annotations re-attach by supervoxel and coverage re-attaches by L2 id;
  chunks the edit touched get new L2 ids, so those branch paths automatically return
  to `to-review` (this is the mechanism behind Phase C reconciliation).
- The data model is more indirect than "store everything under the root id," which is
  the obvious-but-wrong approach a future reader might otherwise expect.
