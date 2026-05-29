# The tool is read-only; edits are a separate manual phase

The annotator only **reads** segmentation and **writes** annotations — it never
performs chunkedgraph edits. The actual splits/merges happen in a distinct
**Phase B**, by a human, in the neuroglancer proofreading UI. The workflow is
therefore two-phase: *annotate the whole cell first, edit second*, then re-enter
with the new root id to reconcile remaining coverage.

## Why

Mixing irreversible chunkedgraph edits into an exploratory fly-through is exactly
where accidents happen; a read-only tool cannot damage the segmentation. The
two-phase boundary also keeps v1 simple — no per-annotation resolution state is
needed, because Phase B consumes the annotations and Phase C just recomputes what's
left to review.

## Consequences & future

- Root ids change between Phase A and Phase C; handled by stable-id anchoring
  ([ADR 0001](0001-stable-id-anchoring.md)).
- Future automation is asymmetric: a `split error` / `extend` fix is a 2-point
  **merge** (automatable); a `merge error` fix is a min-cut **split** needing
  multiple side-labeled seeds (likely human-driven even long-term). Capturing those
  extra operands is deferred until automation is actually built.
