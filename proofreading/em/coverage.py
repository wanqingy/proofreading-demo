"""Coverage: which branch paths are reviewed, omitted, or still to-do.

Coverage is keyed on **L2 ids** (stable), not branch paths (transient). A branch
path's state is derived by comparing its L2 ids against the visited / omitted sets:

- ``OMITTED``   -- all its L2 ids are omitted (distal to a merge error).
- ``COVERED``   -- all its L2 ids are reviewed or omitted (incl. a path truncated at a merge).
- ``TO_REVIEW`` -- anything else.

Because the key is L2 ids, this also handles re-entry after edits for free: the
chunks an edit touched get *new* L2 ids that aren't in ``visited``, so their branch
paths fall back to ``TO_REVIEW`` (the Phase C reconciliation, no special code).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .skeleton_tree import BranchPath, SkeletonTree
from .wal import WalState


class PathState:
    TO_REVIEW = "to_review"
    COVERED = "covered"
    OMITTED = "omitted"


@dataclass
class Coverage:
    visited_l2: Set[int] = field(default_factory=set)
    omitted_l2: Set[int] = field(default_factory=set)

    @classmethod
    def from_wal_state(cls, ws: WalState) -> "Coverage":
        return cls(visited_l2=set(ws.visited_l2), omitted_l2=set(ws.omitted_l2))

    # ----- updates (mirror what gets written to the WAL) ----------------- #
    def mark_visited(self, l2_ids) -> None:
        self.visited_l2.update(int(x) for x in l2_ids)

    def mark_omitted(self, l2_ids) -> None:
        self.omitted_l2.update(int(x) for x in l2_ids)

    # ----- classification ------------------------------------------------ #
    def path_state(self, tree: SkeletonTree, bp: BranchPath) -> str:
        # Classify by the path's *body* (exclude the shared proximal branch node,
        # which the parent path owns). This matches SkeletonTree.prune_merge_error,
        # so a merge's child paths -- which start *at* the kept merge vertex -- still
        # classify as OMITTED.
        body = bp.vertices[1:] if len(bp.vertices) > 1 else bp.vertices
        l2 = set(int(x) for x in tree.l2_ids_for_vertices(body))
        if not l2:
            return PathState.COVERED  # nothing to review
        if l2 <= self.omitted_l2:
            return PathState.OMITTED
        if l2 <= (self.visited_l2 | self.omitted_l2):
            return PathState.COVERED  # reviewed, or trunk reviewed + distal omitted
        return PathState.TO_REVIEW

    def checklist(self, tree: SkeletonTree) -> List[Tuple[int, str]]:
        return [(bp.id, self.path_state(tree, bp)) for bp in tree.branch_paths]

    def to_review(self, tree: SkeletonTree) -> List[int]:
        return [bp.id for bp in tree.branch_paths
                if self.path_state(tree, bp) == PathState.TO_REVIEW]

    def summary(self, tree: SkeletonTree) -> Dict[str, int]:
        counts = {PathState.TO_REVIEW: 0, PathState.COVERED: 0, PathState.OMITTED: 0}
        for _, st in self.checklist(tree):
            counts[st] += 1
        return counts
