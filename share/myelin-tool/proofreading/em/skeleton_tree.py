"""Rooted L2-skeleton tree: branch paths, subtree queries, merge-error pruning.

Built from a skeleton-service dict (``client.skeleton.get_skeleton(root_id,
output_format='dict')``). The dict carries ``vertices`` (N, 3 nm), ``edges``
(N-1, 2), the soma-rooted ``root`` vertex index, plus ``lvl2_ids`` / ``mesh_to_skel_map``
(per L2 node, with M > N) and per-vertex ``compartment`` / ``radius``.

The algorithm here was validated on a real cell (root 864691135572530981): the
tree is connected, the ``root`` vertex equals ``meta.soma_pt``, and a merge-error
prune of the *strict* distal subtree agrees between the geometric (vertex-subtree)
and topological (branch-path-tree) methods. See ``docs/proofreading-workflow.md``.

Nothing here is persisted -- it is regenerated from the skeleton each session.
Durable state lives in the WAL keyed on stable supervoxel / L2 ids.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BranchPath:
    """One unbranched stretch of skeleton between branch nodes / tips.

    ``vertices`` is ordered from the proximal (soma-ward) branch node to the distal
    branch node or tip. ``parent`` / ``children`` index into the owning tree's
    ``branch_paths`` list and form the branch-path tree.
    """

    id: int
    vertices: np.ndarray  # ordered vertex indices, proximal -> distal
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)

    @property
    def start(self) -> int:
        """Proximal (soma-ward) branch node."""
        return int(self.vertices[0])

    @property
    def end(self) -> int:
        """Distal branch node or tip."""
        return int(self.vertices[-1])


class SkeletonTree:
    """A soma-rooted skeleton with branch-path and subtree structure."""

    def __init__(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        root: int,
        lvl2_ids: Optional[np.ndarray] = None,
        mesh_to_skel_map: Optional[np.ndarray] = None,
        compartment: Optional[np.ndarray] = None,
        radius: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
    ):
        self.vertices = np.asarray(vertices, dtype=float)
        self.edges = np.asarray(edges, dtype=np.int64)
        self.root = int(root)
        self.n = len(self.vertices)
        self.lvl2_ids = None if lvl2_ids is None else np.asarray(lvl2_ids)
        self.mesh_to_skel_map = (
            None if mesh_to_skel_map is None else np.asarray(mesh_to_skel_map)
        )
        self.compartment = None if compartment is None else np.asarray(compartment)
        self.radius = None if radius is None else np.asarray(radius)
        self.meta = meta or {}

        self._build_tree()
        self._build_euler_intervals()
        self._build_branch_paths()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_skeleton_dict(cls, sk: dict) -> "SkeletonTree":
        """Build from the dict returned by ``get_skeleton(output_format='dict')``."""
        return cls(
            vertices=sk["vertices"],
            edges=sk["edges"],
            root=int(sk["root"]),
            lvl2_ids=sk.get("lvl2_ids"),
            mesh_to_skel_map=sk.get("mesh_to_skel_map"),
            compartment=sk.get("compartment"),
            radius=sk.get("radius"),
            meta=sk.get("meta"),
        )

    def _build_tree(self) -> None:
        """BFS from the root to assign parent / children (distal = away from root)."""
        adj: List[List[int]] = [[] for _ in range(self.n)]
        for a, b in self.edges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))

        self.parent = np.full(self.n, -1, dtype=np.int64)
        self.children: List[List[int]] = [[] for _ in range(self.n)]
        seen = np.zeros(self.n, dtype=bool)
        dq = deque([self.root])
        seen[self.root] = True
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    self.parent[w] = u
                    self.children[u].append(w)
                    dq.append(w)
        self.is_tree = bool(seen.all()) and len(self.edges) == self.n - 1
        self.child_count = np.array([len(c) for c in self.children])

    def _build_euler_intervals(self) -> None:
        """Iterative DFS in/out times for O(1) ancestor/descendant tests."""
        tin = np.zeros(self.n, dtype=np.int64)
        tout = np.zeros(self.n, dtype=np.int64)
        t = 0
        stack = [(self.root, False)]
        while stack:
            u, processed = stack.pop()
            if processed:
                tout[u] = t
                t += 1
            else:
                tin[u] = t
                t += 1
                stack.append((u, True))
                for w in self.children[u]:
                    stack.append((w, False))
        self._tin, self._tout = tin, tout

    def _build_branch_paths(self) -> None:
        """Split the tree at the root, branch points (>=2 children) and tips."""
        branch_nodes = {self.root} | set(np.where(self.child_count != 1)[0].tolist())
        paths: List[BranchPath] = []
        end_to_path: Dict[int, int] = {}
        for b in branch_nodes:
            for c in self.children[b]:
                vids = [b, c]
                u = c
                while len(self.children[u]) == 1:
                    u = self.children[u][0]
                    vids.append(u)
                bp = BranchPath(id=len(paths), vertices=np.array(vids, dtype=np.int64))
                end_to_path[bp.end] = bp.id
                paths.append(bp)

        # link parent/child: a path's parent is the path ending at its start node
        for bp in paths:
            parent_id = end_to_path.get(bp.start)
            if parent_id is not None and parent_id != bp.id:
                bp.parent = parent_id
                paths[parent_id].children.append(bp.id)
        self.branch_paths = paths

    # ------------------------------------------------------------------ #
    # subtree queries
    # ------------------------------------------------------------------ #
    def is_descendant(self, u: int, v: int) -> bool:
        """True if ``u`` lies in the subtree rooted at ``v`` (inclusive)."""
        return bool(self._tin[v] <= self._tin[u] and self._tout[u] <= self._tout[v])

    def subtree_mask(self, v: int, include_root: bool = True) -> np.ndarray:
        """Boolean mask of the subtree rooted at vertex ``v``.

        ``include_root=False`` gives the *strictly distal* subtree (excludes ``v``) --
        the set pruned by a merge error (the trunk stays reviewed up to ``v``).
        """
        mask = (self._tin >= self._tin[v]) & (self._tout <= self._tout[v])
        if not include_root:
            mask[v] = False
        return mask

    def path_between(self, u: int, v: int) -> np.ndarray:
        """Ordered vertex indices from ``u`` to ``v`` via their lowest common ancestor.

        Used to join two arbitrary points along the tree into a polyline (e.g. a myelin
        interval's start/end vertices, which need not lie on the same branch path).
        """
        u, v = int(u), int(v)
        up = [u]
        a = u
        while not self.is_descendant(v, a):
            a = int(self.parent[a])
            up.append(a)
        lca = a
        down: List[int] = []
        b = v
        while b != lca:
            down.append(b)
            b = int(self.parent[b])
        down.reverse()
        return np.array(up + down, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # vertex <-> L2 id mapping
    # ------------------------------------------------------------------ #
    def l2_ids_for_vertices(self, vertex_ids) -> np.ndarray:
        """Unique L2 ids attached to the given skeleton vertices (via mesh_to_skel_map)."""
        if self.lvl2_ids is None or self.mesh_to_skel_map is None:
            raise ValueError("skeleton has no lvl2_ids / mesh_to_skel_map")
        sel = np.isin(self.mesh_to_skel_map, np.asarray(vertex_ids))
        return np.unique(self.lvl2_ids[sel])

    def vertices_for_l2_ids(self, l2_ids) -> np.ndarray:
        """Skeleton vertices whose L2 node is in ``l2_ids``."""
        sel = np.isin(self.lvl2_ids, np.asarray(l2_ids))
        return np.unique(self.mesh_to_skel_map[sel])

    def nearest_vertex(self, point_nm) -> int:
        """Index of the skeleton vertex nearest a 3D point (nm). Brute force."""
        d = self.vertices - np.asarray(point_nm, dtype=float)
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    # ------------------------------------------------------------------ #
    # merge-error pruning
    # ------------------------------------------------------------------ #
    def prune_merge_error(self, vertex: int) -> "MergePrune":
        """Compute what a merge error at ``vertex`` omits.

        Omits the *strict* distal subtree (excludes ``vertex``; the trunk path ending
        at ``vertex`` stays reviewed). Returns the omitted vertices, their L2 ids, the
        branch paths fully omitted, and the path truncated at the merge point (if any).
        """
        distal = self.subtree_mask(vertex, include_root=False)
        distal_vertices = np.where(distal)[0]
        omitted_l2 = (
            self.l2_ids_for_vertices(distal_vertices)
            if self.lvl2_ids is not None
            else np.array([], dtype=np.int64)
        )

        fully_omitted: List[int] = []
        truncated: Optional[int] = None
        for bp in self.branch_paths:
            body = bp.vertices[1:]  # exclude the shared proximal branch node
            if len(body) and distal[body].all():
                fully_omitted.append(bp.id)
            elif distal[bp.vertices].any():
                # the merge vertex is interior to this path -> it is truncated at v
                truncated = bp.id
        return MergePrune(
            vertex=vertex,
            omitted_vertices=distal_vertices,
            omitted_l2_ids=omitted_l2,
            fully_omitted_paths=fully_omitted,
            truncated_path=truncated,
        )


@dataclass
class MergePrune:
    """Result of :meth:`SkeletonTree.prune_merge_error`."""

    vertex: int
    omitted_vertices: np.ndarray
    omitted_l2_ids: np.ndarray
    fully_omitted_paths: List[int]
    truncated_path: Optional[int]
