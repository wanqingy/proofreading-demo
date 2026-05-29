"""EM proofreading workflow (Phase A: annotate).

A skeleton-driven fly-through over MICrONS minnie65 that lets a human review a
whole cell and drop typed annotations marking proofreading errors. See
``docs/proofreading-workflow.md`` for the design and ``CONTEXT.md`` for vocabulary.

Phase A engine (this package):

- :mod:`~proofreading.em.client`        -- CAVE / CloudVolume wrapper.
- :mod:`~proofreading.em.skeleton_tree` -- rooted tree, branch paths, merge-error pruning.
- :mod:`~proofreading.em.path`          -- resample + rotation-minimizing camera frame.
- :mod:`~proofreading.em.wal`           -- append-only write-ahead log.
- :mod:`~proofreading.em.coverage`      -- visited / omitted L2 ids + the branch-path checklist.

Requires the ``em`` extra (``uv sync --extra em`` -> caveclient + cloud-volume) and a
CAVE token at ``~/.cloudvolume/secrets/cave-secret.json``.
"""

from .skeleton_tree import SkeletonTree, BranchPath
from .path import resample_path, tangents, rotation_minimizing_frame, frame_to_quaternion
from .wal import WAL, Annotation, TAGS
from .coverage import Coverage, PathState

try:  # EMClient needs the `em` extra (caveclient + cloud-volume)
    from .client import EMClient
except ImportError:  # pragma: no cover
    EMClient = None

try:  # the interactive session needs neuroglancer (core dep)
    from .annotator import ProofreadSession
except ImportError:  # pragma: no cover
    ProofreadSession = None

__all__ = [
    "SkeletonTree",
    "BranchPath",
    "resample_path",
    "tangents",
    "rotation_minimizing_frame",
    "frame_to_quaternion",
    "WAL",
    "Annotation",
    "TAGS",
    "Coverage",
    "PathState",
    "EMClient",
    "ProofreadSession",
]
