"""Turn an ordered branch path into a smooth, oriented camera path.

Pipeline (per branch path): resample to ~uniform arc length -> edge-based unit
tangents -> a rotation-minimizing frame (T, N, B) -> a per-node quaternion for
neuroglancer's ``crossSectionOrientation``.

Edge-based tangents are sign-consistent on an *ordered* path (no flip hacks), and
the rotation-minimizing frame (double-reflection method, Wang et al. 2008) is
twist-free and never degenerates on straight runs -- replacing the old
``cross(tangent_i, tangent_{i+1})`` binormal.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def resample_path(points: np.ndarray, step_nm: float = 1000.0) -> np.ndarray:
    """Resample a polyline to ~uniform arc-length spacing ``step_nm``.

    L2 vertices are jagged and irregularly spaced; this stabilizes tangents and
    evens out fly-through speed. Returns at least the original endpoints.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total == 0:
        return pts[:1].copy()
    n = max(2, int(round(total / step_nm)) + 1)
    su = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(su, s, pts[:, k]) for k in range(3)])


def tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents along an ordered path (central differences, sign-consistent)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    t = np.zeros_like(pts)
    if n == 1:
        t[0] = [0.0, 0.0, 1.0]
        return t
    t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return t / norm


def _arbitrary_normal(t: np.ndarray) -> np.ndarray:
    """Some unit vector orthogonal to ``t``."""
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n = a - (a @ t) * t
    return n / np.linalg.norm(n)


def rotation_minimizing_frame(
    points: np.ndarray, T: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Twist-free orthonormal frame (T, N, B) along the path (double reflection)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if T is None:
        T = tangents(pts)
    N = np.zeros_like(pts)
    B = np.zeros_like(pts)
    N[0] = _arbitrary_normal(T[0])
    B[0] = np.cross(T[0], N[0])
    for i in range(n - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = v1 @ v1
        if c1 < 1e-20:
            N[i + 1], B[i + 1] = N[i], B[i]
            continue
        r_l = N[i] - (2.0 / c1) * (v1 @ N[i]) * v1
        t_l = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - t_l
        c2 = v2 @ v2
        n_next = r_l if c2 < 1e-20 else r_l - (2.0 / c2) * (v2 @ r_l) * v2
        # re-orthogonalize against the (more trusted) tangent and normalize
        n_next = n_next - (n_next @ T[i + 1]) * T[i + 1]
        nn = np.linalg.norm(n_next)
        N[i + 1] = n_next / nn if nn > 0 else N[i]
        B[i + 1] = np.cross(T[i + 1], N[i + 1])
    return T, N, B


def _matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion ``[x, y, z, w]`` (Shepperd's method)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def frame_to_quaternion(T: np.ndarray, N: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-node ``crossSectionOrientation`` quaternions from a frame.

    Columns ``[N, B, T]`` so that neuroglancer's first cross-section panel views down
    **T** (the cross-section ⊥ the neurite), with N, B spanning that panel. The exact
    handedness/sign convention is confirmed live in the interactive session.
    """
    T = np.atleast_2d(T)
    N = np.atleast_2d(N)
    B = np.atleast_2d(B)
    out = np.empty((len(T), 4))
    for i in range(len(T)):
        out[i] = _matrix_to_quaternion(np.column_stack([N[i], B[i], T[i]]))
    return out
