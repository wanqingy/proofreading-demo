"""Camera fly-through for Neuroglancer.

Two layers:

1. Stateless helpers (:func:`interpolate_to`, :func:`move_to`, :func:`zoom_to`)
   that tween the viewer state -- the same functions the notebooks defined inline.

2. :class:`FlyThrough`, a controller that runs the animation on a **background
   worker thread** so the Jupyter kernel stays responsive. The old notebooks
   drove the loop with ``time.sleep`` on the main thread, which blocked the
   kernel and made the Pause/Continue buttons unresponsive. Here the worker
   owns all the sleeping and the buttons merely flip thread-safe state.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Callable, List, Optional, Sequence

import neuroglancer
import numpy as np


# --------------------------------------------------------------------------- #
# Stateless tween helpers
# --------------------------------------------------------------------------- #
def interpolate_to(
    viewer: neuroglancer.Viewer,
    final_state: neuroglancer.ViewerState,
    frames_per_second: float = 30,
    seconds: float = 1.0,
    should_continue: Optional[Callable[[], bool]] = None,
) -> None:
    """Smoothly tween from the current viewer state to ``final_state``.

    ``should_continue`` is checked before every frame; returning ``False`` aborts
    the tween early (used by :class:`FlyThrough` to make stop/pause responsive
    mid-transition).
    """
    total_frames = max(1, int(round(seconds * frames_per_second)))
    initial_state = viewer.state
    for frame_i in range(total_frames):
        if should_continue is not None and not should_continue():
            return
        t = frame_i / total_frames
        viewer.set_state(
            neuroglancer.ViewerState.interpolate(initial_state, final_state, t)
        )
        time.sleep(1 / frames_per_second)
    if should_continue is None or should_continue():
        viewer.set_state(final_state)


def build_target_state(
    viewer: neuroglancer.Viewer,
    voxel_coordinates: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    cross_section_scale: Optional[float] = None,
) -> neuroglancer.ViewerState:
    """Copy the current state and apply position / orientation / zoom overrides."""
    state = copy.deepcopy(viewer.state)
    if voxel_coordinates is not None:
        state.voxel_coordinates = list(voxel_coordinates)
    if orientation is not None:
        # camelCase attribute names match what the notebooks used successfully.
        state.crossSectionOrientation = orientation
        state.projectionOrientation = orientation
    if cross_section_scale is not None:
        state.cross_section_scale = cross_section_scale
    return state


def move_to(
    viewer: neuroglancer.Viewer,
    voxel_coordinates: Sequence[float],
    orientation: Sequence[float],
    **kwargs,
) -> None:
    """Animate the camera to a position + orientation."""
    target = build_target_state(viewer, voxel_coordinates, orientation)
    interpolate_to(viewer, target, **kwargs)


def zoom_to(viewer: neuroglancer.Viewer, cross_section_scale: float, **kwargs) -> None:
    """Animate a zoom by changing the cross-section scale."""
    target = build_target_state(viewer, cross_section_scale=cross_section_scale)
    interpolate_to(viewer, target, **kwargs)


# --------------------------------------------------------------------------- #
# Threaded controller
# --------------------------------------------------------------------------- #
class FlyThrough:
    """Drive the camera node-by-node along a precomputed path.

    Parameters
    ----------
    viewer:
        The Neuroglancer viewer to control.
    positions:
        ``(N, 3)`` array of voxel coordinates (``[x, y, z]`` order).
    orientations:
        ``(N, 4)`` array of per-node camera quaternions. May be ``None`` to keep
        the current orientation throughout.
    seconds_per_step / frames_per_second:
        Timing of each node-to-node transition. Both can be changed live via
        :meth:`set_speed`.
    on_index_change:
        Optional callback ``f(index)`` fired (from the worker thread) whenever the
        current node changes -- the UI uses this to keep its progress slider in sync.

    The controller is driven entirely by flipping thread-safe state, so button
    callbacks return instantly and never block the kernel.
    """

    def __init__(
        self,
        viewer: neuroglancer.Viewer,
        positions: Sequence[Sequence[float]],
        orientations: Optional[Sequence[Sequence[float]]] = None,
        seconds_per_step: float = 0.3,
        frames_per_second: float = 30,
        on_index_change: Optional[Callable[[int], None]] = None,
    ):
        self.viewer = viewer
        self.positions = np.asarray(positions, dtype=float)
        self.orientations = (
            None if orientations is None else np.asarray(orientations, dtype=float)
        )
        if self.orientations is not None and len(self.orientations) != len(self.positions):
            raise ValueError("positions and orientations must have the same length")

        self.seconds_per_step = seconds_per_step
        self.frames_per_second = frames_per_second
        self.on_index_change = on_index_change

        self._index = 0
        self._direction = 1  # +1 forward, -1 reverse
        self._lock = threading.Lock()
        self._play = threading.Event()  # set => playing, clear => paused
        self._stop = threading.Event()  # set => worker should exit
        self._thread: Optional[threading.Thread] = None

    # ----- properties ----------------------------------------------------- #
    @property
    def index(self) -> int:
        with self._lock:
            return self._index

    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    @property
    def is_playing(self) -> bool:
        return self._play.is_set() and self._thread is not None and self._thread.is_alive()

    # ----- lifecycle ------------------------------------------------------ #
    def start(self) -> "FlyThrough":
        """Spin up the worker thread (paused) and snap to the current node."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._play.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="flythrough")
        self._thread.start()
        self._goto(self.index, animate=False)
        return self

    def stop(self) -> None:
        """Terminate the worker thread."""
        self._stop.set()
        self._play.set()  # wake the worker so it can see the stop flag
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    # ----- playback controls (called from button callbacks) -------------- #
    def play(self) -> None:
        """Resume autoplay in the current direction."""
        if self._thread is None or not self._thread.is_alive():
            self.start()
        self._play.set()

    def pause(self) -> None:
        """Pause autoplay (worker stops at the end of the current transition)."""
        self._play.clear()

    def toggle(self) -> None:
        self.pause() if self.is_playing else self.play()

    def forward(self) -> None:
        """Set direction forward and play."""
        with self._lock:
            self._direction = 1
        self.play()

    def reverse(self) -> None:
        """Set direction backward and play."""
        with self._lock:
            self._direction = -1
        self.play()

    def step(self, direction: int = 1) -> None:
        """Advance a single node (auto-pauses first). Safe to call repeatedly."""
        self.pause()
        with self._lock:
            target = self._index + (1 if direction >= 0 else -1)
            target = max(0, min(self.n_nodes - 1, target))
        self._goto(target, animate=True)
        self._set_index(target)

    def seek(self, index: int, animate: bool = True) -> None:
        """Jump to an absolute node index (auto-pauses first)."""
        self.pause()
        index = max(0, min(self.n_nodes - 1, int(index)))
        self._goto(index, animate=animate)
        self._set_index(index)

    def set_speed(
        self,
        seconds_per_step: Optional[float] = None,
        frames_per_second: Optional[float] = None,
    ) -> None:
        if seconds_per_step is not None:
            self.seconds_per_step = seconds_per_step
        if frames_per_second is not None:
            self.frames_per_second = frames_per_second

    # ----- internals ------------------------------------------------------ #
    def _set_index(self, index: int) -> None:
        with self._lock:
            self._index = index
        if self.on_index_change is not None:
            try:
                self.on_index_change(index)
            except Exception:  # never let a UI callback kill the worker
                pass

    def _goto(self, index: int, animate: bool) -> None:
        orientation = None if self.orientations is None else self.orientations[index]
        target = build_target_state(self.viewer, self.positions[index], orientation)
        if animate:
            interpolate_to(
                self.viewer,
                target,
                frames_per_second=self.frames_per_second,
                seconds=self.seconds_per_step,
                should_continue=lambda: not self._stop.is_set(),
            )
        else:
            self.viewer.set_state(target)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._play.wait()  # block here while paused (no busy loop)
            if self._stop.is_set():
                break

            with self._lock:
                nxt = self._index + self._direction

            if nxt < 0 or nxt >= self.n_nodes:
                # Hit an end -- pause and wait for the user to change direction.
                self._play.clear()
                continue

            self._goto(nxt, animate=True)
            if self._stop.is_set():
                break
            self._set_index(nxt)
