"""ipywidgets control panel for a :class:`~proofreading.flythrough.FlyThrough`.

Renders a row of buttons (Reverse / Step back / Play-Pause / Step fwd / Forward),
a speed slider, and a scrubber that stays in sync with the worker thread.

The buttons only flip state on the controller, so they respond instantly even
while the fly-through is animating.
"""

from __future__ import annotations

from ipywidgets import (
    Button,
    FloatSlider,
    HBox,
    IntSlider,
    Label,
    Output,
    VBox,
)
from IPython.display import display

from .flythrough import FlyThrough


class FlyThroughControls:
    """A button panel bound to a ``FlyThrough`` instance.

    Creating the object renders the panel immediately (set ``auto_display=False``
    to suppress and call :meth:`display` yourself).
    """

    def __init__(self, fly: FlyThrough, auto_display: bool = True):
        self.fly = fly
        # Route the controller's index changes back into our scrubber/label.
        fly.on_index_change = self._on_index_change

        self.btn_reverse = Button(description="◀ Reverse", tooltip="Autoplay backward")
        self.btn_step_back = Button(description="Step ◁", tooltip="One node back")
        self.btn_playpause = Button(description="▶ Play", button_style="success")
        self.btn_step_fwd = Button(description="▷ Step", tooltip="One node forward")
        self.btn_forward = Button(description="Forward ▶", tooltip="Autoplay forward")

        self.speed = FloatSlider(
            value=fly.seconds_per_step,
            min=0.05,
            max=2.0,
            step=0.05,
            description="sec/step",
            continuous_update=True,
        )
        self.scrubber = IntSlider(
            value=fly.index,
            min=0,
            max=max(0, fly.n_nodes - 1),
            description="node",
            continuous_update=False,
        )
        self.status = Label(value=self._status_text())
        self.output = Output()

        self._wire_events()

        buttons = HBox(
            [
                self.btn_reverse,
                self.btn_step_back,
                self.btn_playpause,
                self.btn_step_fwd,
                self.btn_forward,
            ]
        )
        self.panel = VBox([buttons, self.speed, self.scrubber, self.status, self.output])

        # Make sure the worker thread is alive before the user touches anything.
        self.fly.start()
        self._refresh_playpause()

        if auto_display:
            self.display()

    # ----- public --------------------------------------------------------- #
    def display(self) -> None:
        display(self.panel)

    def close(self) -> None:
        """Stop the worker thread (call when done proofreading)."""
        self.fly.stop()

    # ----- wiring --------------------------------------------------------- #
    def _wire_events(self) -> None:
        self.btn_reverse.on_click(lambda _b: self._do(self.fly.reverse))
        self.btn_step_back.on_click(lambda _b: self._do(lambda: self.fly.step(-1)))
        self.btn_playpause.on_click(lambda _b: self._do(self._toggle))
        self.btn_step_fwd.on_click(lambda _b: self._do(lambda: self.fly.step(1)))
        self.btn_forward.on_click(lambda _b: self._do(self.fly.forward))

        self.speed.observe(self._on_speed, names="value")
        self.scrubber.observe(self._on_scrub, names="value")

    def _do(self, action) -> None:
        """Run a button action, surfacing errors in the Output instead of swallowing."""
        with self.output:
            try:
                action()
            except Exception as exc:  # pragma: no cover - surfaced to the user
                print(f"error: {exc!r}")
        self._refresh_playpause()

    def _toggle(self) -> None:
        self.fly.toggle()

    # ----- callbacks ------------------------------------------------------ #
    def _on_speed(self, change) -> None:
        self.fly.set_speed(seconds_per_step=change["new"])

    def _on_scrub(self, change) -> None:
        # Only seek when the change came from the user, not our own sync update.
        if change["new"] != self.fly.index:
            self.fly.seek(change["new"])
            self._refresh_playpause()

    def _on_index_change(self, index: int) -> None:
        # Fired from the worker thread; ipywidgets is fine with cross-thread sets.
        if self.scrubber.value != index:
            self.scrubber.value = index
        self.status.value = self._status_text()

    # ----- helpers -------------------------------------------------------- #
    def _refresh_playpause(self) -> None:
        if self.fly.is_playing:
            self.btn_playpause.description = "⏸ Pause"
            self.btn_playpause.button_style = "warning"
        else:
            self.btn_playpause.description = "▶ Play"
            self.btn_playpause.button_style = "success"
        self.status.value = self._status_text()

    def _status_text(self) -> str:
        state = "playing" if self.fly.is_playing else "paused"
        return f"node {self.fly.index} / {self.fly.n_nodes - 1}  ({state})"
