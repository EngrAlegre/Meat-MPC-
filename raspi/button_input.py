from __future__ import annotations

import logging
from typing import Callable

import config

try:
    from gpiozero import Button
except Exception:  # pragma: no cover - platform specific import
    Button = None


LOGGER = logging.getLogger(__name__)


class ButtonInputError(RuntimeError):
    pass


class ScrollButtonController:
    """Detects physical button presses via Tk-thread polling (no gpiozero
    background-thread dependency, which can be unreliable on RPi5/lgpio)."""

    def __init__(
        self,
        *,
        on_scroll_up: Callable[[], None],
        on_scroll_down: Callable[[], None],
        on_capture_empty_reference: Callable[[], None],
    ) -> None:
        if Button is None:
            raise ButtonInputError(
                "gpiozero is not available. Install gpiozero on the Raspberry Pi to use the physical buttons."
            )

        self._buttons: dict[str, Button] = {}
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._last_pressed: dict[str, bool] = {}
        self._errors: list[str] = []

        self._add_button("scroll_up", config.SCROLL_UP_GPIO_PIN, on_scroll_up)
        self._add_button("scroll_down", config.SCROLL_DOWN_GPIO_PIN, on_scroll_down)
        self._add_button("capture_empty_reference", config.RESERVED_BUTTON_GPIO_PIN, on_capture_empty_reference)

        if not self._buttons:
            details = "; ".join(self._errors) if self._errors else "unknown GPIO initialization error"
            raise ButtonInputError(f"Failed to initialize all physical buttons: {details}")

        LOGGER.info("Buttons initialized: %s", self.status_summary())

    def _add_button(self, name: str, pin: int, callback: Callable[[], None]) -> None:
        try:
            button = Button(
                pin,
                pull_up=config.BUTTON_PULL_UP,
                bounce_time=config.BUTTON_BOUNCE_SECONDS,
            )
            self._buttons[name] = button
            self._callbacks[name] = callback
            self._last_pressed[name] = bool(button.is_pressed)
        except Exception as exc:  # pragma: no cover - hardware specific
            message = f"{name} on GPIO{pin} failed: {exc}"
            self._errors.append(message)
            LOGGER.warning(message)

    def poll(self) -> None:
        """Call from the Tk main-thread timer to detect rising-edge presses."""
        for name, button in self._buttons.items():
            try:
                pressed_now = bool(button.is_pressed)
            except Exception as exc:  # pragma: no cover - hardware specific
                LOGGER.warning("Failed to read %s: %s", name, exc)
                continue

            was_pressed = self._last_pressed.get(name, False)
            self._last_pressed[name] = pressed_now

            if pressed_now and not was_pressed:
                LOGGER.info("Physical %s button pressed.", name)
                callback = self._callbacks.get(name)
                if callback is not None:
                    callback()

    def status_summary(self) -> str:
        available = ", ".join(sorted(self._buttons.keys())) or "none"
        mode = "pull_up" if config.BUTTON_PULL_UP else "pull_down"
        if not self._errors:
            return f"available={available} | mode={mode}"
        return f"available={available} | mode={mode} | errors={' ; '.join(self._errors)}"

    def close(self) -> None:
        for button in self._buttons.values():
            try:
                button.close()
            except Exception:
                pass
