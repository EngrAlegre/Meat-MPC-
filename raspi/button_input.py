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
        self._on_scroll_up = on_scroll_up
        self._on_scroll_down = on_scroll_down
        self._on_capture_empty_reference = on_capture_empty_reference
        self._errors: list[str] = []

        self._init_button("scroll_up", config.SCROLL_UP_GPIO_PIN, self._handle_scroll_up)
        self._init_button("scroll_down", config.SCROLL_DOWN_GPIO_PIN, self._handle_scroll_down)
        self._init_button("capture_empty_reference", config.RESERVED_BUTTON_GPIO_PIN, self._handle_capture_empty_reference)

        if not self._buttons:
            details = "; ".join(self._errors) if self._errors else "unknown GPIO initialization error"
            raise ButtonInputError(f"Failed to initialize all physical buttons: {details}")

        LOGGER.info("Buttons initialized: %s", self.status_summary())

    def _init_button(self, name: str, pin: int, callback: Callable[[], None]) -> None:
        try:
            button = Button(
                pin,
                pull_up=True,
                bounce_time=config.BUTTON_BOUNCE_SECONDS,
            )
            button.when_pressed = callback
            self._buttons[name] = button
        except Exception as exc:  # pragma: no cover - hardware specific
            message = f"{name} on GPIO{pin} failed: {exc}"
            self._errors.append(message)
            LOGGER.warning(message)

    def _handle_scroll_up(self) -> None:
        LOGGER.info("Physical scroll-up button pressed.")
        self._on_scroll_up()

    def _handle_scroll_down(self) -> None:
        LOGGER.info("Physical scroll-down button pressed.")
        self._on_scroll_down()

    def _handle_capture_empty_reference(self) -> None:
        LOGGER.info("Physical capture-empty-reference button pressed.")
        self._on_capture_empty_reference()

    def status_summary(self) -> str:
        available = ", ".join(sorted(self._buttons.keys())) or "none"
        if not self._errors:
            return f"available={available}"
        return f"available={available} | errors={' ; '.join(self._errors)}"

    def close(self) -> None:
        for button in self._buttons.values():
            try:
                button.close()
            except Exception:
                pass
