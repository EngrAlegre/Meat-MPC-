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

        try:
            up_button = Button(
                config.SCROLL_UP_GPIO_PIN,
                pull_up=True,
                bounce_time=config.BUTTON_BOUNCE_SECONDS,
            )
            up_button.when_pressed = self._handle_scroll_up
            self._buttons["scroll_up"] = up_button

            down_button = Button(
                config.SCROLL_DOWN_GPIO_PIN,
                pull_up=True,
                bounce_time=config.BUTTON_BOUNCE_SECONDS,
            )
            down_button.when_pressed = self._handle_scroll_down
            self._buttons["scroll_down"] = down_button

            reference_button = Button(
                config.RESERVED_BUTTON_GPIO_PIN,
                pull_up=True,
                bounce_time=config.BUTTON_BOUNCE_SECONDS,
            )
            reference_button.when_pressed = self._handle_capture_empty_reference
            self._buttons["capture_empty_reference"] = reference_button

            LOGGER.info(
                "Buttons initialized | up=GPIO%d | down=GPIO%d | capture_empty_reference=GPIO%d",
                config.SCROLL_UP_GPIO_PIN,
                config.SCROLL_DOWN_GPIO_PIN,
                config.RESERVED_BUTTON_GPIO_PIN,
            )
        except Exception as exc:  # pragma: no cover - hardware specific
            raise ButtonInputError(f"Failed to initialize scroll buttons: {exc}") from exc

    def _handle_scroll_up(self) -> None:
        LOGGER.info("Physical scroll-up button pressed.")
        self._on_scroll_up()

    def _handle_scroll_down(self) -> None:
        LOGGER.info("Physical scroll-down button pressed.")
        self._on_scroll_down()

    def _handle_capture_empty_reference(self) -> None:
        LOGGER.info("Physical capture-empty-reference button pressed.")
        self._on_capture_empty_reference()

    def close(self) -> None:
        for button in self._buttons.values():
            try:
                button.close()
            except Exception:
                pass
