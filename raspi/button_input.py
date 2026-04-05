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


class MeatButtonController:
    def __init__(self, on_meat_selected: Callable[[str], None]) -> None:
        if Button is None:
            raise ButtonInputError(
                "gpiozero is not available. Install gpiozero on the Raspberry Pi to use the physical meat buttons."
            )

        self._buttons: dict[str, Button] = {}
        self._on_meat_selected = on_meat_selected

        try:
            for meat_type, gpio_pin in config.MEAT_BUTTON_GPIO_MAP.items():
                button = Button(gpio_pin, pull_up=True, bounce_time=config.BUTTON_BOUNCE_SECONDS)
                button.when_pressed = lambda label=meat_type: self._handle_press(label)
                self._buttons[meat_type] = button
            LOGGER.info("Physical meat buttons initialized on GPIO pins: %s", config.MEAT_BUTTON_GPIO_MAP)
        except Exception as exc:  # pragma: no cover - hardware specific
            raise ButtonInputError(f"Failed to initialize physical meat buttons: {exc}") from exc

    def _handle_press(self, meat_type: str) -> None:
        LOGGER.info("Physical meat button pressed: %s", meat_type)
        self._on_meat_selected(meat_type)

    def close(self) -> None:
        for button in self._buttons.values():
            try:
                button.close()
            except Exception:
                pass
