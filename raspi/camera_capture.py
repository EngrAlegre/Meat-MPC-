from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import config
from PIL import Image, ImageOps

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover - hardware-specific import
    Picamera2 = None

try:
    import cv2
except Exception:  # pragma: no cover - environment-specific import
    cv2 = None


LOGGER = logging.getLogger(__name__)


class CameraCaptureError(RuntimeError):
    pass


class CameraCaptureService:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or config.CAMERA_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._picamera2 = None
        self._fallback_capture = None

    def _initialize_picamera2(self) -> bool:
        if Picamera2 is None:
            return False

        if self._picamera2 is not None:
            return True

        try:
            camera = Picamera2()
            still_config = camera.create_still_configuration(
                main={"size": config.CAMERA_STILL_SIZE}
            )
            camera.configure(still_config)
            camera.start()
            time.sleep(config.CAMERA_STARTUP_DELAY_SECONDS)
            self._picamera2 = camera
            LOGGER.info("Pi camera initialized with Picamera2.")
            return True
        except Exception as exc:  # pragma: no cover - hardware-specific
            LOGGER.warning("Picamera2 initialization failed: %s", exc)
            self._picamera2 = None
            return False

    def _initialize_opencv_fallback(self) -> bool:
        if not config.ALLOW_OPENCV_CAMERA_FALLBACK or cv2 is None:
            return False

        if self._fallback_capture is not None:
            return True

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            return False

        self._fallback_capture = capture
        LOGGER.warning("Using OpenCV camera fallback instead of Picamera2.")
        return True

    def capture_image(self, prefix: str | None = None) -> Path:
        prefix = prefix or config.CAMERA_FILENAME_PREFIX
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.output_dir / f"{prefix}_{timestamp}.jpg"

        if self._initialize_picamera2():
            try:
                self._picamera2.capture_file(str(image_path))
                LOGGER.info("Image captured via Picamera2: %s", image_path)
                return image_path
            except Exception as exc:  # pragma: no cover - hardware-specific
                raise CameraCaptureError(f"Failed to capture image with Picamera2: {exc}") from exc

        if self._initialize_opencv_fallback():
            success, frame = self._fallback_capture.read()
            if not success:
                raise CameraCaptureError("OpenCV fallback camera failed to capture a frame.")
            cv2.imwrite(str(image_path), frame)
            LOGGER.info("Image captured via OpenCV fallback: %s", image_path)
            return image_path

        raise CameraCaptureError(
            "No camera is available. Check the Raspberry Pi camera connection and Picamera2 setup."
        )

    def get_preview_image(self, size: tuple[int, int] | None = None) -> Image.Image:
        size = size or config.CAMERA_PREVIEW_SIZE

        if self._initialize_picamera2():
            try:
                frame = self._picamera2.capture_array()
                image = Image.fromarray(frame).convert("RGB")
                return ImageOps.contain(image, size)
            except Exception as exc:  # pragma: no cover - hardware-specific
                LOGGER.warning("Picamera2 preview capture failed: %s", exc)

        if self._initialize_opencv_fallback():
            success, frame = self._fallback_capture.read()
            if not success:
                raise CameraCaptureError("OpenCV fallback camera failed to capture a preview frame.")
            if cv2 is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame).convert("RGB")
            return ImageOps.contain(image, size)

        raise CameraCaptureError(
            "No camera is available. Check the Raspberry Pi camera connection and Picamera2 setup."
        )

    def close(self) -> None:
        if self._picamera2 is not None:
            try:
                self._picamera2.stop()
            except Exception:
                pass
        if self._fallback_capture is not None:
            try:
                self._fallback_capture.release()
            except Exception:
                pass
