"""
Timelapse capture script para sa continuous image data collection.

Usage:
    python timelapse_capture.py --meat-type Chicken --interval 5 --hours 12

Ini-save niya yung images sa isang folder na may timestamp,
tapos may log file din para ma-track yung session.

HINDI automatic mag-start — kailangan i-press yung physical button
(GPIO 22) or Enter sa keyboard bago mag-start ng capture.
I-on muna yung ilaw, lagay yung meat, tapos saka lang i-press.

Pag nag-Ctrl+C ka, safe lang — mag-sstop siya nang maayos.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Reuse the existing camera service from the raspi app ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from camera_capture import CameraCaptureService

try:
    from gpiozero import Button as GpioButton
except Exception:
    GpioButton = None


STOP_FLAG = False
START_FLAG = threading.Event()


def _handle_signal(_sig, _frame):
    global STOP_FLAG
    STOP_FLAG = True
    START_FLAG.set()  # unblock waiting din pag Ctrl+C


def _keyboard_listener():
    """Fallback: wait for Enter key (runs sa background thread)."""
    try:
        input()
        START_FLAG.set()
    except EOFError:
        pass


def wait_for_start_signal() -> None:
    """Block hanggang ma-press yung physical button o Enter key."""
    gpio_button = None

    if GpioButton is not None:
        try:
            gpio_button = GpioButton(
                config.RESERVED_BUTTON_GPIO_PIN,
                pull_up=getattr(config, "BUTTON_PULL_UP", True),
                bounce_time=getattr(config, "BUTTON_BOUNCE_SECONDS", 0.15),
            )
            gpio_button.when_pressed = lambda: START_FLAG.set()
            print(f"  GPIO button ready sa pin {config.RESERVED_BUTTON_GPIO_PIN}.")
        except Exception as exc:
            print(f"  [WARNING] Hindi ma-init yung GPIO button: {exc}")
            gpio_button = None

    # Keyboard fallback — laging available
    kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    kb_thread.start()

    if gpio_button is not None:
        print("  >> I-PRESS yung BUTTON o ENTER para mag-start. <<")
    else:
        print("  >> I-PRESS ang ENTER para mag-start. <<")

    START_FLAG.wait()

    if gpio_button is not None:
        try:
            gpio_button.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timelapse capture: kukunan ng image every N minutes."
    )
    parser.add_argument(
        "--meat-type",
        type=str,
        required=True,
        help="Ano yung meat? e.g. Chicken, Pork, Beef",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Ilang minutes between captures (default: 5)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=12,
        help="Ilang oras itatakbo yung session (default: 12)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Saan i-save yung images (default: captures/timelapse_<meat>_<date>)",
    )
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = parse_args()
    meat_type = args.meat_type.strip().title()
    interval_seconds = args.interval * 60
    max_captures = int((args.hours * 3600) / interval_seconds) + 1

    # Setup output folder
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.CAPTURE_DIR / f"timelapse_{meat_type}_{session_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "capture_log.csv"

    print("=" * 60)
    print(f"  Meat type:       {meat_type}")
    print(f"  Interval:        every {args.interval} minutes")
    print(f"  Total duration:  {args.hours} hours ({max_captures} captures)")
    print(f"  Output folder:   {output_dir}")
    print(f"  Log file:        {log_path}")
    print("=" * 60)
    print()
    print("  Setup checklist:")
    print("    1. I-ON yung ilaw (switch)")
    print("    2. Ilagay yung meat sa chamber")
    print("    3. I-close yung chamber")
    print("    4. Saka lang i-press yung button / Enter")
    print()

    # ── Wait for button press bago mag-start ──
    wait_for_start_signal()

    if STOP_FLAG:
        print("\nCancelled bago mag-start.")
        return

    print()
    print("Starting camera...")

    camera = CameraCaptureService(output_dir=output_dir)

    # Setup CSV log
    log_existed = log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_existed:
        writer.writerow([
            "capture_number",
            "timestamp",
            "elapsed_minutes",
            "meat_type",
            "filename",
        ])
        log_file.flush()

    start_time = time.monotonic()
    capture_count = 0

    print(f"Capturing... (Ctrl+C para i-stop)")
    print()

    try:
        while capture_count < max_captures and not STOP_FLAG:
            elapsed_minutes = (time.monotonic() - start_time) / 60.0
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")

            prefix = f"{meat_type}_{capture_count:04d}"
            try:
                image_path = camera.capture_image(prefix=prefix)
                filename = Path(image_path).name
            except Exception as exc:
                print(f"  [ERROR] Capture failed: {exc}")
                filename = "FAILED"

            capture_count += 1
            writer.writerow([
                capture_count,
                timestamp_str,
                f"{elapsed_minutes:.1f}",
                meat_type,
                filename,
            ])
            log_file.flush()

            print(
                f"  [{capture_count}/{max_captures}] "
                f"{timestamp_str} | {elapsed_minutes:.1f} min | {filename}"
            )

            # Wait for next interval (check stop flag every second para responsive sa Ctrl+C)
            if capture_count < max_captures:
                wait_until = time.monotonic() + interval_seconds
                while time.monotonic() < wait_until and not STOP_FLAG:
                    time.sleep(1)

    finally:
        log_file.close()
        camera.close()
        print()
        print("=" * 60)
        print(f"  Done! {capture_count} images captured.")
        print(f"  Saved sa: {output_dir}")
        print(f"  Log file: {log_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
