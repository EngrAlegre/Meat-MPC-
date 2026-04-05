from __future__ import annotations

import logging
import queue
import tkinter as tk
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
import time
from tkinter import ttk
from typing import Any, Callable

from PIL import Image, ImageOps, ImageTk

from button_input import ButtonInputError, MeatButtonController
import config
from camera_capture import CameraCaptureError, CameraCaptureService
from predict_live import HybridFreshnessPredictor, PredictionLoadError
from sensor_reader import MQSensorReader, SensorInitializationError, SensorReadError


config.ensure_runtime_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(config.APP_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)


class HybridFreshnessGUI:
    BG = "#07131e"
    PANEL = "#102536"
    PANEL_ALT = "#142c40"
    CARD = "#1a3349"
    BORDER = "#264d67"
    TEXT = "#edf4fb"
    MUTED = "#9fb3c7"
    INFO = "#67c4ff"
    SUCCESS = "#43d4a1"
    WARNING = "#f4ba5f"
    DANGER = "#ff7585"
    BUTTON = "#27c288"
    BUTTON_ALT = "#54a7f5"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Hybrid Meat Freshness Scanner")
        self.root.configure(bg=self.BG)
        self.is_fullscreen = True
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.compact_layout = self.screen_width <= 1280 or self.screen_height <= 720
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self._exit_fullscreen)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)

        self.sensor_lock = Lock()
        self.camera_lock = Lock()
        self.worker_queue: queue.Queue[Callable[[], None]] = queue.Queue()

        self.sensor_reader: MQSensorReader | None = None
        self.camera_service: CameraCaptureService | None = None
        self.predictor: HybridFreshnessPredictor | None = None
        self.button_controller: MeatButtonController | None = None

        self.meat_type = tk.StringVar(value="Chicken")
        self.system_state = tk.StringVar(value="Initializing")
        self.message_text = tk.StringVar(value="Starting Raspberry Pi meat freshness scanner...")
        self.warmup_text = tk.StringVar(value="Warm-up status unavailable")
        self.prediction_text = tk.StringVar(value="--")
        self.confidence_text = tk.StringVar(value="Confidence: --")
        self.confidence_note_text = tk.StringVar(value="No prediction yet.")
        self.environment_values: dict[str, tk.StringVar] = {
            "temperature_c": tk.StringVar(value="--"),
            "humidity_percent": tk.StringVar(value="--"),
            "status": tk.StringVar(value="DHT22 not read yet."),
        }

        self.sensor_values: dict[str, tk.StringVar] = {
            "nh3_ratio": tk.StringVar(value="--"),
            "h2s_ratio": tk.StringVar(value="--"),
            "voc_ratio": tk.StringVar(value="--"),
            "nh3_debug": tk.StringVar(value="V: -- | Rs: --"),
            "h2s_debug": tk.StringVar(value="V: -- | Rs: --"),
            "voc_debug": tk.StringVar(value="V: -- | Rs: --"),
        }

        self.last_sensor_snapshot: dict[str, Any] | None = None
        self.latest_image_path: Path | None = None
        self.latest_prediction: dict[str, Any] | None = None
        self.sensor_ready = False
        self.last_photo_image = None
        self.environment_refresh_in_progress = False
        self.last_environment_poll_monotonic = 0.0

        self._configure_styles()
        self._build_layout()
        self._setup_hardware_buttons()
        self._schedule_worker_poll()
        self._schedule_status_refresh()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("Root.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL, relief="flat")
        style.configure("PanelAlt.TFrame", background=self.PANEL_ALT, relief="flat")
        style.configure("Card.TFrame", background=self.CARD, relief="flat")
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 24, "bold"))
        style.configure("SubTitle.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 11))
        style.configure("PanelTitle.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 15, "bold"))
        style.configure("Body.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=self.CARD, foreground=self.MUTED, font=("Segoe UI", 11))
        style.configure("Value.TLabel", background=self.CARD, foreground=self.TEXT, font=("Segoe UI", 26, "bold"))
        style.configure("Debug.TLabel", background=self.CARD, foreground=self.MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=self.PANEL, foreground=self.INFO, font=("Segoe UI", 12, "bold"))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg=self.BG, padx=24, pady=18)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Hybrid Meat Freshness Scanner", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Native Raspberry Pi 5 touchscreen interface using live MQ sensing, Pi camera capture, and the trained hybrid model.",
            style="SubTitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        top_status = tk.Frame(header, bg=self.BG)
        top_status.grid(row=0, column=1, rowspan=2, sticky="e")
        self.state_badge = tk.Label(top_status, textvariable=self.system_state, bg="#16344c", fg=self.INFO, font=("Segoe UI", 12, "bold"), padx=16, pady=10)
        self.state_badge.grid(row=0, column=0, padx=(0, 8))
        self.warmup_badge = tk.Label(top_status, textvariable=self.warmup_text, bg="#12283b", fg=self.MUTED, font=("Segoe UI", 10), padx=16, pady=10)
        self.warmup_badge.grid(row=0, column=1)

        content_shell = tk.Frame(self.root, bg=self.BG)
        content_shell.grid(row=1, column=0, sticky="nsew")
        content_shell.columnconfigure(0, weight=1)
        content_shell.rowconfigure(0, weight=1)

        self.content_canvas = tk.Canvas(
            content_shell,
            bg=self.BG,
            highlightthickness=0,
            bd=0,
        )
        self.content_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(content_shell, orient="vertical", command=self.content_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.content_canvas.configure(yscrollcommand=scrollbar.set)

        self.content = tk.Frame(self.content_canvas, bg=self.BG, padx=24, pady=8)
        self.content_window = self.content_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda _event: self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all")),
        )
        self.content_canvas.bind("<Configure>", self._on_canvas_configure)

        if self.compact_layout:
            self.content.columnconfigure(0, weight=1)
            self.content.rowconfigure(0, weight=1)
            self.content.rowconfigure(1, weight=1)
            self.content.rowconfigure(2, weight=1)
            self.content.rowconfigure(3, weight=1)
        else:
            self.content.columnconfigure(0, weight=2)
            self.content.columnconfigure(1, weight=2)
            self.content.columnconfigure(2, weight=3)
            self.content.rowconfigure(0, weight=1)
            self.content.rowconfigure(1, weight=1)

        self._build_controls_panel(self.content)
        self._build_sensors_panel(self.content)
        self._build_preview_panel(self.content)
        self._build_log_panel(self.content)

    def _build_controls_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        if self.compact_layout:
            panel.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        else:
            panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Controls", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="Choose the meat type, then tap Start Scan. The system will stabilize sensors, capture the image, and predict freshness automatically.", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 16))
        selector_frame = tk.Frame(panel, bg=self.PANEL)
        selector_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(selector_frame, text="Meat Type", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        meat_button_row = tk.Frame(selector_frame, bg=self.PANEL)
        meat_button_row.grid(row=1, column=0, sticky="ew")
        self.meat_buttons: dict[str, tk.Button] = {}
        for idx, meat_type in enumerate(config.MEAT_TYPES):
            button = tk.Button(
                meat_button_row,
                text=meat_type,
                command=lambda value=meat_type: self._set_meat_type(value),
                bg="#183850",
                fg=self.TEXT,
                activebackground=self.BUTTON_ALT,
                activeforeground=self.TEXT,
                bd=0,
                padx=18,
                pady=14,
                font=("Segoe UI", 12, "bold"),
            )
            button.grid(row=0, column=idx, padx=(0 if idx == 0 else 8, 0), sticky="ew")
            meat_button_row.columnconfigure(idx, weight=1)
            self.meat_buttons[meat_type] = button
        self._refresh_meat_buttons()

        actions = [
            ("Start Scan", self.start_scan, self.BUTTON),
            ("Exit App", self.root.destroy, "#8a3243"),
        ]

        for idx, (label, command, color) in enumerate(actions, start=3):
            button = tk.Button(
                panel,
                text=label,
                command=command,
                bg=color,
                fg=self.TEXT,
                activebackground=color,
                activeforeground=self.TEXT,
                bd=0,
                padx=18,
                pady=14,
                font=("Segoe UI", 12, "bold"),
            )
            button.grid(row=idx, column=0, sticky="ew", pady=(0, 10))

        message_panel = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        message_panel.grid(row=10, column=0, sticky="ew", pady=(8, 0))
        self.message_label = tk.Label(
            message_panel,
            textvariable=self.message_text,
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            justify="left",
            wraplength=340,
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.message_label.pack(fill="x")

    def _build_sensors_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        if self.compact_layout:
            panel.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        else:
            panel.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=(0, 12))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Live Sensors", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="The scan uses stabilized averaged ratios. Voltage and Rs stay visible for verification.", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 14))

        cards = tk.Frame(panel, bg=self.PANEL)
        cards.grid(row=2, column=0, sticky="nsew")
        cards.columnconfigure((0, 1, 2), weight=1)

        sensor_layout = [
            ("NH3", "nh3_ratio", "nh3_debug"),
            ("H2S", "h2s_ratio", "h2s_debug"),
            ("VOC", "voc_ratio", "voc_debug"),
        ]

        for idx, (title, ratio_key, debug_key) in enumerate(sensor_layout):
            card = tk.Frame(cards, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 8, 0))
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.sensor_values[ratio_key], style="Value.TLabel").pack(anchor="w", pady=(12, 2))
            ttk.Label(card, text="Rs/Ro", style="Debug.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.sensor_values[debug_key], style="Debug.TLabel").pack(anchor="w", pady=(14, 0))

        notes_panel = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        notes_panel.grid(row=4, column=0, sticky="nsew", pady=(14, 0))
        ttk.Label(notes_panel, text="Stability Notes", style="PanelTitle.TLabel").pack(anchor="w")
        self.stability_text = tk.Text(
            notes_panel,
            height=7,
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            bd=0,
            wrap="word",
            font=("Segoe UI", 10),
            highlightthickness=0,
        )
        self.stability_text.pack(fill="both", expand=True, pady=(8, 0))
        self._set_text_widget(self.stability_text, "No stabilization data yet.")

        env_panel = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        env_panel.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ttk.Label(env_panel, text="Environment Monitor", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            env_panel,
            text="DHT22 temperature and humidity are shown for environmental context only.",
            style="Body.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))
        env_panel.columnconfigure((0, 1), weight=1)

        temp_card = tk.Frame(env_panel, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        temp_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(temp_card, text="Temperature", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(temp_card, textvariable=self.environment_values["temperature_c"], style="Value.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(temp_card, text="Celsius", style="Debug.TLabel").pack(anchor="w")

        humidity_card = tk.Frame(env_panel, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        humidity_card.grid(row=2, column=1, sticky="nsew")
        ttk.Label(humidity_card, text="Humidity", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(humidity_card, textvariable=self.environment_values["humidity_percent"], style="Value.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(humidity_card, text="Relative Humidity (%)", style="Debug.TLabel").pack(anchor="w")

        tk.Label(
            env_panel,
            textvariable=self.environment_values["status"],
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            anchor="w",
            justify="left",
            wraplength=500,
            font=("Segoe UI", 10),
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_preview_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        if self.compact_layout:
            panel.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        else:
            panel.grid(row=0, column=2, rowspan=2, sticky="nsew", pady=(0, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=3)

        ttk.Label(panel, text="Camera Preview and Prediction", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")

        image_frame = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1)
        image_frame.grid(row=1, column=0, sticky="nsew", pady=(14, 12))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = tk.Label(
            image_frame,
            text="No image captured yet.",
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            font=("Segoe UI", 12),
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        prediction_panel = tk.Frame(panel, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        prediction_panel.grid(row=2, column=0, sticky="ew")
        tk.Label(prediction_panel, text="Predicted Freshness", bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 11)).pack(anchor="center")
        tk.Label(prediction_panel, textvariable=self.prediction_text, bg=self.CARD, fg=self.TEXT, font=("Segoe UI", 28, "bold")).pack(anchor="center", pady=(8, 4))
        tk.Label(prediction_panel, textvariable=self.confidence_text, bg=self.CARD, fg=self.SUCCESS, font=("Segoe UI", 12, "bold")).pack(anchor="center")
        tk.Label(
            prediction_panel,
            textvariable=self.confidence_note_text,
            bg=self.CARD,
            fg=self.MUTED,
            wraplength=360,
            justify="center",
            font=("Segoe UI", 10),
        ).pack(anchor="center", pady=(8, 0))

        scores_panel = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        scores_panel.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(scores_panel, text="Class Scores", style="PanelTitle.TLabel").pack(anchor="w")
        self.class_scores_label = tk.Label(
            scores_panel,
            text="No prediction yet.",
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            justify="left",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.class_scores_label.pack(fill="x", pady=(8, 0))

    def _build_log_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        if self.compact_layout:
            panel.grid(row=3, column=0, sticky="nsew")
        else:
            panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0, 12))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        ttk.Label(panel, text="Live Debug Log", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(
            panel,
            height=12,
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            bd=0,
            wrap="word",
            font=("Consolas", 10),
            highlightthickness=0,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self._append_log("GUI initialized. Waiting for hardware actions.")

    def _schedule_worker_poll(self) -> None:
        try:
            while True:
                callback = self.worker_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        self.root.after(120, self._schedule_worker_poll)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.content_canvas.itemconfigure(self.content_window, width=event.width)

    def _schedule_status_refresh(self) -> None:
        self._update_warmup_state()
        self._request_environment_refresh()
        self.root.after(1000, self._schedule_status_refresh)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _set_text_widget(self, widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _set_state(self, label: str, fg: str | None = None, bg: str | None = None) -> None:
        self.system_state.set(label)
        self.state_badge.configure(fg=fg or self.INFO, bg=bg or "#16344c")

    def _set_message(self, message: str, color: str | None = None) -> None:
        self.message_text.set(message)
        self.message_label.configure(fg=color or self.MUTED)
        self._append_log(message)

    def _run_async(self, task_name: str, work: Callable[[], Any], on_success: Callable[[Any], None]) -> None:
        self._set_message(f"{task_name}...", self.INFO)

        def worker() -> None:
            try:
                result = work()
                self.worker_queue.put(lambda: on_success(result))
            except Exception as exc:
                LOGGER.exception("%s failed", task_name)
                self.worker_queue.put(lambda: self._handle_error(task_name, exc))

        Thread(target=worker, daemon=True).start()

    def _handle_error(self, task_name: str, exc: Exception) -> None:
        self._set_state("Error", self.DANGER, "#4a1f28")
        self._set_message(f"{task_name} failed: {exc}", self.DANGER)

    def _refresh_meat_buttons(self) -> None:
        current = self.meat_type.get()
        for meat_type, button in self.meat_buttons.items():
            selected = meat_type == current
            button.configure(bg=self.BUTTON if selected else "#183850")

    def _set_meat_type(self, meat_type: str) -> None:
        self.meat_type.set(meat_type)
        self._refresh_meat_buttons()
        self._set_message(f"Selected meat type: {meat_type}", self.INFO)

    def _set_meat_type_from_button(self, meat_type: str) -> None:
        self.meat_type.set(meat_type)
        self._refresh_meat_buttons()
        self._set_state("Meat Selected", self.INFO, "#17364d")
        self._set_message(f"Physical button selected meat type: {meat_type}", self.SUCCESS)

    def _setup_hardware_buttons(self) -> None:
        try:
            self.button_controller = MeatButtonController(
                lambda meat_type: self.worker_queue.put(lambda: self._set_meat_type_from_button(meat_type))
            )
        except ButtonInputError as exc:
            self._append_log(str(exc))
        except Exception as exc:
            self._append_log(f"Physical button setup error: {exc}")

    def _get_sensor_reader(self) -> MQSensorReader:
        if self.sensor_reader is None:
            self.sensor_reader = MQSensorReader()
        return self.sensor_reader

    def _get_camera_service(self) -> CameraCaptureService:
        if self.camera_service is None:
            self.camera_service = CameraCaptureService()
        return self.camera_service

    def _get_predictor(self) -> HybridFreshnessPredictor:
        if self.predictor is None:
            self.predictor = HybridFreshnessPredictor()
        return self.predictor

    def _update_warmup_state(self) -> None:
        try:
            reader = self._get_sensor_reader()
            remaining = reader.warmup_remaining_seconds()
            if remaining > 0:
                self.warmup_text.set(f"Warm-up remaining: {remaining:.1f}s")
                if not self.sensor_ready:
                    self._set_state("Warming Up", self.WARNING, "#4d3b1d")
            else:
                self.warmup_text.set("Warm-up complete")
                if not self.sensor_ready:
                    self._set_state("Stabilize Sensors", self.INFO, "#16344c")
        except SensorInitializationError as exc:
            self.warmup_text.set("Sensor init failed")
            self._set_message(str(exc), self.DANGER)

    def _update_sensor_display(self, snapshot: dict[str, Any]) -> None:
        self.last_sensor_snapshot = snapshot
        self.sensor_values["nh3_ratio"].set(f"{snapshot['nh3_ratio']:.2f}")
        self.sensor_values["h2s_ratio"].set(f"{snapshot['h2s_ratio']:.2f}")
        self.sensor_values["voc_ratio"].set(f"{snapshot['voc_ratio']:.2f}")
        self.sensor_values["nh3_debug"].set(f"V: {snapshot['nh3_voltage']:.2f} | Rs: {snapshot['nh3_rs']:.1f}")
        self.sensor_values["h2s_debug"].set(f"V: {snapshot['h2s_voltage']:.2f} | Rs: {snapshot['h2s_rs']:.1f}")
        self.sensor_values["voc_debug"].set(f"V: {snapshot['voc_voltage']:.2f} | Rs: {snapshot['voc_rs']:.1f}")

        reasons = snapshot.get("stability_reasons", [])
        if reasons:
            notes = "\n".join(f"- {reason}" for reason in reasons)
        elif snapshot.get("stable"):
            notes = "Sensors are stable and ready to scan."
        else:
            notes = "No stability notes available."
        self._set_text_widget(self.stability_text, notes)

    def _update_environment_display(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("available"):
            self.environment_values["temperature_c"].set(f"{snapshot['temperature_c']:.1f}")
            self.environment_values["humidity_percent"].set(f"{snapshot['humidity_percent']:.1f}")
        else:
            self.environment_values["temperature_c"].set("--")
            self.environment_values["humidity_percent"].set("--")
        self.environment_values["status"].set(snapshot.get("status", "No DHT22 status available."))

    def _request_environment_refresh(self) -> None:
        if self.environment_refresh_in_progress:
            return
        if (time.monotonic() - self.last_environment_poll_monotonic) < config.DHT22_REFRESH_SECONDS:
            return
        self.environment_refresh_in_progress = True

        def worker() -> None:
            try:
                snapshot = self._get_sensor_reader().read_environment()
            except Exception as exc:
                snapshot = {
                    "available": False,
                    "temperature_c": None,
                    "humidity_percent": None,
                    "status": f"DHT22 refresh failed: {exc}",
                }

            def apply_snapshot() -> None:
                self.environment_refresh_in_progress = False
                self.last_environment_poll_monotonic = time.monotonic()
                self._update_environment_display(snapshot)

            self.worker_queue.put(apply_snapshot)

        Thread(target=worker, daemon=True).start()

    def _update_image_preview(self, image_path: Path) -> None:
        self.latest_image_path = image_path
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.contain(image, (520, 360))
        photo = ImageTk.PhotoImage(image)
        self.last_photo_image = photo
        self.image_label.configure(image=photo, text="")

    def _update_prediction_display(self, prediction: dict[str, Any]) -> None:
        self.latest_prediction = prediction
        self.prediction_text.set(prediction["predicted_freshness"])
        if prediction["confidence"] is None:
            self.confidence_text.set("Confidence: Not available")
        else:
            self.confidence_text.set(f"Confidence: {prediction['confidence']:.4f}")
        self.confidence_note_text.set(prediction["confidence_note"])

        class_scores = prediction.get("class_probabilities") or {}
        if class_scores:
            text = "\n".join(f"{label}: {score:.4f}" for label, score in class_scores.items())
        else:
            text = "No class score breakdown available."
        self.class_scores_label.configure(text=text)

    def start_scan(self) -> None:
        def work():
            with self.sensor_lock:
                reader = self._get_sensor_reader()
                if not reader.is_warmed_up():
                    remaining = reader.warmup_remaining_seconds()
                    raise RuntimeError(f"Sensors are still warming up. {remaining:.1f} seconds remaining.")

                self.worker_queue.put(
                    lambda: (
                        self._set_state("Stabilizing", self.INFO, "#17364d"),
                        self._set_message("Stabilizing sensors for scan...", self.INFO),
                    )
                )
                sensor_snapshot = reader.stabilize()
                environment_snapshot = reader.read_environment()

            if not sensor_snapshot.get("stable"):
                raise RuntimeError(
                    "Sensors are not stable yet. " + " ".join(sensor_snapshot.get("stability_reasons", []))
                )

            with self.camera_lock:
                self.worker_queue.put(
                    lambda: (
                        self._set_state("Capturing Image", self.INFO, "#17364d"),
                        self._set_message("Capturing image for scan...", self.INFO),
                    )
                )
                image_path = self._get_camera_service().capture_image()

            self.worker_queue.put(
                    lambda: (
                        self._update_sensor_display(sensor_snapshot),
                        self._update_environment_display(environment_snapshot),
                        self._update_image_preview(image_path),
                        self._set_state("Predicting", self.INFO, "#17364d"),
                        self._set_message("Running hybrid freshness prediction...", self.INFO),
                    )
            )

            predictor = self._get_predictor()
            result = predictor.predict(
                image_path=image_path,
                meat_type=self.meat_type.get(),
                sensor_values=sensor_snapshot["model_sensor_values"],
            )
            predictor.append_prediction_log(result)

            return {
                "sensor_snapshot": sensor_snapshot,
                "environment_snapshot": environment_snapshot,
                "image_path": image_path,
                "prediction": {
                    "predicted_freshness": result.predicted_freshness,
                    "confidence": result.confidence,
                    "confidence_note": result.confidence_note,
                    "class_probabilities": result.class_probabilities,
                },
            }

        def on_success(result: dict[str, Any]) -> None:
            self.sensor_ready = True
            self._update_sensor_display(result["sensor_snapshot"])
            self._update_environment_display(result["environment_snapshot"])
            self._update_image_preview(result["image_path"])
            self._update_prediction_display(result["prediction"])
            self._set_state("Scan Complete", self.SUCCESS, "#184236")
            self._set_message(
                f"Scan complete. {self.meat_type.get()} is predicted as {result['prediction']['predicted_freshness']}.",
                self.SUCCESS,
            )

        self._run_async("Start scan", work, on_success)

    def run(self) -> None:
        self.root.mainloop()

    def _shutdown(self) -> None:
        try:
            if self.button_controller is not None:
                self.button_controller.close()
            if self.sensor_reader is not None:
                self.sensor_reader.close()
        finally:
            self.root.destroy()

    def _exit_fullscreen(self, _event=None) -> None:
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)
        width = min(self.screen_width - 80, 1280)
        height = min(self.screen_height - 120, 820)
        self.root.geometry(f"{width}x{height}+20+20")
        self.root.update_idletasks()

    def _toggle_fullscreen(self, _event=None) -> None:
        if self.is_fullscreen:
            self._exit_fullscreen()
        else:
            self.is_fullscreen = True
            self.root.attributes("-fullscreen", True)


if __name__ == "__main__":
    gui = HybridFreshnessGUI()
    gui.run()
