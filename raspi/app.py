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
from meat_classifier import MeatClassificationResult, MeatClassifierLoadError, MeatClassifierService
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
        self.root.title("FreshTo")
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
        self.meat_classifier: MeatClassifierService | None = None
        self.button_controller: MeatButtonController | None = None

        self.system_state = tk.StringVar(value="Initializing")
        self.message_text = tk.StringVar(value="Starting FreshTo...")
        self.warmup_text = tk.StringVar(value="Warm-up status unavailable")
        self.model_mode_text = tk.StringVar(value=f"Freshness mode: {getattr(config, 'MODEL_MODE', 'hybrid')}")
        self.detected_meat_text = tk.StringVar(value="--")
        self.detected_meat_confidence_text = tk.StringVar(value="Confidence: --")
        self.detected_meat_note_text = tk.StringVar(value="No meat detection yet.")
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
        self.latest_meat_detection: MeatClassificationResult | None = None
        self.sensor_ready = False
        self.last_photo_image = None
        self.environment_refresh_in_progress = False
        self.last_environment_poll_monotonic = 0.0
        self.sensor_refresh_in_progress = False
        self.last_sensor_poll_monotonic = 0.0
        self.preview_refresh_in_progress = False
        self.last_preview_poll_monotonic = 0.0
        self.scan_in_progress = False

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

        ttk.Label(header, text="FreshTo", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Hybrid meat freshness detection using live MQ sensing, Pi camera feed, and machine learning prediction.",
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

        self.content.columnconfigure(0, weight=3)
        self.content.columnconfigure(1, weight=2)
        self.content.rowconfigure(0, weight=3)
        self.content.rowconfigure(1, weight=2)
        self.content.rowconfigure(2, weight=1)

        self._build_preview_panel(self.content)
        self._build_result_panel(self.content)
        self._build_sensors_panel(self.content)
        self._build_log_panel(self.content)

    def _build_sensors_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        panel.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 12))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Live Sensor and Environment Data", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="Sensor ratios, voltage, resistance, temperature, and humidity update automatically even before scanning.", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 14))

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
        ttk.Label(notes_panel, text="Live Notes", style="PanelTitle.TLabel").pack(anchor="w")
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
        self._set_text_widget(self.stability_text, "Waiting for live sensor updates.")

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
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        ttk.Label(panel, text="Live Camera Feed", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")

        image_frame = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1)
        image_frame.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = tk.Label(
            image_frame,
            text="Waiting for camera preview...",
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            font=("Segoe UI", 12),
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    def _build_result_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        panel.grid(row=0, column=1, sticky="nsew", pady=(0, 12))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Prediction Result", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            panel,
            text="Any physical hardware button starts a scan. Meat type is now detected automatically from the camera image.",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))
        tk.Label(
            panel,
            textvariable=self.model_mode_text,
            bg=self.PANEL,
            fg=self.INFO,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(0, 12))

        trigger_frame = tk.Frame(panel, bg=self.PANEL)
        trigger_frame.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(trigger_frame, text="Physical Scan Triggers", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        trigger_row = tk.Frame(trigger_frame, bg=self.PANEL)
        trigger_row.grid(row=1, column=0, sticky="ew")
        for idx, label in enumerate(config.MEAT_TYPES):
            badge = tk.Label(
                trigger_row,
                text=f"{label} Button",
                bg="#183850",
                fg=self.TEXT,
                padx=14,
                pady=12,
                font=("Segoe UI", 12, "bold"),
            )
            badge.grid(row=0, column=idx, padx=(0 if idx == 0 else 8, 0), sticky="ew")
            trigger_row.columnconfigure(idx, weight=1)

        detection_panel = tk.Frame(panel, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        detection_panel.grid(row=4, column=0, sticky="ew")
        tk.Label(detection_panel, text="Detected Meat Type", bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 11)).pack(anchor="center")
        tk.Label(detection_panel, textvariable=self.detected_meat_text, bg=self.CARD, fg=self.TEXT, font=("Segoe UI", 26, "bold")).pack(anchor="center", pady=(8, 4))
        tk.Label(detection_panel, textvariable=self.detected_meat_confidence_text, bg=self.CARD, fg=self.INFO, font=("Segoe UI", 12, "bold")).pack(anchor="center")
        tk.Label(
            detection_panel,
            textvariable=self.detected_meat_note_text,
            bg=self.CARD,
            fg=self.MUTED,
            wraplength=320,
            justify="center",
            font=("Segoe UI", 10),
        ).pack(anchor="center", pady=(8, 0))

        prediction_panel = tk.Frame(panel, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        prediction_panel.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        tk.Label(prediction_panel, text="Predicted Freshness", bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 11)).pack(anchor="center")
        tk.Label(prediction_panel, textvariable=self.prediction_text, bg=self.CARD, fg=self.TEXT, font=("Segoe UI", 30, "bold")).pack(anchor="center", pady=(8, 4))
        tk.Label(prediction_panel, textvariable=self.confidence_text, bg=self.CARD, fg=self.SUCCESS, font=("Segoe UI", 12, "bold")).pack(anchor="center")
        tk.Label(
            prediction_panel,
            textvariable=self.confidence_note_text,
            bg=self.CARD,
            fg=self.MUTED,
            wraplength=320,
            justify="center",
            font=("Segoe UI", 10),
        ).pack(anchor="center", pady=(8, 0))

        scores_panel = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        scores_panel.grid(row=6, column=0, sticky="ew", pady=(12, 12))
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

        message_panel = tk.Frame(panel, bg=self.PANEL_ALT, highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=14)
        message_panel.grid(row=7, column=0, sticky="ew", pady=(0, 12))
        self.message_label = tk.Label(
            message_panel,
            textvariable=self.message_text,
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            justify="left",
            wraplength=320,
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.message_label.pack(fill="x")

        exit_button = tk.Button(
            panel,
            text="Exit App",
            command=self.root.destroy,
            bg="#8a3243",
            fg=self.TEXT,
            activebackground="#8a3243",
            activeforeground=self.TEXT,
            bd=0,
            padx=18,
            pady=12,
            font=("Segoe UI", 12, "bold"),
        )
        exit_button.grid(row=8, column=0, sticky="ew")

    def _build_log_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=18)
        panel.grid(row=2, column=0, columnspan=2, sticky="nsew")
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
        self._request_live_sensor_refresh()
        self._request_environment_refresh()
        self._request_camera_preview_refresh()
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
        self.scan_in_progress = False
        if task_name.lower() == "start scan":
            self._clear_meat_detection_display("The latest scan failed before meat detection could complete.")
            self._clear_prediction_display("The latest scan failed. Please try again.")
        self._set_state("Error", self.DANGER, "#4a1f28")
        self._set_message(f"{task_name} failed: {exc}", self.DANGER)

    def _handle_scan_button_press(self, button_label: str) -> None:
        self._set_state("Scan Requested", self.INFO, "#17364d")
        self._set_message(
            f"Physical button pressed ({button_label}). Starting automatic meat detection scan...",
            self.SUCCESS,
        )
        self.start_scan(trigger_source=button_label)

    def _setup_hardware_buttons(self) -> None:
        try:
            self.button_controller = MeatButtonController(
                lambda button_label: self.worker_queue.put(lambda: self._handle_scan_button_press(button_label))
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

    def _get_meat_classifier(self) -> MeatClassifierService:
        if self.meat_classifier is None:
            self.meat_classifier = MeatClassifierService()
        return self.meat_classifier

    def _get_predictor(self) -> HybridFreshnessPredictor:
        if self.predictor is None:
            self.predictor = HybridFreshnessPredictor()
            self.model_mode_text.set(f"Freshness mode: {self.predictor.mode}")
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
                    self._set_state("Ready to Scan", self.INFO, "#16344c")
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
        else:
            notes = f"Live view uses {snapshot.get('ads_average_samples', config.ADS_AVERAGE_SAMPLES)} averaged sensor reads per refresh."
        self._set_text_widget(self.stability_text, notes)

    def _request_live_sensor_refresh(self) -> None:
        if self.sensor_refresh_in_progress:
            return
        if (time.monotonic() - self.last_sensor_poll_monotonic) < config.SENSOR_LIVE_REFRESH_SECONDS:
            return
        self.sensor_refresh_in_progress = True

        def worker() -> None:
            try:
                with self.sensor_lock:
                    snapshot = self._get_sensor_reader().read_once()
            except Exception as exc:
                snapshot = None
                error_message = str(exc)
            else:
                error_message = None

            def apply_snapshot() -> None:
                self.sensor_refresh_in_progress = False
                self.last_sensor_poll_monotonic = time.monotonic()
                if snapshot is not None:
                    self._update_sensor_display(snapshot)
                elif error_message and "Failed to initialize ADS1115" in error_message:
                    # Keep the UI quiet while the hardware is still being fixed.
                    pass

            self.worker_queue.put(apply_snapshot)

        Thread(target=worker, daemon=True).start()

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

    def _update_preview_from_image(self, image: Image.Image) -> None:
        photo = ImageTk.PhotoImage(image)
        self.last_photo_image = photo
        self.image_label.configure(image=photo, text="")

    def _request_camera_preview_refresh(self) -> None:
        if self.preview_refresh_in_progress:
            return
        if (time.monotonic() - self.last_preview_poll_monotonic) < config.CAMERA_PREVIEW_REFRESH_SECONDS:
            return
        self.preview_refresh_in_progress = True

        def worker() -> None:
            preview_image = None
            error_message = None
            acquired = self.camera_lock.acquire(blocking=False)
            try:
                if not acquired:
                    error_message = "Camera busy."
                    return
                preview_image = self._get_camera_service().get_preview_image()
            except Exception as exc:
                error_message = str(exc)
            finally:
                if acquired:
                    self.camera_lock.release()

            def apply_preview() -> None:
                self.preview_refresh_in_progress = False
                self.last_preview_poll_monotonic = time.monotonic()
                if preview_image is not None:
                    self._update_preview_from_image(preview_image)
                elif error_message and error_message != "Camera busy.":
                    self.image_label.configure(text="Camera preview unavailable.", image="")

            self.worker_queue.put(apply_preview)

        Thread(target=worker, daemon=True).start()

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

    def _update_meat_detection_display(self, result: MeatClassificationResult) -> None:
        self.latest_meat_detection = result
        if result.predicted_class == config.MEAT_CLASSIFIER_NOT_MEAT_LABEL:
            self.detected_meat_text.set("No valid meat")
            self.detected_meat_note_text.set("Freshness prediction was skipped because the image was classified as not_meat.")
        else:
            detected_name = result.hybrid_meat_type or result.predicted_class.replace("_", " ").title()
            self.detected_meat_text.set(detected_name)
            self.detected_meat_note_text.set("Detected from the new Option B image classifier before running freshness prediction.")
        self.detected_meat_confidence_text.set(f"Confidence: {result.confidence:.4f}")

    def _clear_meat_detection_display(self, note: str = "Waiting for a new scan.") -> None:
        self.latest_meat_detection = None
        self.detected_meat_text.set("--")
        self.detected_meat_confidence_text.set("Confidence: --")
        self.detected_meat_note_text.set(note)

    def _clear_prediction_display(self, note: str = "Scanning in progress.") -> None:
        self.latest_prediction = None
        self.prediction_text.set("--")
        self.confidence_text.set("Confidence: --")
        self.confidence_note_text.set(note)
        self.class_scores_label.configure(text="Waiting for a new prediction...")

    def start_scan(self, trigger_source: str | None = None) -> None:
        if self.scan_in_progress:
            self._set_message("A scan is already running. Please wait.", self.WARNING)
            return
        self.scan_in_progress = True
        self._clear_meat_detection_display("Running meat detection from the latest captured image.")
        self._clear_prediction_display("Scanning in progress.")

        def work():
            with self.camera_lock:
                self.worker_queue.put(
                    lambda: (
                        self._set_state("Capturing Image", self.INFO, "#17364d"),
                        self._set_message("Capturing image for automatic meat detection...", self.INFO),
                    )
                )
                image_path = self._get_camera_service().capture_image()

            meat_classifier = self._get_meat_classifier()
            self.worker_queue.put(
                lambda: (
                    self._update_image_preview(image_path),
                    self._set_state("Detecting Meat", self.INFO, "#17364d"),
                    self._set_message("Running meat classifier on the captured image...", self.INFO),
                )
            )
            meat_detection = meat_classifier.classify(image_path)

            if not meat_detection.is_valid_meat:
                return {
                    "image_path": image_path,
                    "meat_detection": meat_detection,
                    "prediction": None,
                    "sensor_snapshot": None,
                    "environment_snapshot": None,
                    "skipped_reason": "No valid meat detected",
                    "trigger_source": trigger_source,
                }

            with self.sensor_lock:
                reader = self._get_sensor_reader()
                if not reader.is_warmed_up():
                    remaining = reader.warmup_remaining_seconds()
                    raise RuntimeError(f"Sensors are still warming up. {remaining:.1f} seconds remaining.")

                self.worker_queue.put(
                    lambda: (
                        self._set_state("Collecting Scan Data", self.INFO, "#17364d"),
                        self._set_message(
                            f"Detected {meat_detection.hybrid_meat_type}. Collecting MQ sensor window for freshness scan...",
                            self.INFO,
                        ),
                    )
                )
                sensor_snapshot = reader.stabilize(read_count=config.SCAN_SUMMARY_READS)
                environment_snapshot = reader.read_environment()

            predictor = self._get_predictor()
            prediction_mode = getattr(predictor, "mode", getattr(config, "MODEL_MODE", "hybrid"))
            self.worker_queue.put(
                lambda: (
                    self._update_sensor_display(sensor_snapshot),
                    self._update_environment_display(environment_snapshot),
                    self._update_image_preview(image_path),
                    self._update_meat_detection_display(meat_detection),
                    self._set_state("Predicting", self.INFO, "#17364d"),
                    self._set_message(
                        f"Detected {meat_detection.hybrid_meat_type}. Running {prediction_mode} freshness prediction...",
                        self.INFO,
                    ),
                )
            )

            sensor_input = {
                "nh3_ratio": sensor_snapshot["model_sensor_values"]["nh3_ratio"],
                "nh3_ratio_raw": sensor_snapshot["model_sensor_values"].get("nh3_ratio_raw"),
                "h2s_ratio": sensor_snapshot["model_sensor_values"]["h2s_ratio"],
                "h2s_ratio_raw": sensor_snapshot["model_sensor_values"].get("h2s_ratio_raw"),
                "voc_ratio": sensor_snapshot["model_sensor_values"]["voc_ratio"],
                "voc_ratio_raw": sensor_snapshot["model_sensor_values"].get("voc_ratio_raw"),
            }
            ratio_summary_features = {
                key: value
                for key, value in sensor_snapshot.get("sensor_summary_features", {}).items()
                if key.startswith("sensor_nh3_ratio_")
                or key.startswith("sensor_h2s_ratio_")
                or key.startswith("sensor_voc_ratio_")
            }
            sensor_input.update(ratio_summary_features)
            result = predictor.predict(
                image_path=image_path,
                meat_type=meat_detection.hybrid_meat_type or "Chicken",
                sensor_values=sensor_input,
            )
            predictor.append_prediction_log(result)

            return {
                "meat_detection": meat_detection,
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
            self.scan_in_progress = False
            self.sensor_ready = True
            self._update_image_preview(result["image_path"])
            self._update_meat_detection_display(result["meat_detection"])

            if result["prediction"] is None:
                self._clear_prediction_display("No valid meat detected. Freshness model was not executed.")
                self._set_state("No Valid Meat", self.WARNING, "#4d3b1d")
                self._set_message(
                    "Scan stopped because the captured image was classified as not_meat. No valid meat detected.",
                    self.WARNING,
                )
                return

            self._update_sensor_display(result["sensor_snapshot"])
            self._update_environment_display(result["environment_snapshot"])
            self._update_prediction_display(result["prediction"])
            detected_name = result["meat_detection"].hybrid_meat_type or result["meat_detection"].predicted_class
            self._set_state("Scan Complete", self.SUCCESS, "#184236")
            self._set_message(
                f"Scan complete. Detected {detected_name} and predicted {result['prediction']['predicted_freshness']}.",
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
