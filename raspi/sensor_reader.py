from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config

try:
    import board
    import busio
except Exception:  # pragma: no cover - hardware-specific import
    board = None
    busio = None

try:
    import adafruit_dht
except Exception:  # pragma: no cover - hardware-specific import
    adafruit_dht = None

try:
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
except Exception:  # pragma: no cover - hardware-specific import
    ADS = None
    AnalogIn = None


LOGGER = logging.getLogger(__name__)


class SensorInitializationError(RuntimeError):
    pass


class SensorReadError(RuntimeError):
    pass


class EnvironmentReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SensorSpec:
    key: str
    channel_index: int
    rl_kohm: float
    ro_kohm: float


class MQSensorReader:
    def __init__(self) -> None:
        self.boot_monotonic = time.monotonic()
        self._ads = None
        self._channels: dict[str, Any] = {}
        self._dht_device = None
        self._initialize_ads()

    def _initialize_ads(self) -> None:
        if board is None or busio is None or ADS is None or AnalogIn is None:
            raise SensorInitializationError(
                "ADS1115 libraries are not available. Install adafruit-blinka and "
                "adafruit-circuitpython-ads1x15 on the Raspberry Pi."
            )

        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c, address=config.ADS_I2C_ADDRESS)
            ads.gain = config.ADS_GAIN
            ads.data_rate = config.ADS_DATA_RATE

            self._ads = ads
            self._channels = {
                "nh3": AnalogIn(ads, config.ADS_CHANNEL_NH3),
                "h2s": AnalogIn(ads, config.ADS_CHANNEL_H2S),
                "voc": AnalogIn(ads, config.ADS_CHANNEL_VOC),
            }
            LOGGER.info("ADS1115 initialized at I2C address 0x%02X", config.ADS_I2C_ADDRESS)
        except Exception as exc:  # pragma: no cover - hardware-specific
            raise SensorInitializationError(f"Failed to initialize ADS1115: {exc}") from exc

    @property
    def sensor_specs(self) -> tuple[SensorSpec, ...]:
        return (
            SensorSpec("nh3", config.ADS_CHANNEL_NH3, config.RL_NH3_KOHM, config.RO_NH3_KOHM),
            SensorSpec("h2s", config.ADS_CHANNEL_H2S, config.RL_H2S_KOHM, config.RO_H2S_KOHM),
            SensorSpec("voc", config.ADS_CHANNEL_VOC, config.RL_VOC_KOHM, config.RO_VOC_KOHM),
        )

    def warmup_remaining_seconds(self) -> float:
        elapsed = time.monotonic() - self.boot_monotonic
        return max(0.0, config.SENSOR_WARMUP_SECONDS - elapsed)

    def is_warmed_up(self) -> bool:
        return self.warmup_remaining_seconds() <= 0.0

    def _get_dht_pin(self):
        if board is None:
            raise EnvironmentReadError("board library is not available for DHT22 access.")

        pin_name = f"D{config.DHT22_GPIO_PIN}"
        gpio_pin = getattr(board, pin_name, None)
        if gpio_pin is None:
            raise EnvironmentReadError(f"board.{pin_name} is not available for DHT22.")
        return gpio_pin

    def _get_dht_device(self):
        if not config.DHT22_ENABLED:
            raise EnvironmentReadError("DHT22 monitoring is disabled in config.py.")
        if adafruit_dht is None:
            raise EnvironmentReadError(
                "DHT22 library is not available. Install adafruit-circuitpython-dht on the Raspberry Pi."
            )
        if self._dht_device is None:
            try:
                self._dht_device = adafruit_dht.DHT22(self._get_dht_pin(), use_pulseio=False)
                LOGGER.info("DHT22 initialized on GPIO%d", config.DHT22_GPIO_PIN)
            except Exception as exc:  # pragma: no cover - hardware-specific
                raise EnvironmentReadError(f"Failed to initialize DHT22: {exc}") from exc
        return self._dht_device

    def _compute_rs_kohm(self, sensor_voltage: float, load_resistance_kohm: float) -> float:
        if sensor_voltage <= 0.001:
            return 999999.0
        safe_voltage = min(max(sensor_voltage, 0.001), config.VC - 0.001)
        return load_resistance_kohm * ((config.VC - safe_voltage) / safe_voltage)

    def _compute_ratio(self, rs_kohm: float, ro_kohm: float) -> float:
        if ro_kohm <= 0:
            return 0.0
        return rs_kohm / ro_kohm

    def _apply_ratio_adjustment(self, ratio_key: str, ratio_value: float) -> float:
        if not getattr(config, "RUNTIME_RATIO_ADJUSTMENT_ENABLED", False):
            return ratio_value
        scale_map = getattr(config, "RUNTIME_RATIO_SCALE", {})
        scale = float(scale_map.get(ratio_key, 1.0))
        return ratio_value * scale

    def _read_channel_average(self, key: str, sample_count: int) -> tuple[float, float]:
        if key not in self._channels:
            raise SensorReadError(f"Unknown sensor channel key: {key}")

        channel = self._channels[key]
        raw_samples: list[float] = []
        voltage_samples: list[float] = []

        try:
            for _ in range(sample_count):
                raw_samples.append(float(channel.value))
                voltage_samples.append(float(channel.voltage))
                time.sleep(config.ADS_SAMPLE_DELAY_SECONDS)
        except Exception as exc:  # pragma: no cover - hardware-specific
            raise SensorReadError(f"Failed while reading ADS1115 channel '{key}': {exc}") from exc

        avg_raw = float(np.mean(raw_samples))
        avg_voltage = float(np.mean(voltage_samples))
        return avg_raw, max(0.0, avg_voltage)

    def read_once(self, sample_count: int | None = None) -> dict[str, Any]:
        sample_count = sample_count or config.ADS_AVERAGE_SAMPLES
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ads_average_samples": int(sample_count),
            "warmup_remaining_seconds": round(self.warmup_remaining_seconds(), 3),
        }

        for spec in self.sensor_specs:
            avg_raw, voltage = self._read_channel_average(spec.key, sample_count)
            rs_kohm = self._compute_rs_kohm(voltage, spec.rl_kohm)
            raw_ratio = self._compute_ratio(rs_kohm, spec.ro_kohm)
            ratio = self._apply_ratio_adjustment(f"{spec.key}_ratio", raw_ratio)

            payload[f"{spec.key}_raw_adc"] = avg_raw
            payload[f"{spec.key}_voltage"] = voltage
            payload[f"{spec.key}_rs"] = rs_kohm
            payload[f"{spec.key}_ratio_raw"] = raw_ratio
            payload[f"{spec.key}_ratio"] = ratio

        LOGGER.debug(
            "Single sensor read | nh3: V=%.4f Rs=%.4f Ratio=%.4f | "
            "h2s: V=%.4f Rs=%.4f Ratio=%.4f | voc: V=%.4f Rs=%.4f Ratio=%.4f",
            payload["nh3_voltage"],
            payload["nh3_rs"],
            payload["nh3_ratio"],
            payload["h2s_voltage"],
            payload["h2s_rs"],
            payload["h2s_ratio"],
            payload["voc_voltage"],
            payload["voc_rs"],
            payload["voc_ratio"],
        )
        return payload

    def stabilize(self, read_count: int | None = None) -> dict[str, Any]:
        read_count = int(read_count or config.STABILIZATION_WINDOW_READS)
        read_count = max(config.STABILIZATION_MIN_READS, min(read_count, config.STABILIZATION_MAX_READS))

        samples = [self.read_once() for _ in range(read_count)]
        aggregated: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "read_count": read_count,
            "stable": True,
            "stability_reasons": [],
            "warmup_remaining_seconds": round(self.warmup_remaining_seconds(), 3),
        }

        numeric_keys = [
            "nh3_raw_adc",
            "h2s_raw_adc",
            "voc_raw_adc",
            "nh3_voltage",
            "h2s_voltage",
            "voc_voltage",
            "nh3_rs",
            "h2s_rs",
            "voc_rs",
            "nh3_ratio",
            "h2s_ratio",
            "voc_ratio",
        ]

        for key in numeric_keys:
            values = [float(sample[key]) for sample in samples if key in sample]
            if not values:
                continue
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

        if read_count < config.STABILIZATION_MIN_READS:
            aggregated["stable"] = False
            aggregated["stability_reasons"].append(
                f"Need at least {config.STABILIZATION_MIN_READS} stabilization reads."
            )

        for ratio_key, limit in config.STABILITY_STD_LIMITS.items():
            observed_std = float(aggregated.get(f"{ratio_key}_std", math.inf))
            if observed_std > limit:
                aggregated["stable"] = False
                aggregated["stability_reasons"].append(
                    f"{ratio_key} std {observed_std:.4f} exceeds limit {limit:.4f}."
                )

        aggregated["model_sensor_values"] = self.to_model_sensor_values(aggregated)

        LOGGER.info(
            "Stabilized sensor window | stable=%s | nh3_ratio=%.4f | h2s_ratio=%.4f | voc_ratio=%.4f",
            aggregated["stable"],
            aggregated.get("nh3_ratio", float("nan")),
            aggregated.get("h2s_ratio", float("nan")),
            aggregated.get("voc_ratio", float("nan")),
        )
        return aggregated

    def capture_baseline(self, read_count: int | None = None) -> dict[str, Any]:
        baseline = self.stabilize(read_count=read_count)
        baseline["mode"] = "baseline"
        return baseline

    def read_environment(self) -> dict[str, Any]:
        if not config.DHT22_ENABLED:
            return {
                "available": False,
                "temperature_c": None,
                "humidity_percent": None,
                "status": "DHT22 monitoring disabled.",
            }

        try:
            dht_device = self._get_dht_device()
        except EnvironmentReadError as exc:
            return {
                "available": False,
                "temperature_c": None,
                "humidity_percent": None,
                "status": str(exc),
            }

        last_error: str | None = None
        for _ in range(config.DHT22_READ_RETRIES):
            try:
                temperature_c = dht_device.temperature
                humidity_percent = dht_device.humidity
                if temperature_c is None or humidity_percent is None:
                    raise EnvironmentReadError("DHT22 returned an empty reading.")

                payload = {
                    "available": True,
                    "temperature_c": float(temperature_c),
                    "humidity_percent": float(humidity_percent),
                    "status": "DHT22 reading OK.",
                }
                LOGGER.debug(
                    "Environment read | temperature=%.2fC | humidity=%.2f%%",
                    payload["temperature_c"],
                    payload["humidity_percent"],
                )
                return payload
            except Exception as exc:  # pragma: no cover - hardware-specific
                last_error = str(exc)
                time.sleep(config.DHT22_RETRY_DELAY_SECONDS)

        return {
            "available": False,
            "temperature_c": None,
            "humidity_percent": None,
            "status": f"DHT22 read failed: {last_error or 'unknown error'}",
        }

    def to_model_sensor_values(self, values: dict[str, Any]) -> dict[str, float]:
        return {
            "nh3_ratio": float(values["nh3_ratio"]),
            "h2s_ratio": float(values["h2s_ratio"]),
            "voc_ratio": float(values["voc_ratio"]),
            "nh3_v": float(values["nh3_voltage"]),
            "nh3_rs": float(values["nh3_rs"]),
            "h2s_v": float(values["h2s_voltage"]),
            "h2s_rs": float(values["h2s_rs"]),
            "voc_v": float(values["voc_voltage"]),
            "voc_rs": float(values["voc_rs"]),
        }

    def close(self) -> None:
        if self._dht_device is not None:
            try:
                self._dht_device.exit()
            except Exception:
                pass
