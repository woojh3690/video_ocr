import json
from pathlib import Path
from threading import Lock
from typing import Optional

from pydantic import BaseModel, ValidationError, Field, ConfigDict


DEFAULT_SETTINGS = {
    "docker_enabled": False,
    "docker_url": "",
    "detector_docker_name": "",
    "recognizer_docker_name": "",
    "kafka_enabled": False,
    "kafka_url": "192.168.0.2:19092",
    "detector_llm_base_url": None,
    "detector_llm_model": "datalab-to/chandra-ocr-2",
    "recognizer_llm_base_url": None,
    "recognizer_llm_model": "PaddlePaddle/PaddleOCR-VL-1.5",
    "llm_api_key": None,
}


def normalize_llm_base_url(base_url: Optional[str]) -> Optional[str]:
    if base_url is None:
        return None

    value = base_url.strip().rstrip("/")
    if not value:
        return None
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


class AppSettings(BaseModel):
    docker_enabled: bool = Field(default=DEFAULT_SETTINGS["docker_enabled"])
    docker_url: str = Field(default=DEFAULT_SETTINGS["docker_url"])
    detector_docker_name: str = Field(default=DEFAULT_SETTINGS["detector_docker_name"])
    recognizer_docker_name: str = Field(default=DEFAULT_SETTINGS["recognizer_docker_name"])
    kafka_enabled: bool = Field(default=DEFAULT_SETTINGS["kafka_enabled"])
    kafka_url: str = Field(default=DEFAULT_SETTINGS["kafka_url"])
    detector_llm_base_url: Optional[str] = Field(default=DEFAULT_SETTINGS["detector_llm_base_url"])
    detector_llm_model: str = Field(default=DEFAULT_SETTINGS["detector_llm_model"])
    recognizer_llm_base_url: Optional[str] = Field(default=DEFAULT_SETTINGS["recognizer_llm_base_url"])
    recognizer_llm_model: str = Field(default=DEFAULT_SETTINGS["recognizer_llm_model"])
    llm_api_key: Optional[str] = Field(default=DEFAULT_SETTINGS["llm_api_key"])

    model_config = ConfigDict(validate_assignment=True)

    def normalized(self) -> "AppSettings":
        """
        Return a copy of the settings with trimmed string fields.
        """
        data = self.model_dump()
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.strip()
        data["detector_llm_base_url"] = normalize_llm_base_url(data.get("detector_llm_base_url"))
        data["recognizer_llm_base_url"] = normalize_llm_base_url(data.get("recognizer_llm_base_url"))
        return AppSettings(**data)


class SettingsManager:
    def __init__(self, settings_path: Optional[Path] = None):
        if settings_path is None:
            settings_path = Path(__file__).resolve().parent.parent / "settings.json"
        self._settings_path = settings_path
        self._lock = Lock()
        self._settings = self._load_from_disk()

    @property
    def settings_path(self) -> Path:
        return self._settings_path

    def _load_from_disk(self) -> AppSettings:
        if not self._settings_path.exists():
            settings = AppSettings().normalized()
            self._write_to_disk(settings)
            return settings

        try:
            raw = self._settings_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw else {}
            if isinstance(data, dict):
                data = self._migrate_legacy_settings(data)
            settings = AppSettings(**data).normalized()
        except (OSError, json.JSONDecodeError, ValidationError):
            settings = AppSettings().normalized()
            self._write_to_disk(settings)
        return settings

    def _migrate_legacy_settings(self, data: dict) -> dict:
        migrated = dict(data)
        legacy_base_url = migrated.get("llm_base_url")
        if legacy_base_url and not migrated.get("detector_llm_base_url"):
            migrated["detector_llm_base_url"] = legacy_base_url
        if legacy_base_url and not migrated.get("recognizer_llm_base_url"):
            migrated["recognizer_llm_base_url"] = legacy_base_url
        return migrated

    def _write_to_disk(self, settings: AppSettings) -> None:
        payload = settings.model_dump()
        self._settings_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def get_settings(self) -> AppSettings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def update(self, **updates) -> AppSettings:
        with self._lock:
            data = self._settings.model_dump()
            data.update(updates)
            new_settings = AppSettings(**data).normalized()
            self._write_to_disk(new_settings)
            self._settings = new_settings
            return new_settings


settings_manager = SettingsManager()


def get_settings() -> AppSettings:
    return settings_manager.get_settings()


def update_settings(**updates) -> AppSettings:
    return settings_manager.update(**updates)
