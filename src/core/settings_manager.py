import json
from pathlib import Path
from threading import Lock
from typing import Optional

from pydantic import BaseModel, ValidationError, Field, ConfigDict


DEFAULT_SETTINGS = {
    "docker_url": "tcp://192.168.1.63:2375",
    "docker_name": "vllm_7b",
    "kafka_enabled": False,
    "kafka_url": "192.168.1.17:19092",
    "llm_base_url": None,
    "llm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
}


class AppSettings(BaseModel):
    docker_url: str = Field(default=DEFAULT_SETTINGS["docker_url"])
    docker_name: str = Field(default=DEFAULT_SETTINGS["docker_name"])
    kafka_enabled: bool = Field(default=DEFAULT_SETTINGS["kafka_enabled"])
    kafka_url: str = Field(default=DEFAULT_SETTINGS["kafka_url"])
    llm_base_url: Optional[str] = Field(default=DEFAULT_SETTINGS["llm_base_url"])
    llm_model: str = Field(default=DEFAULT_SETTINGS["llm_model"])

    model_config = ConfigDict(validate_assignment=True)

    def normalized(self) -> "AppSettings":
        """
        Return a copy of the settings with trimmed string fields.
        """
        data = self.model_dump()
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.strip()
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
            settings = AppSettings(**data).normalized()
        except (OSError, json.JSONDecodeError, ValidationError):
            settings = AppSettings().normalized()
            self._write_to_disk(settings)
        return settings

    def _write_to_disk(self, settings: AppSettings) -> None:
        payload = settings.model_dump()
        self._settings_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def get_settings(self) -> AppSettings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def reload(self) -> AppSettings:
        with self._lock:
            self._settings = self._load_from_disk()
            return self._settings

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
