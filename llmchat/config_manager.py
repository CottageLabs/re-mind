import json
from pathlib import Path
from typing import Any


class ConfigManager:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
        return self._config

    def save(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            self._config = config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    @property
    def config(self) -> dict[str, Any]:
        return self._config
