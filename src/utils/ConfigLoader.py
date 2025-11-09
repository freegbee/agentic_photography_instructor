import os
from pathlib import Path
from typing import Optional, Dict, Union, Iterable, List

import yaml


class ConfigLoader:

    def __init__(self, base_dir: Optional[Path] = None, default_config: str = "configs/default.yaml"):
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.default_config = default_config

    def _deep_update(self, base: Dict, override: Dict) -> Dict:
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = self._deep_update(base[k], v)
            else:
                base[k] = v
        return base

    def _load_yaml_file(self, path: Union[str, Path]) -> Dict:
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        if not p.exists():
            # Falls eine optionale Datei nicht existiert, leise ignorieren
            return {}
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_env_path(self, env: str) -> Optional[Path]:
        # Sucht nach configs/<env>.yaml relativ zu base_dir
        candidate = self.base_dir / "configs" / f"{env}.yaml"
        return candidate if candidate.exists() else None

    def load(
            self,
            cfg_paths: Optional[Union[str, Path, Iterable[Union[str, Path]]]] = None,
            env: Optional[str] = None,
            include_default_if_missing: bool = True,
    ) -> Dict:
        """
        Lädt und merged Konfigurationsdateien.
        - cfg_paths: einzelne Datei oder Liste von Dateien in Reihenfolge (spätere überschreiben frühere).
        - env: optionaler Umgebungsname; versucht configs/<env>.yaml anzuhängen.
        - include_default_if_missing: wenn True und cfg_paths leer ist, wird default_config verwendet.
        """
        paths: List[Union[str, Path]] = []
        if cfg_paths is None:
            if include_default_if_missing and self.default_config:
                paths.append(self.default_config)
        elif isinstance(cfg_paths, (str, Path)):
            paths.append(cfg_paths)
        else:
            paths.extend(list(cfg_paths))

        # Optional: env-override ans Ende hängen
        if env:
            env_path = self._resolve_env_path(env)
            if env_path:
                paths.append(env_path)

        config: Dict = {}
        for p in paths:
            part = self._load_yaml_file(p)
            config = self._deep_update(config, part)

        return config

    @staticmethod
    def load_dataset_config(dataset_id: str) -> Dict:
        config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
        if not config:
            raise ValueError("Error loading config")

        image_acq = config.get('image_acquisition')
        if dataset_id not in image_acq:
            raise ValueError(f"No configuration found for dataset_id: {dataset_id}")

        return config['image_acquisition'][dataset_id]