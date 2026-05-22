import json
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    path = Path(config_path)

    with path.open("r") as file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to load YAML config files") from exc
            return yaml.safe_load(file)

        if path.suffix.lower() == ".json":
            return json.load(file)

        raise ValueError(f"Unsupported config file format: {path.suffix}")