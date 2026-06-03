import logging
import os
from pathlib import Path
import json

LOG_PATH = "results/logs"

os.makedirs(LOG_PATH, exist_ok=True)

logger = logging.getLogger("project")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(os.path.join(LOG_PATH, "project.log"))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger():
    return logger


def log_configuration_snapshot(logger_instance: logging.Logger, *, config: dict, model_config: dict, experiment_config: dict) -> None:
    logger_instance.info("Configuration loaded successfully")
    logger_instance.info("Loaded model configuration sections: %s", ", ".join(model_config.keys()))
    logger_instance.info("Loaded experiment configuration sections: %s", ", ".join(experiment_config.keys()))
    logger_instance.info("Resolved project config: %s", json.dumps(config, sort_keys=True))
    logger_instance.info("Resolved model config: %s", json.dumps(model_config, sort_keys=True))
    logger_instance.info("Resolved experiment config: %s", json.dumps(experiment_config, sort_keys=True))


def log_missing_data_audit(logger_instance: logging.Logger, *, dataset_name: str, summary: dict[str, object]) -> None:
    logger_instance.info("Missing data audit for %s: %s", dataset_name.upper(), json.dumps(summary, sort_keys=True))


def append_json_record(file_name: str, payload: dict[str, object]) -> None:
    path = Path(LOG_PATH) / file_name
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
