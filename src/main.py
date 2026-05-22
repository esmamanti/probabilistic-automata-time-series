from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def _ensure_directories(config: dict) -> None:
    for path_value in config.get("paths", {}).values():
        Path(path_value).mkdir(parents=True, exist_ok=True)


def main():
    config = load_config()
    model_config = load_config("configs/models.yaml")
    experiment_config = load_config("configs/experiments.yaml")

    _ensure_directories(config)
    logger = get_logger()
    logger.info("Project started")

    seed = config["project"]["random_seeds"][0]
    set_seed(seed)

    logger.info("Seed set to %s", seed)
    logger.info("Configuration loaded successfully")
    logger.info("Loaded model configuration sections: %s", ", ".join(model_config.keys()))
    logger.info("Loaded experiment configuration sections: %s", ", ".join(experiment_config.keys()))


if __name__ == "__main__":
    main()
