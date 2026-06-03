from pathlib import Path

from data.load_batadal import get_batadal_feature_columns, load_batadal_dataset
from data.load_skab import get_skab_feature_columns, load_skab_dataset
from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from utils.config import load_config
from utils.logger import get_logger, log_configuration_snapshot, log_missing_data_audit
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
    log_configuration_snapshot(
        logger,
        config=config,
        model_config=model_config,
        experiment_config=experiment_config,
    )

    raw_data_path = Path(config["paths"]["raw_data"])
    skab_dataset = load_skab_dataset(config["datasets"]["skab"], raw_data_path)
    batadal_dataset = load_batadal_dataset(config["datasets"]["batadal"], raw_data_path)
    log_missing_data_audit(
        logger,
        dataset_name="skab",
        summary=PreprocessingPipeline.summarize_missing_values(
            skab_dataset[get_skab_feature_columns(skab_dataset, config["datasets"]["skab"])]
        ),
    )
    log_missing_data_audit(
        logger,
        dataset_name="batadal",
        summary=PreprocessingPipeline.summarize_missing_values(
            batadal_dataset[get_batadal_feature_columns(batadal_dataset, config["datasets"]["batadal"])]
        ),
    )


if __name__ == "__main__":
    main()
