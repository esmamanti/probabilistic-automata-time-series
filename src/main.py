from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed

def main():

    config = load_config()

    logger = get_logger()

    logger.info("Project started")

    seed = config["random_seeds"][0]

    set_seed(seed)

    logger.info(f"Seed set to {seed}")

    logger.info("Configuration loaded successfully")

if __name__ == "__main__":
    main()