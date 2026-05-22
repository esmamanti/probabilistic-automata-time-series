import logging
import os

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