import os
import yaml

def load_yaml_config(yaml_path: str):
    """
    Loads a YAML config file given a relative or absolute path,
    always resolving relative to this script's directory.
    Returns the config as a Python dict.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

from dataclasses import dataclass
import logging


@dataclass
class PromptCompletionPair:
    prompt: str
    completion: str


class LoggerFactory:
    @staticmethod
    def get_logger(name: str,
                   level=logging.INFO,
                   fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(fmt)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger
