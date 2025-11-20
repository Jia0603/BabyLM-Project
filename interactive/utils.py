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
