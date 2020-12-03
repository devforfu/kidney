import logging
from logging import getLogger


def get_logger(name: str, level: int = logging.DEBUG):
    logger = getLogger(name)
    logger.setLevel(level)
    return logger
