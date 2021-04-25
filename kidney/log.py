import logging
from logging import basicConfig
from logging import getLogger

basicConfig()


def get_logger(name: str, level: int = logging.DEBUG):
    logger = getLogger(name)
    logger.setLevel(level)
    return logger
