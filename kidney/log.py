import logging
from logging import basicConfig, getLogger


def init_logging() -> logging.Logger:
    basicConfig()
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger
