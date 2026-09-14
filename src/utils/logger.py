"""
utils/logger.py
---------------
Centralised logging setup. Import `logger` from here everywhere.
"""

import logging
import sys


def get_logger(name: str = "credit_score") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler("outputs/pipeline.log", mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = get_logger()
