import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
