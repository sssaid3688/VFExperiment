"""Minimal stub of vllm.logger — init_logger returning a stdlib logger."""

import logging


def init_logger(name: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)
