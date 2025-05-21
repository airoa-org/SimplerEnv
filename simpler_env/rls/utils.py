import tempfile
import os

from ray.tune.logger import UnifiedLogger


def make_logger(path: str):
    def _logger_creator(cfg):
        os.makedirs(path, exist_ok=True)
        logdir = tempfile.mkdtemp(prefix="log_", dir=path)
        return UnifiedLogger(cfg, logdir, loggers=None)
    return _logger_creator
