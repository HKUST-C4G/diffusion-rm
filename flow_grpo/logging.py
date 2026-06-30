import logging
from typing import Optional

_default_handler: Optional[logging.Handler] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = _get_library_name()

    _configure_root_logger()
    return logging.getLogger(name)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _configure_root_logger() -> None:
    global _default_handler
    if _default_handler is not None:
        return

    _default_handler = logging.StreamHandler()
    root_logger = logging.getLogger(_get_library_name())
    root_logger.addHandler(_default_handler)
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _default_handler.setFormatter(formatter)
