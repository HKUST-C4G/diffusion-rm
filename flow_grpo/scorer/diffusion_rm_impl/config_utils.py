import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class AttrDict(dict):
    """dict with dot access (nested)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(x: Any) -> Any:
    if isinstance(x, Mapping):
        return AttrDict({k: _to_attrdict(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_attrdict(v) for v in x]
    return x


def load_config(config_path: str) -> AttrDict:
    """Load Diffusion-RM training config exported as json/yaml. Returns AttrDict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    with open(config_path, "r") as f:
        raw = f.read()

    # Most outputs are saved as JSON (often with .json extension) but are YAML-compatible.
    try:
        data = json.loads(raw)
    except Exception:
        try:
            import yaml  # optional dependency
        except Exception as e:
            raise RuntimeError(
                "Failed to parse config as JSON and PyYAML is not installed for YAML fallback."
            ) from e
        data = yaml.safe_load(raw)

    return _to_attrdict(data)

