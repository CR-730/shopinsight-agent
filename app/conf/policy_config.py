"""Policy configuration used by guard and semantic resolver layers."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_policy_config() -> dict[str, Any]:
    """Load policy config once per process."""

    path = Path(__file__).parents[2] / "conf" / "policy_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
