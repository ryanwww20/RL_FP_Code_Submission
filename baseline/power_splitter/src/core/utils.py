"""Utility functions for configuration and file handling."""

import dataclasses
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Handle both relative imports (when used as module) and absolute imports (when run as script)
if __name__ == "__main__" or not __package__:
    # Add src directory to path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/core
    src_dir = os.path.dirname(script_dir)  # .../src
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.config import SplitterConfig
else:
    from .config import SplitterConfig


def scalar_from_any(value: Any) -> float | None:
    """Convert stored monitor data to a Python float.
    
    Args:
        value: Value that may be None, array, or scalar
        
    Returns:
        Float value or None if conversion fails
    """
    if value is None:
        return None
    
    import numpy as np
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def load_dict_from_path(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML file describing configuration overrides.
    
    Args:
        path: Path to JSON or YAML file
        
    Returns:
        Dictionary of configuration overrides
        
    Raises:
        RuntimeError: If YAML file is provided but PyYAML is not installed
    """
    _, ext = os.path.splitext(path.lower())
    
    with open(path, "r", encoding="utf-8") as fp:
        if ext in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required to parse YAML config files; "
                    "install via `pip install pyyaml`."
                ) from exc
            return yaml.safe_load(fp) or {}
        
        return json.load(fp)


def merge_dataclass(instance: Any, updates: Dict[str, Any]) -> None:
    """Recursively apply overrides to a dataclass instance.
    
    Args:
        instance: Dataclass instance to update
        updates: Dictionary of field updates (supports nested structures)
        
    Raises:
        ValueError: If nested config structure is invalid
    """
    for field in dataclasses.fields(instance):
        if field.name not in updates:
            continue
        
        value = getattr(instance, field.name)
        override = updates[field.name]
        
        if dataclasses.is_dataclass(value):
            if not isinstance(override, dict):
                raise ValueError(
                    f"Expected mapping for nested config '{field.name}'."
                )
            merge_dataclass(value, override or {})
        else:
            setattr(instance, field.name, override)


def build_config_from_file(
    path: str | None,
    target_ratio: float | None,
    max_iters: int
) -> SplitterConfig:
    """Create SplitterConfig with optional file overrides.
    
    Args:
        path: Optional path to JSON/YAML config file
        target_ratio: Optional target power ratio override
        max_iters: Maximum optimization iterations
        
    Returns:
        Configured SplitterConfig instance
    """
    config = SplitterConfig()
    
    if target_ratio is not None:
        config.optimization.target_ratio = target_ratio
    
    config.optimization.max_iters = max_iters
    
    if path:
        overrides = load_dict_from_path(path)
        if not isinstance(overrides, dict):
            raise ValueError("Top-level config overrides must be a dictionary.")
        merge_dataclass(config, overrides)
    
    return config


def resolve_save_folder(path: str, action: str) -> str:
    """Derive a timestamped save directory for optimization runs.
    
    Args:
        path: Base path for save directory
        action: Action type ('run' creates timestamped subdirectory)
        
    Returns:
        Absolute path to save directory
    """
    if action != "run":
        return path

    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        abs_path = os.path.join(abs_path, timestamp)

    os.makedirs(abs_path, exist_ok=True)
    print(f"Using save folder: {abs_path}")
    return abs_path

