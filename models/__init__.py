# models/__init__.py
"""
Public API for the models package.

External code (e.g. main.py, spot_loader.py) can import:
    from models import Axis, CrossSection, BaseObject, VisoContext, mapping
    from models import load_from_json, build_viso_object, create_input_for_visualisation

Inside package modules, prefer relative imports to avoid cycles:
    from .base import BaseObject
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

__all__ = [
    # Core dataclasses / objects
    "Axis", "AxisVariable", "CrossSection",
    "DeckObject", "PierObject", "FoundationObject",
    "SecondaryObject", "MainStation", "VisoContext",
    # Shared base + helpers
    "BaseObject", "load_from_json",
    # Visualisation adapter (not plotting)
    "build_viso_object", "create_input_for_visualisation",
    # Config / mapping
    "mapping",
]

# Map exported names -> submodule that defines them
_EXPORT_MAP = {
    # Core models
    "Axis": "models.axis",
    "AxisVariable": "models.axis_variable",
    "CrossSection": "models.cross_section",
    "DeckObject": "models.deck_object",
    "PierObject": "models.pier_object",
    "FoundationObject": "models.foundation_object",
    "SecondaryObject": "models.secondary_object",
    "MainStation": "models.main_station",
    "VisoContext": "models.viso_context",
    "Utils": "utils",
    # Base + helpers
    "BaseObject": "models.base",
    "load_from_json": "models.base",

    # Visualisation adapter (kept in models, but separate from BaseObject)
    "build_viso_object": "models.vis_adapter",
    "create_input_for_visualisation": "models.vis_adapter",

    # Config
    "mapping": "models.mapping",
}

def __getattr__(name: str):
    """Lazy attribute loader to avoid import-time cycles."""
    mod_name = _EXPORT_MAP.get(name)
    if not mod_name:
        raise AttributeError(f"module 'models' has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, name)
    except AttributeError:
        # For module-level variables like `mapping`
        return getattr(mod, name)

if TYPE_CHECKING:
    # Eager imports for static type checkers / IDEs only.
    from .axis import Axis
    from .axis_variable import AxisVariable
    from .cross_section import CrossSection
    from .deck_object import DeckObject
    from .pier_object import PierObject
    from .foundation_object import FoundationObject
    from .secondary_object import SecondaryObject
    from .main_station import MainStation
    from .viso_context import VisoContext

    from .base import BaseObject
    from .vis_adapter import build_viso_object, create_input_for_visualisation

    from .mapping import mapping
