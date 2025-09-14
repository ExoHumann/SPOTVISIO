from __future__ import annotations
from dataclasses import dataclass, field
from .base import BaseObject  # relative



@dataclass(kw_only=True)
class MainStation(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    placement_id: str
    ref_placement_id: str
    ref_station_offset: float
    station_value: float
    station_type: str
    station_rotation_x: float = 0.0
    station_rotation_z: float = 0.0
    sofi_code: str = field(default="", kw_only=True)
    bearing_dimensions: str | None = None
    
