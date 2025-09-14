from __future__ import annotations
from dataclasses import dataclass

from .base import BaseObject


@dataclass(kw_only=True)
class BearingArticulation(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    ref_placement_id: str
    ref_station_offset: float
    station_value: float
    pier_object_name: str
    top_cross_section_points_name: str
    top_yoffset: float
    top_xoffset: float
    bot_cross_section_points_name: str
    bot_yoffset: float
    bot_xoffset: float
    kx: float
    ky: float
    kz: float
    rx: float
    ry: float
    rz: float
    grp_offset: float
    grp: str
    bearing_dimensions: str
    rotation_x: float
    rotation_z: float
    fixation: str
    eval_pier_object_name: str
    eval_station_value: float
    eval_top_cross_section_points_name: str
    eval_top_yoffset: float
    eval_top_zoffset: float
    sofi_code: str