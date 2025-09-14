from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .base import BaseObject
from .axis import Axis
from .cross_section import CrossSection
from .axis_variable import AxisVariable

@dataclass(kw_only=True)
class FoundationObject(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    object_axis_name: str
    cross_section_type: str
    cross_section_name: str
    ref_placement_id: str
    ref_station_offset: float
    station_value: float
    cross_section_points_name: str
    foundation_ref_point_y_offset: float
    foundation_ref_point_x_offset: float
    foundation_level: float
    rotation_angle: float
    axis_name: str
    pier_object_name: List[str]
    point1: str
    point2: str
    point3: str
    point4: str
    thickness: str
    grp: str
    cross_section_ncs2: int
    top_z_offset: float
    bot_z_offset: float
    top_x_offset: float
    top_y_offset: float
    pile_dir_angle: float
    pile_slope: float
    kx: float
    ky: float
    kz: float
    rx: float
    ry: float
    rz: float
    fixation: str
    eval_pier_object_name: str
    eval_station_value: float
    eval_bot_cross_section_points_name: str
    eval_bot_y_offset: float
    eval_bot_z_offset: float
    eval_bot_pier_elevation: float
    internal_placement_id: List[str]
    internal_ref_placement_id: List[float]
    internal_ref_station_offset: List[float]
    internal_station_value: List[float]
    internal_cross_section_ncs: List[float]
    grp_offset: List[float]
    axis_variables: List[Dict] = field(default_factory=list)


    def compute_geometry(
        self,
        *,
        ctx,
        stations_m: Optional[List[float]] = None,
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> Dict[str, object]:
        axis: Optional[Axis] = getattr(self, "axis_obj", None)
        if axis is None:
            return {"ids": [], "stations_mm": np.array([], float), "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": []}

        if stations_m is None:
            S = np.asarray(getattr(axis, "stations", []), float)
            stations_m = (S / 1000.0).tolist()
        stations_mm = np.asarray(stations_m, float) * 1000.0

        axis_results = AxisVariable.evaluate_at_stations_cached(self.axis_variables_obj or [], stations_m)

        section: Optional[CrossSection] = None
        if hasattr(ctx, "crosssec_by_name") and self.cross_section_name:
            section = ctx.crosssec_by_name.get(self.cross_section_name)
        if section is None and hasattr(ctx, "crosssec_by_ncs"):
            ncs_list = getattr(self, "cross_section_ncs", []) or []
            if ncs_list:
                section = ctx.crosssec_by_ncs.get(int(ncs_list[0]))
        if section is None:
            return {"ids": [], "stations_mm": stations_mm, "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": []}

        ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = section.compute_embedded_points(
            axis=axis,
            axis_var_results=axis_results,
            stations_m=stations_m,
            twist_deg=float(twist_deg or 0.0),
            negate_x=negate_x,
        )
        return {
            "ids": ids,
            "stations_mm": S_mm,
            "points_mm": P_mm,
            "local_Y_mm": X_mm,
            "local_Z_mm": Y_mm,
            "loops_idx": loops_idx,
        }