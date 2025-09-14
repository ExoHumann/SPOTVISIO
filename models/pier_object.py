from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseObject
from .cross_section import CrossSection
from .axis import Axis
from .axis_variable import AxisVariable


from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .base import BaseObject

@dataclass(kw_only=True)
class PierObject(BaseObject):
    # identity / metadata
    no: str = ""
    class_name: str = "PierObject"
    type: str = ""
    description: str = ""
    name: str = ""
    inactive: str = "0"

    # axis
    object_axis_name: str = ""                 # leave if you also use axis_name elsewhere
    axis_variables: List[Dict] = field(default_factory=list)

    # cross-section references (DEFAULTS ADDED)
    bot_cross_section_name: str = ""           # <- default added
    top_cross_section_name: str = ""           # <- default added
    internal_cross_section_ncs: List[int] = field(default_factory=list)

    # geometry controls (DEFAULT ADDED)
    slices: int = 0                             # <- default added (0 => use axis stations)
    top_zoffset: float = 0.0
    bot_pier_elevation: float = 0.0
    rotation_angle: float = 0.0

    # other fields you already had (give safe defaults)
    grp: int = 0
    fixation: str = ""
    internal_placement_id: List[str] = field(default_factory=list)
    internal_ref_placement_id: List[str] = field(default_factory=list)
    internal_ref_station_offset: List[float] = field(default_factory=list)
    grp_offset: List[float] = field(default_factory=list)

    # optional helper names used elsewhere
    top_cross_section_points_name: str = ""
    bot_cross_section_points_name: str = ""
    internal_station_value: List[float] = field(default_factory=list)

    # ------------------------------------------------------------
    # New: simple metadata formatter (optional; preserves your style)
    # ------------------------------------------------------------
    def get_object_metadata(self) -> Dict:
        data = asdict(self)
        # compress noisy fields like base does
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = "<Axis object>" if getattr(self, "axis_obj", None) is not None else None
        # remove UI-only noise
        data.pop('axis_variables_obj', None)
        data.pop('axis_obj', None)
        return data

    # ------------------------------------------------------------
    # Ownership: pier-specific geometry
    # ------------------------------------------------------------
    def compute_length_mm(self) -> float:
        """
        Pier length in mm.
        Uses bottom pier elevation and top_zoffset. Assumes inputs in meters unless clearly mm.
        """
        bot = float(self.bot_pier_elevation or 0.0)
        top = float(self.top_zoffset or 0.0)
        # Heuristic: values < 1000 are likely meters -> convert to mm
        if abs(bot) < 1000.0: bot *= 1000.0
        if abs(top) < 1000.0: top *= 1000.0
        return float(top - bot)

    def _derive_stations_mm(self, *, axis: Axis, slices: Optional[int] = None) -> np.ndarray:
        """
        Build stations along the axis for this pier:
        - if user_stations set on the object -> use those
        - else if slices set -> interpolate evenly between visible axis range
        - else -> use full axis stations (dense path)
        """
        if getattr(self, "user_stations", None):
            s_mm = np.asarray(self.user_stations, float) * 1000.0
            return s_mm

        if slices is None:
            slices = int(getattr(self, "slices", 0) or 0)

        axS = np.asarray(getattr(axis, "stations", []), float)
        if axS.size == 0:
            return np.zeros((0,), float)

        smin, smax = float(np.min(axS)), float(np.max(axS))
        if slices and slices > 1:
            return np.linspace(smin, smax, num=int(slices), dtype=float)

        # fallback: use original axis stations
        return axS.copy()

    # ------------------------------------------------------------
    # Main: compute and embed geometry
    # ------------------------------------------------------------
    def compute_geometry(
        self,
        *,
        ctx,                                # VisoContext (needed to fetch CrossSection by NCS/name if desired)
        stations_m: Optional[List[float]] = None,
        slices: Optional[int] = None,
        twist_deg: float = 0.0,             # extra in-plane rotation, defaults to 0
        negate_x: bool = True,              # historical: local X sign flip
    ) -> Dict[str, object]:
        """
        Returns:
          {
            "ids": List[str],
            "stations_mm": (S,),
            "points_mm": (S,N,3),
            "local_Y_mm": (S,N),
            "local_Z_mm": (S,N),
            "loops_idx": List[np.ndarray],
            "length_mm": float,
          }
        """
        # 1) Axis ready?
        axis: Optional[Axis] = getattr(self, "axis_obj", None)
        if axis is None:
            # If your BaseObject usually constructs axis_obj from internal data, call that here.
            # Otherwise, we require it to be resolved by the loader before calling geometry.
            return {"ids": [], "stations_mm": np.array([], float), "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": [], "length_mm": 0.0}

        # 2) Stations (mm)
        if stations_m is not None:
            stations_mm = np.asarray(stations_m, float) * 1000.0
        else:
            stations_mm = self._derive_stations_mm(axis=axis, slices=slices)

        if stations_mm.size == 0:
            return {"ids": [], "stations_mm": stations_mm, "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": [], "length_mm": self.compute_length_mm()}

        # 3) Axis variables evaluation -> rows of {name: value}
        #    Use AxisVariable.evaluate_at_stations_cached (already vectorized & cached).
        #    NOTE: AxisVariable expects 'stations' in the same units the curves are defined in
        #    (your codebase typically uses meters on input). Convert back to meters for eval.
        stations_m_eval = stations_mm / 1000.0
        axis_results = AxisVariable.evaluate_at_stations_cached(self.axis_variables_obj or [], stations_m_eval.tolist())

        # 4) Cross-sections: prefer named top/bottom; otherwise, resolve by NCS via ctx if available.
        top_cs: Optional[CrossSection] = None
        bot_cs: Optional[CrossSection] = None

        if hasattr(ctx, "crosssec_by_name"):
            if self.top_cross_section_name:
                top_cs = ctx.crosssec_by_name.get(self.top_cross_section_name)
            if self.bot_cross_section_name:
                bot_cs = ctx.crosssec_by_name.get(self.bot_cross_section_name)

        if (top_cs is None or bot_cs is None) and hasattr(ctx, "crosssec_by_ncs"):
            ncs_list = getattr(self, "internal_cross_section_ncs", []) or []
            if len(ncs_list) >= 1 and bot_cs is None:
                bot_cs = ctx.crosssec_by_ncs.get(int(ncs_list[0]))
            if len(ncs_list) >= 2 and top_cs is None:
                top_cs = ctx.crosssec_by_ncs.get(int(ncs_list[1]))

        # Minimal fallback: if only one available, use it for both to stay functional
        if top_cs is None and bot_cs is not None:
            top_cs = bot_cs
        if bot_cs is None and top_cs is not None:
            bot_cs = top_cs
        if top_cs is None and bot_cs is None:
            return {"ids": [], "stations_mm": stations_mm, "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": [], "length_mm": self.compute_length_mm()}

        # For piers we usually use the *top* cross-section shape for the body contour;
        # if your design needs a tween between bottom/top, you can blend X/Y arrays here.
        section = top_cs

        # 5) Compute local points (vectorized) and embed through Axis
        ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = section.compute_embedded_points(
            axis=axis,
            axis_var_results=axis_results,
            stations_m=stations_m_eval.tolist(),
            twist_deg=float(self.rotation_angle or 0.0) + float(twist_deg or 0.0),
            negate_x=negate_x,
        )

        return {
            "ids": ids,
            "stations_mm": S_mm,
            "points_mm": P_mm,
            "local_Y_mm": X_mm,
            "local_Z_mm": Y_mm,
            "loops_idx": loops_idx,
            "length_mm": self.compute_length_mm(),
        }
