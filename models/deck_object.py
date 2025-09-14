from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .base import BaseObject
from .axis import Axis
from .cross_section import CrossSection
from .axis_variable import AxisVariable
def _resolve_section_key(ctx, key):
    if key is None: return None
    if isinstance(key, str):
        return getattr(ctx, "crosssec_by_name", {}).get(key)
    try:
        return getattr(ctx, "crosssec_by_ncs", {}).get(int(key))
    except Exception:
        return None

def _sections_for_stations(ctx, stations_m, *, schedule=None, names=None, ncs_list=None):
    S = len(stations_m)
    if not S: return []
    # piecewise-constant schedule [{'station_m':..., 'name'/'ncs':...}, ...]
    if schedule:
        pairs = []
        for it in schedule:
            sm = it.get("station_m", it.get("station_mm", None))
            if sm is None: continue
            sm = float(sm) if "station_m" in it else float(sm)/1000.0
            key = it.get("name", it.get("ncs", None))
            sec = _resolve_section_key(ctx, key)
            if sec: pairs.append((sm, sec))
        pairs.sort(key=lambda x: x[0])
        out, j = [], 0
        for s in stations_m:
            while j+1 < len(pairs) and pairs[j+1][0] <= s: j += 1
            out.append(pairs[j][1] if pairs else None)
        return out

    # per-station names
    if names and len(names) == S:
        return [_resolve_section_key(ctx, nm) for nm in names]

    # constant fallback
    for seq in (names or []), (ncs_list or []):
        for key in (seq or []):
            sec = _resolve_section_key(ctx, key)
            if sec: return [sec]*S
    return [None]*S


@dataclass(kw_only=True)
class DeckObject(BaseObject):
    no: str = ""
    class_name: str = ""
    type: str = ""
    description: str = ""
    name: str = ""         # keeps your current override; fine
    inactive: str = ""

    cross_section_types: List[str] = field(default_factory=list)
    cross_section_names: List[str] = field(default_factory=list)
    grp_offset: List[float] = field(default_factory=list)
    placement_id: List[str] = field(default_factory=list)
    placement_description: List[str] = field(default_factory=list)
    ref_placement_id: List[str] = field(default_factory=list)
    ref_station_offset: List[float] = field(default_factory=list)
    station_value: List[float] = field(default_factory=list)
    cross_section_points_name: List[str] = field(default_factory=list)
    grp: List[str] = field(default_factory=list)
    cross_section_ncs: List[int] = field(default_factory=list)


    def get_object_metada(self):
        data = asdict(self)
        # Replace or remove verbose fields
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = f"<Axis object>" if self.axis_obj is not None else None


        # Remove 'colors' key from output
        data.pop('colors', None)  # 'None' avoids KeyError if it's missing
        data.pop('user_stations', None)  # 'None' avoids KeyError if it's missing
        data.pop('axis_obj', None)  # 'None' avoids KeyError if it's missing
        data.pop('axis_variables_obj', None)  # 'None' avoids KeyError if it's missing
        data.pop('axis_rotation', None)  # 'None' avoids KeyError if it's missing


        return data
    
    # Optional pretty metadata (preserves your existing style)
    def get_object_metadata(self) -> Dict:
        data = asdict(self)
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = "<Axis object>" if getattr(self, "axis_obj", None) is not None else None
        # remove UI-only noise (match your current trimming)
        data.pop('colors', None)
        data.pop('user_stations', None)
        data.pop('axis_obj', None)
        data.pop('axis_variables_obj', None)
        data.pop('axis_rotation', None)
        return data

    # ------------------------------------------------------------
    # New: Deck geometry (2D local -> 3D embedded)
    # ------------------------------------------------------------
    @staticmethod
    def _embed_with_shape_fallback(axis, stations_mm: np.ndarray, local_yz: np.ndarray, twist_deg: float):
        """
        Try the shapes Axis supports:
        - (S, N, 2)  stations-first
        - (N, S, 2)  points-first
        - (N, 2)     single-station
        Re-raises unexpected errors.
        """
        try:
            return axis.embed_section_points_world(
                stations_mm, yz_points_mm=local_yz, x_offsets_mm=None, rotation_deg=float(twist_deg)
            )
        except ValueError as e:
            if "yz_points_mm must be (M,2) or (N,M,2)" not in str(e):
                raise

        # Fallback #1: transpose to points-first
        if local_yz.ndim == 3 and local_yz.shape[0] == len(stations_mm):
            try:
                return axis.embed_section_points_world(
                    stations_mm, yz_points_mm=np.transpose(local_yz, (1, 0, 2)),
                    x_offsets_mm=None, rotation_deg=float(twist_deg)
                )
            except ValueError as e2:
                if "yz_points_mm must be (M,2) or (N,M,2)" not in str(e2):
                    raise

        # Fallback #2: single-station
        if local_yz.ndim == 3 and local_yz.shape[0] == 1:
            return axis.embed_section_points_world(
                stations_mm, yz_points_mm=local_yz[0],
                x_offsets_mm=None, rotation_deg=float(twist_deg)
            )

        raise ValueError(
            f"Incompatible yz_points_mm shape {local_yz.shape}. "
            "Tried (S,N,2), (N,S,2), and (N,2) for single station."
        )

    @staticmethod
    def _pick_section(ctx, names: List[str], ncs_list: List[int]) -> Optional[CrossSection]:
        # Prefer explicit names first
        if hasattr(ctx, "crosssec_by_name"):
            for nm in names or []:
                cs = ctx.crosssec_by_name.get(nm)
                if cs is not None:
                    return cs
        # Then by NCS
        if hasattr(ctx, "crosssec_by_ncs"):
            for ncs in ncs_list or []:
                try:
                    cs = ctx.crosssec_by_ncs.get(int(ncs))
                except Exception:
                    cs = None
                if cs is not None:
                    return cs
        return None


    def compute_geometry(
        self,
        *,
        ctx,
        stations_m: Optional[List[float]] = None,
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> Dict[str, object]:
        axis: Optional[Axis] = getattr(self, "axis_obj", None)
        EMPTY = {
            "ids": [], "stations_mm": np.array([], float),
            "points_mm": np.zeros((0, 0, 3), float),
            "local_Y_mm": np.zeros((0, 0), float),
            "local_Z_mm": np.zeros((0, 0), float),
            "loops_idx": [],
        }
        if axis is None:
            return EMPTY

        # ---- Stations (meters) -------------------------------------------------
        if stations_m is None:
            S = np.asarray(getattr(axis, "stations", []), dtype=float)  # mm
            stations_m = (S / 1000.0).tolist()
        stations_m = [float(s) for s in (stations_m or [])]
        if not stations_m:
            return EMPTY
        stations_mm = np.asarray(stations_m, float) * 1000.0

        # ---- Axis variables per station ---------------------------------------
        axis_results = AxisVariable.evaluate_at_stations_cached(
            getattr(self, "axis_variables_obj", []) or [], stations_m
        )

        # ---- Resolve a single section for this object -------------------------
        section = self._pick_section(ctx, self.cross_section_names or self.cross_section_types, self.cross_section_ncs)
        if section is None:
            return EMPTY

        # ---- Fast path: let CrossSection embed (may succeed outright) ---------
        try:
            ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = section.compute_embedded_points(
                axis=axis,
                axis_var_results=axis_results,
                stations_m=stations_m,
                twist_deg=float(twist_deg or 0.0),
                negate_x=negate_x,
            )
            return {
                "ids": ids, "stations_mm": S_mm, "points_mm": P_mm,
                "local_Y_mm": X_mm, "local_Z_mm": Y_mm, "loops_idx": loops_idx,
            }
        except ValueError as e:
            # If the error is NOT the shape complaint, surface it
            if "yz_points_mm must be (M,2) or (N,M,2)" not in str(e):
                raise

        # ---- Fallback: embed locally with shape-robustness --------------------
        # Evaluate local YZ for all (kept) stations
        ids, X_mm, Y_mm, loops_idx = section.compute_local_points(
            axis_var_results=axis_results, negate_x=negate_x
        )
        # convention: local Y == X_mm, local Z == Y_mm
        local_yz = np.dstack([X_mm, Y_mm])  # (S, N, 2) by construction

        # Let Axis embed with fallback shapes
        P_mm = self._embed_with_shape_fallback(axis, stations_mm, local_yz, twist_deg=float(twist_deg or 0.0))

        return {
            "ids": ids,
            "stations_mm": stations_mm,
            "points_mm": P_mm,
            "local_Y_mm": X_mm,
            "local_Z_mm": Y_mm,
            "loops_idx": loops_idx,
        }
