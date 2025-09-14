from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from models.mapping import mapping
from models.axis import Axis
from models.base import BaseObject, _build_axis_index


@dataclass(kw_only=True)
class TwoAxisBase(BaseObject):
    # keep BaseObject's primary axis_name for "beg" by convention
    beg_axis_name: str = ""
    end_axis_name: str = ""
    beg_axis_obj: Optional[Axis] = None
    end_axis_obj: Optional[Axis] = None

    def set_axes(self, axis_data: List[Dict]):
        """Resolve beg_axis_obj/end_axis_obj using the same fast index used by BaseObject.set_axis()."""
        if axis_data is None:
            return
        axis_map = mapping.get(Axis, {})
        idx = axis_map.get("_axis_index")
        if idx is None:
            idx = _build_axis_index(axis_data, axis_map)
            axis_map["_axis_index"] = idx
            mapping[Axis] = axis_map

        def resolve(name: str) -> Optional[Axis]:
            if not name:
                return None
            d = idx.get(name) or idx.get(str(name)) or idx.get(str(name).strip()) or idx.get(str(name).strip().lower())
            if not d:
                return None
            stations = [float(s) for s in d.get(axis_map.get('stations', 'StaionValue'), [])]
            x_coords = [float(x) for x in d.get(axis_map.get('x_coords', 'CurvCoorX'), [])]
            y_coords = [float(y) for y in d.get(axis_map.get('y_coords', 'CurvCoorY'), [])]
            z_coords = [float(z) for z in d.get(axis_map.get('z_coords', 'CurvCoorZ'), [])]
            return Axis(stations=stations, x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)

        self.beg_axis_obj = resolve(self.beg_axis_name)
        self.end_axis_obj = resolve(self.end_axis_name)

@dataclass(kw_only=True)
class SecondaryObject(TwoAxisBase):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str

    beg_placement_id: str
    beg_placement_description: str
    beg_ref_placement_id: str
    beg_ref_station_offset: float
    beg_station_value: float
    beg_cross_section_points_name: str
    beg_ncs: int

    end_placement_id: str
    end_placement_description: str
    end_ref_placement_id: str
    end_ref_station_offset: float
    end_station_value: float
    end_cross_section_points_name: str
    end_ncs: int

    grp_offset: float
    grp: str