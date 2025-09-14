import json
import math
from typing import Type, Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass, field, fields, is_dataclass,asdict
from Axis import Axis, plot_axis_and_points_transformed, save_plot
from AxisVariables import AxisVariable
import os
import sys

import logging
from abc import ABC
from functools import lru_cache
from dataclasses import dataclass, field
from typing import ClassVar
from utils import _compile_expr, _sanitize_vars, _SCALAR_FUNCS, _VECTOR_FUNCS

logger = logging.getLogger(__name__)
# Turn on detailed traces with: logging.basicConfig(level=logging.DEBUG)

def _tiny_hash(arr) -> str:
    import hashlib
    if arr is None:
        arr = []
    data = (','.join(str(float(v)) for v in arr)).encode('utf-8')
    return hashlib.blake2b(data, digest_size=12).hexdigest()

# ---- Fast JSON lib (optional) ----
try:
    import orjson as _fastjson_AC
except Exception:
    _fastjson_AC = None

# ------- Axis object cache (dedupe identical axes across objects) -------
_AXIS_CACHE: dict = {}

def get_axis_cached(axis_name, stations, x_coords, y_coords, z_coords):
    """
    Return a cached Axis with identical data (by value), or create & cache it.
    axis_name participates in the key to avoid accidental collisions across
    differently named axes that happen to share identical arrays.
    """
    # Normalize + hashable key
    name_key = (axis_name or "").strip().lower()
    key = (
        name_key,
        tuple(float(s) for s in (stations or [])),
        tuple(float(v) for v in (x_coords or [])),
        tuple(float(v) for v in (y_coords or [])),
        tuple(float(v) for v in (z_coords or [])),
    )
    ax = _AXIS_CACHE.get(key)
    if ax is None:
        ax = Axis(stations=stations, x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)
        # keep cache bounded
        if len(_AXIS_CACHE) > 128:
            _AXIS_CACHE.clear()
        _AXIS_CACHE[key] = ax
    return ax

def build_axis_index(axis_data: List[Dict]) -> Dict[str, Dict]:
        axis_map = mapping.get(Axis, {})
        class_key = axis_map.get('class', 'Class')
        name_key  = axis_map.get('name',  'Name')
        idx = {}
        for d in axis_data or []:
            if d.get(class_key) == 'Axis':
                name = str(d.get(name_key, '')).strip().lower()
                if name:
                    idx[name] = d
        return idx

def _build_axis_index(axis_data: list[dict], axis_map: dict) -> dict[str, dict]:
    """Return {axis_name -> axis_dict} using mapping's keys (Name/Class)."""
    class_key = axis_map.get("Class", "Class")
    name_key  = axis_map.get("Name",  "Name")
    out = {}
    for d in axis_data or []:
        if d.get(class_key) == "Axis":
            name = str(d.get(name_key, "")).strip()
            if name:
                out[name] = d
    return out


def build_cross_section_index(cross_sections: List["CrossSection"]) -> Dict[int, "CrossSection"]:
        out = {}
        for cs in cross_sections or []:
            try:
                out[int(cs.ncs)] = cs
            except Exception:
                pass
        return out    


   
VERBOSE = False  # Set to True for detailed debug output


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
from copy import deepcopy

# Assumes Axis, AxisVariable, CrossSection, mapping, create_input_for_visualisation already exist.

@dataclass
class VisoContext:
    axes_by_name: Dict[str, "Axis"] = field(default_factory=dict)
    crosssec_by_ncs: Dict[int, "CrossSection"] = field(default_factory=dict)
    mainstations_by_name: Dict[str, "MainStation"] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls,
        axis_data: List[dict],
        cross_sections: List["CrossSection"],
        mainstations: Optional[List["MainStation"]] = None,
        *,
        mapping_cfg: Dict[type, Dict[str, str]] = None
    ) -> "VisoContext":
        mapping_cfg = mapping_cfg or mapping
        amap = mapping_cfg.get(Axis, {})
        name_k = amap.get("name", "Name")
        sta_k  = amap.get("stations", "StaionValue")
        x_k    = amap.get("x_coords","CurvCoorX")
        y_k    = amap.get("y_coords","CurvCoorY")
        z_k    = amap.get("z_coords","CurvCoorZ")
        cls_k  = amap.get("class", "Class")

        axes = {}
        for row in (axis_data or []):
            if str(row.get(cls_k, "Axis")) == "Axis":
                nm = str(row.get(name_k, "")).strip()
                if nm:
                    axes[nm] = Axis(
                        stations=[float(s) for s in row.get(sta_k, [])],
                        x_coords=[float(v) for v in row.get(x_k, [])],
                        y_coords=[float(v) for v in row.get(y_k, [])],
                        z_coords=[float(v) for v in row.get(z_k, [])],
                    )

        cs_by_ncs = {int(cs.ncs): cs for cs in (cross_sections or [])}
        ms_by_name = {getattr(ms, "name", "").strip(): ms for ms in (mainstations or []) if getattr(ms, "name", "").strip()}
        return cls(axes_by_name=axes, crosssec_by_ncs=cs_by_ncs, mainstations_by_name=ms_by_name)


from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, ClassVar

@dataclass(slots=True)
class BaseObject:
    """
    Shared behavior & state for DeckObject / PierObject / FoundationObject.
    """
    # core identity
    axis_name: str = ""

    # persisted state
    axis_variables: List[Dict] = field(default_factory=list)            # raw JSON rows
    axis_variables_obj: List[AxisVariable] = field(default_factory=list)
    axis_obj: Optional[Axis] = None

    user_stations: List[float] = field(default_factory=list)
    colors: Dict[str, str] = field(default_factory=lambda: {
        "axis": "black",
        "first_station_points": "blue",
        "cross_section_loops": "red",
        "longitudinal_lines": "gray",
    })
    axis_rotation: float = 0.0

    # per-instance memo cache
    _memo_cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # class-wide bounded transform cache
    _TRANSFORM_CACHE: ClassVar[Dict] = {}
    _TRANSFORM_CACHE_MAX: ClassVar[int] = 256

    # ---- tiny utilities ----
    def memo(self, key, builder):
        m = self._memo_cache
        if key not in m:
            m[key] = builder()
        return m[key]

    def log(self, *a, **k):
        flag = getattr(self, 'debug', None)
        if flag is None:
            flag = VERBOSE
        if flag:
            print(*a, **k)

    # ---------- convenience ----------
    def get_object_metada(self) -> Dict[str, Any]:
        data = asdict(self)
        # tidy noisy fields
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = "<Axis object>" if self.axis_obj is not None else None
        # drop runtime-only
        for k in ('colors', 'user_stations', 'axis_obj', 'axis_variables_obj', 'axis_rotation', '_memo_cache'):
            data.pop(k, None)
        return data

    # ---------- axis variables ----------
    def set_axis_variables(self, _json_ignored: List[Dict], axis_var_map: Dict[str, str]):
        try:
            # always use self.axis_variables (already stored by from_dict)
            self.axis_variables_obj = create_axis_variables(self.axis_variables, axis_var_map)
        except Exception as e:
            logger.warning("Axis variables parse failed for %s: %s", getattr(self, 'name', '<unnamed>'), e)
            self.axis_variables_obj = []

    # ---------- axis lookup & creation ----------
    def _axis_dict_from(self, axis_data: Optional[List[Dict]] = None,
                        axis_index: Optional[Dict[str, Dict]] = None) -> Optional[Dict]:
        if not self.axis_name:
            return None
        key = str(self.axis_name).strip().lower()
        if axis_index:
            return axis_index.get(key)
        axis_map = mapping.get(Axis, {})
        class_key = axis_map.get('class', 'Class')
        name_key  = axis_map.get('name',  'Name')
        for d in axis_data or []:
            if d.get(class_key) == 'Axis' and str(d.get(name_key, '')).strip().lower() == key:
                return d
        return None

    def set_axis(self, axis_name: Optional[str], axis_data: Optional[List[Dict]] = None,
                 axis_index: Optional[Dict[str, Dict]] = None):
        if axis_name:
            self.axis_name = axis_name

        axis_dict = self._axis_dict_from(axis_data, axis_index)
        if not axis_dict:
            logger.warning("No Axis data found for axis_name=%r", self.axis_name)
            self.axis_obj = None
            return

        axis_map = mapping.get(Axis, {})
        stations_key = axis_map.get('stations',  'StaionValue')
        x_coords_key = axis_map.get('x_coords', 'CurvCoorX')
        y_coords_key = axis_map.get('y_coords', 'CurvCoorY')
        z_coords_key = axis_map.get('z_coords', 'CurvCoorZ')

        try:
            stations = [float(s) for s in axis_dict.get(stations_key, [])]
            x_coords = [float(x) for x in axis_dict.get(x_coords_key, [])]
            y_coords = [float(y) for y in axis_dict.get(y_coords_key, [])]
            z_coords = [float(z) for z in axis_dict.get(z_coords_key, [])]
            self.axis_obj = get_axis_cached(self.axis_name, stations, x_coords, y_coords, z_coords)
        except Exception as e:
            logger.warning("Failed to create Axis object for %s: %s", self.axis_name, e)
            self.axis_obj = None

    # ---------- transform caching ----------
    def _axis_signature(self) -> Tuple[str, str, str, str, str]:
        if not self.axis_obj:
            return ("", "", "", "", "m")
        return (
            _tiny_hash(self.axis_obj.stations),
            _tiny_hash(self.axis_obj.x_coords),
            _tiny_hash(self.axis_obj.y_coords),
            _tiny_hash(self.axis_obj.z_coords),
            "mm",
        )

    @staticmethod
    def _points_signature(nested_points: List[List[Dict[str, float]]]) -> str:
        flat = []
        for ring in (nested_points or []):
            for p in (ring or []):
                flat.append(f"{float(p.get('x',0)):.6f}:{float(p.get('y',0)):.6f}")
            flat.append("|")
        import hashlib
        return hashlib.blake2b(",".join(flat).encode("utf-8"), digest_size=16).hexdigest()

    def transform_points_cached(
        self,
        local_points: List[List[Dict[str, float]]],
        stations_to_plot: List[float],
        rotation_angle: Optional[float] = None,
    ):
        if self.axis_obj is None:
            raise RuntimeError("axis_obj is not set. Call set_axis(...) first.")
        rot = self.axis_rotation if rotation_angle is None else float(rotation_angle)
        key = (
            self._axis_signature(),
            tuple(round(float(s), 8) for s in (stations_to_plot or [])),
            round(rot, 6),
            self._points_signature(local_points),
        )
        hit = BaseObject._TRANSFORM_CACHE.get(key)
        if hit is not None:
            return hit

        result = self.axis_obj.transform_points(local_points, stations_to_plot, rotation_angle=rot)
        if len(BaseObject._TRANSFORM_CACHE) >= BaseObject._TRANSFORM_CACHE_MAX:
            BaseObject._TRANSFORM_CACHE.clear()
        BaseObject._TRANSFORM_CACHE[key] = result
        return result

    # ---------- visualization input ----------
    def get_input_for_visualisation(
        self,
        *,
        axis_data=None,
        cross_section_objects=None,
        json_file_path=None,
        axis_rotation=None,
        colors=None
    ):
        return create_input_for_visualisation(
            obj=self,
            axis_data=axis_data,
            cross_section_objects=cross_section_objects,
            json_file_path=json_file_path,
            axis_rotation=axis_rotation,
            colors=colors,
        )
    
    def with_axis(self, new_axis: "Axis"):
        c = deepcopy(self)
        c.axis_obj = new_axis
        return c

    def with_mainstations(self, new_ms: "MainStation"):
        c = deepcopy(self)
        c._mainstations = new_ms
        return c

    def _resolve_cross_sections_from_ncs(self, ctx: VisoContext) -> List["CrossSection"]:
        ncs_list = getattr(self, "cross_section_ncs", []) or []
        cs_objs = []
        for n in ncs_list:
            cs = ctx.crosssec_by_ncs.get(int(n))
            if cs: cs_objs.append(cs)
        return cs_objs
    
def build_viso_object(
    obj: BaseObject, 
    ctx: VisoContext,
    *,
    axis: Optional["Axis"] = None,
    mainstations: Optional["MainStation"] = None,
    cross_sections_override: Optional[List["CrossSection"]] = None,
    mapping_cfg: Dict[type, Dict[str, str]] = None
) -> dict:
    """
    Resolve dependencies for `obj` and return the classic vis dict
    through your existing create_input_for_visualisation path.
    No writes, no JSON mutation.
    """
    mapping_cfg = mapping_cfg or mapping

    # Axis: explicit override > ctx by name > existing self.axis_obj
    ax = axis or getattr(obj, "axis_obj", None)
    if ax is None:
        ax_name = getattr(obj, "axis_name", None)
        if ax_name:
            ax = ctx.axes_by_name.get(str(ax_name).strip())
    obj.axis_obj = ax  # safe to set; this is the in-memory dataclass

    # MainStations if you model them later (optional for now)
    if mainstations is not None:
        obj._mainstations = mainstations

    # CrossSections (objects, not filenames)
    if cross_sections_override is not None:
        obj._cross_sections = list(cross_sections_override)
    else:
        obj._cross_sections = getattr(obj, "_cross_sections", None) or obj._resolve_cross_sections_from_ncs(ctx)

    # Axis variables: keep your current raw list and let your existing path build objects
    if hasattr(obj, "axis_variables") and isinstance(obj.axis_variables, list):
        obj.set_axis_variables(obj.axis_variables, mapping_cfg.get(AxisVariable, {}))


    vis_row = obj.get_input_for_visualisation(
        axis_data=None,                      # or whatever you called your raw axis dicts
        cross_section_objects=obj._cross_sections,   # your list of CrossSection objects
        json_file_path=None,
        axis_rotation=getattr(obj, "axis_rotation", 0),
        colors=getattr(obj, "colors", None)
    )

    # # This helper exists in your tree and already supports passing cross_section_objects.
    # vis_row = obj.get_input_for_visualisation(
    #     cross_section_objects=obj._cross_sections,
    #     axis_data=None,                   # not needed if axis_obj is set
    #     json_file_path=None,              # keep file path resolution as-is
    #     axis_rotation=getattr(obj, "axis_rotation", None),
    #     colors=getattr(obj, "colors", None),
    # )

    return vis_row


@dataclass
class CrossSection:
    # ---- existing mapped fields (keep names to avoid breaking from_dict/mapping) ----
    no: Optional[str] = None
    class_name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    name: Optional[str] = None
    inactive: Optional[str] = None
    ncs: Optional[int] = None
    material1: Optional[int] = None
    material2: Optional[int] = None
    material_reinf: Optional[int] = None
    json_name: Union[str, List[str], None] = None
    sofi_code: Union[str, List[str], None] = None
    points: Union[List[Dict[str, Any]], Dict[str, Any], None] = None
    variables: Union[List[Dict[str, Any]], Dict[str, Any], None] = None


    class ReferenceFrame:
        def __init__(self, reference_type, reference=None, points=None, variables=None):
            self.reference_type = reference_type
            self.reference = reference or []
            self.points = points or []
            self.variables = variables or {}

        def eval_equation(self, string_equation):
            try:
                return float(string_equation)
            except (TypeError, ValueError):
                pass
            code = _compile_expr(string_equation)
            env = {**_SCALAR_FUNCS, **_sanitize_vars(self.variables)}
            try:
                val = eval(code, {"__builtins__": {}}, env)
                return float(val)
            except Exception as e:
                print(f"Error evaluating '{string_equation}': {e}")
                return 0.0

        def get_coordinates(self, coords):
            rt = (self.reference_type or '').lower()
            if rt in ("c","carthesian","e","euclidean"): return self._euclid(coords)
            if rt in ("p","polar"):                     return self._polar(coords)
            if rt in ("constructionaly","cy"):          return self._cy()
            if rt in ("constructionalz","cz"):          return self._cz()
            return self._euclid(coords)

        def _euclid(self, coords):
            x = self.eval_equation(coords[0]); y = self.eval_equation(coords[1])
            final_x, final_y = x, y
            if not self.reference:
                maxlen = max(abs(final_x), abs(final_y)) * 1.5
                return {'coords': {'x': final_x, 'y': final_y},
                        'guides': {'isPlane': True,'origin': {'x': 0, 'y': 0},
                                'dirX': {'x': maxlen, 'y': 0}, 'dirY': {'x': 0, 'y': maxlen}}}
            if len(self.reference) == 1:
                p = next((k for k in self.points if k['id']==self.reference[0]), None)
                if not p: return {'coords': {'x': final_x, 'y': final_y}, 'guides': None}
                final_x += p['x']; final_y += p['y']
                maxlen = max(abs(final_x), abs(final_y)) * 1.5
                return {'coords': {'x': final_x, 'y': final_y},
                        'guides': {'isPlane': True,'origin': {'x': p['x'], 'y': p['y']},
                                'dirX': {'x': maxlen, 'y': 0}, 'dirY': {'x': 0, 'y': maxlen}}}
            if len(self.reference) == 2:
                p1 = next((k for k in self.points if k['id']==self.reference[0]), None)
                p2 = next((k for k in self.points if k['id']==self.reference[1]), None)
                if not p1 or not p2: return {'coords': {'x': final_x, 'y': final_y}, 'guides': None}
                final_x += p1['x']; final_y += p2['y']
                maxlen = max(abs(p2['x']), abs(p2['y']))
                return {'coords': {'x': final_x, 'y': final_y},
                        'guides': {'isPlane': True,'origin': {'x': p1['x'], 'y': p2['y']},
                                'dirX': {'x': maxlen, 'y': 0}, 'dirY': {'x': 0, 'y': maxlen}}}
            return {'coords': {'x': final_x, 'y': final_y}, 'guides': None}

        def _polar(self, coords):
            if len(self.reference) < 2:
                return {'coords': {'x': self.eval_equation(coords[0]), 'y': self.eval_equation(coords[1])}, 'guides': None}
            p1 = next((k for k in self.points if k['id']==self.reference[0]), None)
            p2 = next((k for k in self.points if k['id']==self.reference[1]), None)
            if (not p1) or (not p2):
                return {'coords': {'x': self.eval_equation(coords[0]), 'y': self.eval_equation(coords[1])}, 'guides': None}
            x1,y1 = p1['x'],p1['y']; x2,y2 = p2['x'],p2['y']
            r_x = self.eval_equation(coords[0]); r_y = self.eval_equation(coords[1])
            dir_x, dir_y = x2-x1, y2-y1
            L = math.hypot(dir_x, dir_y) or 1.0
            ux, uy = dir_x/L, dir_y/L
            vx, vy = -uy, ux
            final_x = x1 + r_x*ux + r_y*vx
            final_y = y1 + r_x*uy + r_y*vy
            return {'coords': {'x': final_x, 'y': final_y},
                    'guides': {'isPlane': True, 'origin': {'x': x1, 'y': y1},
                            'dirX': {'x': dir_x, 'y': dir_y},
                            'dirY': {'x': -dir_y, 'y': dir_x}}}

        def _cy(self):
            if len(self.reference) != 3: raise ValueError('CY requires three reference points')
            p1 = next((k for k in self.points if k['id']==self.reference[0]), None)
            p2 = next((k for k in self.points if k['id']==self.reference[1]), None)
            p3 = next((k for k in self.points if k['id']==self.reference[2]), None)
            if (not p1) or (not p2) or (not p3): return {'coords': {'x': 0, 'y': 0}, 'guides': None}
            dx = p2['x']-p1['x']; dy=p2['y']-p1['y']
            m = dy/dx if dx!=0 else 0.0
            c = p1['y'] - m*p1['x']
            y = m*p3['x'] + c
            return {'coords': {'x': p3['x'], 'y': y}, 'guides': {'isPlane': False, 'p1': p1, 'p2': p2, 'p3': p3}}

        def _cz(self):
            if len(self.reference) != 3: raise ValueError('CZ requires three reference points')
            p1 = next((k for k in self.points if k['id']==self.reference[0]), None)
            p2 = next((k for k in self.points if k['id']==self.reference[1]), None)
            p3 = next((k for k in self.points if k['id']==self.reference[2]), None)
            if (not p1) or (not p2) or (not p3): return {'coords': {'x': 0, 'y': 0}, 'guides': None}
            dx = p2['x']-p1['x']; dy=p2['y']-p1['y']
            m = dy/dx if dx!=0 else 0.0
            c = p1['y'] - m*p1['x']
            x = (p3['y'] - c)/m if m!=0 else p3['x']
            return {'coords': {'x': x, 'y': p3['y']}, 'guides': {'isPlane': False, 'p1': p1, 'p2': p2, 'p3': p3}}

  

    

def create_axis_variables(json_data: List[Dict], mapping: Dict[str, str]) -> List[AxisVariable]:
    """
    Transform JSON data for AxisVariable using the provided mapping and create AxisVariable objects.
    """
    transformed_data = [
        {mapping.get(k, k): v for k, v in var.items()}
        for var in json_data
    ]
    print(f"Transformed axis_variables: {transformed_data}")  # Debug
    return AxisVariable.create_axis_variables(transformed_data)


def from_dict(cls: Type, data: Dict, mapping_config: Dict[Type, Dict[str, str]], axis_data: List[Dict] = None):
    """
    Convert a dictionary into a dataclass object of type `cls`,
    applying mapping_config for key name translations.
    """
    key_map = mapping_config.get(cls, {})

    # If axis_data is provided, prebuild an index once and stash it in mapping
    if axis_data:
        axis_map = mapping_config.get(Axis, {})
        if axis_map is not None and "_axis_index" not in axis_map:
            axis_map["_axis_index"] = _build_axis_index(axis_data, axis_map)
            mapping_config[Axis] = axis_map

    field_info = {f.name: f.type for f in fields(cls)}
    init_kwargs = {}

    # Debug: Verify Axis type
    print(f"Axis type in from_dict: {Axis}, field_info['axis_obj'] == Axis: {field_info.get('axis_obj') == Axis}")  # Debug

    # Invert key_map to find JSON key for 'axis_name'
    reverse_key_map = {v: k for k, v in key_map.items()}
    axis_name_key = reverse_key_map.get('axis_name', 'axis_name')
    print(f"Using axis_name_key: {axis_name_key}")  # Debug

    for json_key, value in data.items():
        field_name = key_map.get(json_key, json_key)
        if field_name in field_info:
            if field_name == 'axis_variables' and isinstance(value, list) and field_info[field_name] == List[Dict]:
                print(f"Processing axis_variables: {value}")  # Debug
                init_kwargs[field_name] = value  # Store raw data
            elif field_name == 'axis_obj' and field_info[field_name] == Axis and axis_data is not None:
                init_kwargs[field_name] = None  # Processed later
                init_kwargs['axis_name'] = data.get(axis_name_key)  # Use mapped key for axis_name
            elif is_dataclass(field_info[field_name]) and isinstance(value, dict):
                init_kwargs[field_name] = from_dict(field_info[field_name], value, mapping_config, axis_data)
            else:
                init_kwargs[field_name] = value
        else:
            print(f"Skipping unmapped field: {field_name}")  # Debug

    if cls.__name__ == "PierObject":
        init_kwargs.setdefault("top_cross_section_points_name", "")
        init_kwargs.setdefault("bot_cross_section_points_name", "")
        init_kwargs.setdefault("internal_station_value", [])

    print(f"Creating {cls.__name__} with kwargs: {init_kwargs}")  # Debug
    instance = cls(**init_kwargs)  # Create instance with dataclass fields only
    return instance

def load_from_json(
    cls: Type,
    data_or_file: Union[str, List[Dict], Dict],
    mapping_config: Dict[Type, Dict[str, str]] = None,
    axis_data: List[Dict] = None
) -> Tuple[Union[List, object], List[Dict]]:
    """
    Load objects of type `cls` from a JSON file, list of dicts, or a single dict.
    Filters out entries where Class is "ClassInfo" and returns both objects and filtered data.
    """
    mapping_config = mapping_config or mapping  # Use global mapping if none provided

    if isinstance(data_or_file, str):
        if _fastjson_AC:
            with open(data_or_file, "rb") as f:
                data = _fastjson_AC.loads(f.read())
        else:
            with open(data_or_file, "r", encoding="utf-8") as f:
                data = json.load(f)
    else:
        data = data_or_file

    print(f"axis_data provided: {axis_data is not None}")  # Debug

    if isinstance(data, list):
        # Filter out entries where Class is "ClassInfo"
        filtered_data = [d for d in data if d.get("Class") != "ClassInfo"]
        objects = [from_dict(cls, obj, mapping_config, axis_data) for obj in filtered_data]
        return objects, filtered_data
    elif isinstance(data, dict):
        if data.get("Class") == "ClassInfo":
            return None, []
        obj = from_dict(cls, data, mapping_config, axis_data)
        return obj, [data]
    else:
        raise ValueError("data_or_file must be a JSON file path, a list of dicts, or a dict.")

from functools import lru_cache
@lru_cache(maxsize=512)
def find_relative_file(rel_path: str, verbose: bool = False) -> str | None:
    """
    Fast path: direct joins against likely bases (script/exe/MEIPASS) + filename match.
    Fallback: your original multi-level search (kept intact).
    Results are cached so repeated lookups are near O(1).
    """
    if not rel_path:
        return None

    import os, sys
    rel_path = os.path.normpath(rel_path)
    file_name = os.path.basename(rel_path)

    # 1) Quick bases
    bases = []
    if getattr(sys, "frozen", False):
        bases.extend([getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)])
    else:
        bases.append(os.path.abspath(os.path.dirname(__file__)))
    bases = [b for b in bases if b]

    # try direct join (fast)
    for base in bases:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return os.path.abspath(p)

    # also try “found by filename” inside base (1–2 levels down)
    for base in bases:
        for dirpath, _, filenames in os.walk(base):
            # shallow to keep fast
            depth = len(os.path.relpath(dirpath, base).split(os.sep))
            if depth > 2:
                continue
            if file_name.lower() in (f.lower() for f in filenames):
                candidate = os.path.join(dirpath, file_name)
                # make sure the tail matches the relative path shape
                if os.path.normpath(candidate).lower().endswith(os.path.normpath(rel_path).lower()):
                    return os.path.abspath(candidate)

    # 2) Fallback to your full (slower but thorough) search — unchanged logic
    # --- begin fallback (your previous implementation) ---
    # (Copy of your original deep search trimmed into a helper)
    def _deep_search():
        base_dirs = []
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            base_dirs += [getattr(sys, "_MEIPASS", ""), exe_dir,
                          os.path.dirname(exe_dir),
                          os.path.dirname(os.path.dirname(exe_dir)),
                          os.path.dirname(os.path.dirname(os.path.dirname(exe_dir))),
                          os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(exe_dir))))]
        else:
            script_dir = os.path.abspath(os.path.dirname(__file__))
            base_dirs += [script_dir,
                          os.path.dirname(script_dir),
                          os.path.dirname(os.path.dirname(script_dir)),
                          os.path.dirname(os.path.dirname(os.path.dirname(script_dir))),
                          os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))]
        base_dirs = list(dict.fromkeys([b for b in base_dirs if b]))

        def check_file_path(full_path: str) -> str | None:
            if not os.path.exists(full_path):
                return None
            return os.path.abspath(full_path)

        # Search down first (up to 4 levels), then up (up to 4 levels)
        for base_dir in base_dirs:
            current_dir = base_dir
            for level in range(5):
                if level == 0:
                    full_path = os.path.join(current_dir, rel_path)
                    res = check_file_path(full_path)
                    if res: return res
                else:
                    try:
                        subdirs = [d for d in os.listdir(current_dir)
                                   if os.path.isdir(os.path.join(current_dir, d))]
                    except OSError:
                        break
                    for subdir in subdirs:
                        full_path = os.path.join(current_dir, subdir, rel_path)
                        res = check_file_path(full_path)
                        if res: return res
                    if not subdirs: break
                    current_dir = os.path.join(current_dir, subdirs[0])

        for base_dir in base_dirs:
            current_dir = base_dir
            for _ in range(4):
                parent = os.path.dirname(current_dir)
                if parent == current_dir: break
                full_path = os.path.join(parent, rel_path)
                res = check_file_path(full_path)
                if res: return res
                current_dir = parent

        return None
    # --- end fallback ---

    res = _deep_search()
    if not res and verbose:
        print(f"[find_relative_file] Not found: {rel_path}")
    return res


def create_input_for_visualisation(
    obj: object,
    *,
    axis_data: list | None = None,
    cross_section_objects: list | None = None,
    json_file_path: str | None = None,
    axis_rotation: float | None = None,
    colors: dict | None = None,
) -> dict:
    """
    Create input for visualisation with a reproducible axis signature.
    """
    import hashlib
    import numpy as np  # <— add this

    # Colors & rotation defaults
    colors = colors or getattr(obj, 'colors', {
        "axis": "black",
        "first_station_points": "blue",
        "cross_section_loops": "red",
        "longitudinal_lines": "gray"
    })
    axis_rotation = axis_rotation if axis_rotation is not None else getattr(obj, 'axis_rotation', 0)

    # inside create_input_for_visualisation
    def _to_list(x):
        # robust against numpy arrays, scalars, None, strings
        try:
            import numpy as np
            if isinstance(x, np.ndarray): 
                return [float(v) for v in x.tolist()]
        except Exception:
            pass
        if x is None or x == "":
            return []
        if isinstance(x, (list, tuple, set)):
            return [float(v) for v in x]
        if isinstance(x, (int, float)):
            return [float(x)]
        # last ditch: ignore
        return []

    # combine possible station sources
    station_value_main   = _to_list(getattr(obj, 'station_value', None))
    station_value_inner  = _to_list(getattr(obj, 'internal_station_value', None))
    user_stations        = _to_list(getattr(obj, 'user_stations', None))

    base_breaks = sorted(set(station_value_main + station_value_inner + user_stations))

    # Interpolate segments between "breakpoints" to get dense stations_to_plot
    def interpolate_segments(breakpoints: List[float]) -> List[float]:
        if len(breakpoints) < 2:
            return breakpoints
        out = []
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            dist = b - a
            if dist <= 0:
                continue
            if   dist <   5: steps = 2
            elif dist <  10: steps = 3
            elif dist <  50: steps = 10
            elif dist < 100: steps = 20
            elif dist < 200: steps = 40
            elif dist < 300: steps = 30
            elif dist < 400: steps = 40
            elif dist < 500: steps = 50
            else:             steps = 500
            step = dist / steps
            out.extend(round(a + j * step, 8) for j in range(steps))
        out.append(round(breakpoints[-1], 8))
        return out
    stations_to_plot = interpolate_segments(base_breaks)

    # Map Axis keys from mapping config
    axis_map = mapping.get(Axis, {})
    class_key     = axis_map.get('class', 'Class')
    name_key      = axis_map.get('name', 'Name')
    stations_key  = axis_map.get('stations', 'StaionValue')
    x_coords_key  = axis_map.get('x_coords', 'CurvCoorX')
    y_coords_key  = axis_map.get('y_coords', 'CurvCoorY')
    z_coords_key  = axis_map.get('z_coords', 'CurvCoorZ')

    ...
    # Prefer an already-resolved Axis object; otherwise fall back to raw axis_data
    def _as_float_list(val):
        """Robustly turn None / scalar / list / tuple / np.ndarray into List[float]."""
        if val is None:
            return []
        if isinstance(val, np.ndarray):
            val = val.ravel().tolist()
        elif isinstance(val, (list, tuple)):
            val = list(val)
        else:
            val = [val]
        out = []
        for v in val:
            try:
                out.append(float(v))
            except Exception:
                # keep going; skip non-numeric junk rather than crashing
                pass
        return out

    stations = x_coords = y_coords = z_coords = []
    if getattr(obj, "axis_obj", None) is not None:
        ax = obj.axis_obj
        stations = _as_float_list(getattr(ax, "stations", None))
        x_coords = _as_float_list(getattr(ax, "x_coords", None))
        y_coords = _as_float_list(getattr(ax, "y_coords", None))
        z_coords = _as_float_list(getattr(ax, "z_coords", None))
    else:
        # Find axis dict from raw data (axis_data may be None)
        axis_dict = None
        axis_name = getattr(obj, 'axis_name', '') or ''
        for data in (axis_data or []):  # safe when axis_data is None
            if (data.get(class_key) == 'Axis' and str(data.get(name_key, '')).lower() == axis_name.lower()):
                axis_dict = data
                break
        if axis_dict:
            stations = _as_float_list(axis_dict.get(stations_key))
            x_coords = _as_float_list(axis_dict.get(x_coords_key))
            y_coords = _as_float_list(axis_dict.get(y_coords_key))
            z_coords = _as_float_list(axis_dict.get(z_coords_key))

    # AxisVariables to expected shape (keep yours)
    axis_variables = [
        {
            "VariableName": var.name,
            "VariableStations": var.xs,
            "VariableValues": [str(v * 1000) for v in var.ys],  # as in your original
            "VariableIntTypes": var.types,
            "VariableDescription": "empty"
        }
        for var in getattr(obj, 'axis_variables_obj', [])
    ]

    # # Resolve cross-section JSON
    # ncs_list = getattr(obj, 'cross_section_ncs', None)
    # if not isinstance(ncs_list, list) or not ncs_list:
    #     ncs_list = getattr(obj, 'internal_cross_section_ncs', None)
    # final_json_file = "no MASTER SECTION defined!"
    # if isinstance(ncs_list, list) and ncs_list:
    #     ncs_value = ncs_list[0]
    #     for cs_obj in cross_section_objects:
    #         if cs_obj.ncs == ncs_value and cs_obj.json_name:
    #             final_json_file = cs_obj.json_name[0]
    #             break

    # Resolve cross-section JSON
    final_json_file = None
    ncs_list = _to_list(getattr(obj, 'cross_section_ncs', None)) or _to_list(getattr(obj, 'internal_cross_section_ncs', None))

    # Prefer exact ncs hit
    if cross_section_objects:
        cs_by_ncs  = {getattr(cs, 'ncs', None): cs for cs in cross_section_objects if getattr(cs, 'ncs', None) is not None}
        cs_by_name = {str(getattr(cs, 'name', '')).lower(): cs for cs in cross_section_objects}

        for ncs in ncs_list:
            cs = cs_by_ncs.get(ncs)
            if cs and getattr(cs, 'json_name', None):
                final_json_file = cs.json_name[0]
                break

        # Fallback: try by name 'MASTER_*' if exact NCS has empty json_name
        if not final_json_file:
            # best-effort guesses based on object class/type
            wanted = None
            if getattr(obj, 'class_name', '') == 'DeckObject':
                wanted = 'MASTER_Deck'
            elif getattr(obj, 'class_name', '') == 'PierObject':
                wanted = 'MASTER_Pier'
            elif getattr(obj, 'class_name', '') == 'FoundationObject':
                wanted = 'MASTER_Foundation'

            if wanted:
                cs = cs_by_name.get(wanted.lower())
                if cs and getattr(cs, 'json_name', None):
                    final_json_file = cs.json_name[0]

    # If still nothing, leave None; attach step won’t crash and you’ll still see guide lines
    row_json_file = final_json_file or ""

    final_json_file_full = find_relative_file(final_json_file)
    # NEW: fallback to the relative path so SpotLoader can join with branch folder
    json_file_out = final_json_file_full or final_json_file

    # Create an axis signature to help Axis cache hits later
    def tiny_hash(arr):
        data = (','.join(str(float(v)) for v in (arr or []))).encode('utf-8')
        return hashlib.blake2b(data, digest_size=12).hexdigest()
    axis_sig = (tiny_hash(stations), tiny_hash(x_coords), tiny_hash(y_coords), tiny_hash(z_coords), 'm')

    return {
        "json_file": json_file_out,
        "stations_axis": stations,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "stations_to_plot": stations_to_plot,
        "AxisVariables": axis_variables,
        "name": getattr(obj, 'name', ''),
        "colors": colors,
        "AxisRotation": axis_rotation,
        "axis_signature": axis_sig,   # <--- NEW
    }



@dataclass(kw_only=True)
class DeckObject(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    cross_section_types: List[str]
    cross_section_names: List[str]
    grp_offset: List[float]
    placement_id: List[str]
    placement_description: List[str]
    ref_placement_id: List[str]
    ref_station_offset: List[float]
    station_value: List[float]
    cross_section_points_name: List[str]
    grp: List[str]
    cross_section_ncs: List[int]

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

@dataclass(kw_only=True)
class PierObject(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    object_axis_name: str
    cross_section_type: str
    cross_section_name: str
    axis_name: str
    ref_placement_id: str
    ref_station_offset: str
    station_value: float
    top_yoffset: float
    top_zoffset: float
    top_cross_section_ncs: int
    bot_yoffset: float
    bot_zoffset: float
    bot_cross_section_ncs: int
    bot_pier_elevation: float
    rotation_angle: float
    grp: int
    fixation: str
    internal_placement_id: List[str]
    internal_ref_placement_id: List[str]
    internal_ref_station_offset: List[float]
    internal_cross_section_ncs: List[int]
    grp_offset: List[float]
    axis_variables: List[Dict] = field(default_factory=list)

    top_cross_section_points_name: str = ""
    bot_cross_section_points_name: str = ""
    internal_station_value: List[float] = field(default_factory=list)



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
    station_rotation_x: str
    station_rotation_z: str
    sofi_code: str = field(default="", kw_only=True)

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


mapping = {
    DeckObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Axis@Name": "axis_name",
        "CrossSection@Type": "cross_section_types",
        "CrossSection@Name": "cross_section_names",
        "GrpOffset": "grp_offset",
        "PlacementId": "placement_id",
        "PlacementDescription": "placement_description",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "CrossSection_Points@Name": "cross_section_points_name",
        "Grp": "grp",
        "CrossSection@Ncs": "cross_section_ncs",
        "AxisVariables": "axis_variables"
    },
    Axis: {
        "Class": "Class",
        "Name": "Name",
        "StaionValue": "stations",
        "CurvCoorX": "x_coords",
        "CurvCoorY": "y_coords",
        "CurvCoorZ": "z_coords"
    },
    AxisVariable: {
        "VariableName": "VariableName",
        "StationValue": "VariableStations",
        "VariableValues": "VariableValues",
        "VariableIntTypes": "VariableIntTypes",
        "VariableDescription": "VariableDescription"
    },
    PierObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "ObjectAxisName": "object_axis_name",
        "CrossSection@Type": "cross_section_type",
        "CrossSection@Name": "cross_section_name",
        "Axis@Name": "axis_name",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "Top-CrossSection_Point@Name": "top_cross_section_points_name",
        'Top-CrossSection_Points@Name': 'top_cross_section_points_name',
        "Top-Yoffset": "top_yoffset",
        "Top-Zoffset": "top_zoffset",
        "Top-CrossSection@Ncs": "top_cross_section_ncs",
        'Top-CrossSection@NCS': 'top_cross_section_ncs',
        "Bot-CrossSection_Point@Name": "bot_cross_section_points_name",
        'Bot-CrossSection_Points@Name': 'bot_cross_section_points_name',
        "Bot-Yoffset": "bot_yoffset",
        "Bot-Zoffset": "bot_zoffset",
        'Bot-CrossSection@NCS': 'bot_cross_section_ncs',
        "Bot-CrossSection@Ncs": "bot_cross_section_ncs",
        "Bot-PierElevation": "bot_pier_elevation",
        "RotationAngle": "rotation_angle",
        "Grp": "grp",
        "Fixation": "fixation",
        "Internal-PlacementId": "internal_placement_id",
        "Internal_Ref-PlacementId": "internal_ref_placement_id",
        "Internal_Ref-StationOffset": "internal_ref_station_offset",
        "Internal@StationValue": "internal_station_value",
        'Internal-StationValue': 'internal_station_value',
        "Internal-CrossSection@Ncs": "internal_cross_section_ncs",
        'Internal-CrossSection@NCS': 'internal_cross_section_ncs',
        "GrpOffset": "grp_offset",
        "AxisVariables": "axis_variables"
    },
    FoundationObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "ObjectAxisName": "object_axis_name",
        "CrossSection@Type": "cross_section_type",
        "CrossSection@Name": "cross_section_name",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "CrossSection_Points@Name": "cross_section_points_name",
        "FoundationRefPoint-YOffset": "foundation_ref_point_y_offset",
        "FoundationRefPoint-XOffset": "foundation_ref_point_x_offset",
        "FoundationLevel": "foundation_level",
        "RotationAngle": "rotation_angle",
        "Axis@Name": "axis_name",
        "PierObject@Name": "pier_object_name",
        "Point1": "point1",
        "Point2": "point2",
        "Point3": "point3",
        "Point4": "point4",
        "Thickness": "thickness",
        "Grp": "grp",
        "CrossSection@Ncs2": "cross_section_ncs2",
        "Top_ZOffset": "top_z_offset",
        "Bot_ZOffset": "bot_z_offset",
        "Top_Xoffset": "top_x_offset",
        "Top_Yoffset": "top_y_offset",
        "PileDirAngle": "pile_dir_angle",
        "PileSlope": "pile_slope",
        "Kx": "kx",
        "Ky": "ky",
        "Kz": "kz",
        "Rx": "rx",
        "Ry": "ry",
        "Rz": "rz",
        "Fixation": "fixation",
        "Eval-PierObject@Name": "eval_pier_object_name",
        "Eval-StationValue": "eval_station_value",
        "Eval_Bot-CrossSection_Points@Name": "eval_bot_cross_section_points_name",
        "Eval_Bot-Yoffset": "eval_bot_y_offset",
        "Eval_Bot-Zoffset": "eval_bot_z_offset",
        "Eval_Bot-PierElevation": "eval_bot_pier_elevation",
        "Internal-PlacementId": "internal_placement_id",
        "Internal_Ref-PlacementId": "internal_ref_placement_id",
        "Internal_Ref-StationOffset": "internal_ref_station_offset",
        "Internal-StationValue": "internal_station_value",
        "Internal-CrossSection@Ncs": "internal_cross_section_ncs",
        "GrpOffset": "grp_offset",
        "AxisVariables": "axis_variables"
    },
    CrossSection: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "NCS": "ncs",
        "Material1": "material1",
        "Material2": "material2",
        "Material_Reinf": "material_reinf",
        "JSON_name": "json_name",
        "SofiCode": "sofi_code",
        "Points": "points",
        "Variables": "variables"
    },

     MainStation: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Axis@Name": "axis_name",
        "PlacementId": "placement_id",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "StationType": "station_type",
        "StationRotationX": "station_rotation_x",
        "StationRotationZ": "station_rotation_z",
        "SOFi_Code": "sofi_code",
        "SofiCode": "sofi_code",
    },

    BearingArticulation: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Axis@Name": "axis_name",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "PierObject@Name": "pier_object_name",
        "TopCrossSection@PointsName": "top_cross_section_points_name",
        "Top-YOffset": "top_yoffset",
        "Top-XOffset": "top_xoffset",
        "BotCrossSection@PointsName": "bot_cross_section_points_name",
        "Bot-YOffset": "bot_yoffset",
        "Bot-XOffset": "bot_xoffset",
        "Kx": "kx",
        "Ky": "ky",
        "Kz": "kz",
        "Rx": "rx",
        "Ry": "ry",
        "Rz": "rz",
        "GRP-Offset": "grp_offset",
        "GRP": "grp",
        "BearingDimensions": "bearing_dimensions",
        "RotationX": "rotation_x",
        "RotationZ": "rotation_z",
        "Fixation": "fixation",
        "Eval-PierObject@Name": "eval_pier_object_name",
        "Eval-StationValue": "eval_station_value",
        "Eval-TopCrossSection@PointsName": "eval_top_cross_section_points_name",
        "Eval-Top-YOffset": "eval_top_yoffset",
        "Eval-Top-ZOffset": "eval_top_zoffset",
        "SOFi_Code": "sofi_code",
    },

    SecondaryObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",

        # axes (two)
        "Beg-Axis@Name": "beg_axis_name",
        "End-Axis@Name": "end_axis_name",

        # beg side
        "Beg-PlacementId": "beg_placement_id",
        "Beg-PlacementDescription": "beg_placement_description",
        "Beg-Ref-PlacementId": "beg_ref_placement_id",
        "Beg-Ref-StationOffset": "beg_ref_station_offset",
        "Beg-StationValue": "beg_station_value",
        "Beg-CrossSection@PointsName": "beg_cross_section_points_name",
        "Beg-NCS": "beg_ncs",

        # end side
        "End-PlacementId": "end_placement_id",
        "End-PlacementDescription": "end_placement_description",
        "End-Ref-PlacementId": "end_ref_placement_id",
        "End-Ref-StationOffset": "end_ref_station_offset",
        "End-StationValue": "end_station_value",
        "End-CrossSection@PointsName": "end_cross_section_points_name",
        "End-NCS": "end_ncs",

        "GRP-Offset": "grp_offset",
        "GRP": "grp",
    }
}



# Example usage
if __name__ == "__main__":
    # Debug: Verify imports
    print(f"Imported Axis: {Axis}")  # Debug
    print(f"Imported AxisVariable: {AxisVariable}")  # Debug

    # DeckObject JSON data
    deck_data = [
        {
            "No": "1",
            "Class": "DeckObject",
            "Type": "Deck",
            "Description": "MasterDeckSection",
            "Name": "MasterDeck",
            "InActive": "",
            "Axis@Name": "RA",
            "CrossSection@Type": ["Deck", "Deck"],
            "CrossSection@Name": ["MASTER_Deck", "MASTER_Deck"],
            "GrpOffset": [],
            "PlacementId": [],
            "PlacementDescription": [],
            "Ref-PlacementId": ["Str", "End"],
            "Ref-StationOffset": [0, 0],
            "StationValue": [0, 331.3],
            "CrossSection_Points@Name": [],
            "Grp": [],
            "CrossSection@Ncs": [111, 111],
            "AxisVariables": [
                {
                    "VariableName": "TEST",
                    "StationValue": [0, 1],
                    "VariableValues": ["2.75000023841857", "2.75000023841857"],
                    "VariableIntTypes": ["#", "#"],
                    "VariableDescription": "H_QS:Axis Variable"
                },
                {
                    "VariableName": "H_QS",
                    "StationValue": [0, 1],
                    "VariableValues": ["2.75000023841857", "2.75000023841857"],
                    "VariableIntTypes": ["#", "#"],
                    "VariableDescription": "H_QS:Axis Variable"
                }
            ]
        },
        {
            "No": "",
            "Class": "ClassInfo",
            "Type": "PackingSize",
            "Description": "When creating JSON.json files you can choose whether each column vaues maps to a single value or a list of values",
            "Name": "DeckObject",
            "InActive": "",
            "Axis@Name": "",
            "CrossSection@Type": ["[many]"],
            "CrossSection@Name": ["[many]"],
            "GrpOffset": [0],
            "PlacementId": ["[many]"],
            "PlacementDescription": ["[many]"],
            "Ref-PlacementId": ["[many]"],
            "Ref-StationOffset": [0],
            "StationValue": [0],
            "CrossSection_Points@Name": ["[many]"],
            "Grp": ["[many]"],
            "CrossSection@Ncs": [0],
            "AxisVariables": {}
        }
    ]

    # Axis JSON data
    axis_data = [
        {
            "No": "",
            "Class": "ClassInfo",
            "Type": "PackingSize",
            "Description": "When creating JSON.json files you can choose whether each column vaues maps to a single value or a list of values",
            "Name": "Axis",
            "InActive": "",
            "StaionValue": ["[many]"],
            "CurvCoorX": ["[many]"],
            "CurvCoorY": ["[many]"],
            "CurvCoorZ": ["[many]"]
        },
        {
            "No": "",
            "Class": "ClassInfo",
            "Type": "PackingType",
            "Description": "When creating JSON.json files you can choose the type for each column values",
            "Name": "Axis",
            "InActive": "",
            "StaionValue": ["[Str]"],
            "CurvCoorX": ["[Str]"],
            "CurvCoorY": ["[Str]"],
            "CurvCoorZ": ["[Str]"]
        },
        {
            "No": "",
            "Class": "ClassInfo",
            "Type": "Workflow",
            "Description": "When creating JSON.json files you can mark some specific columns which are only relevant for given workflow",
            "Name": "Axis",
            "InActive": "",
            "StaionValue": [],
            "CurvCoorX": [],
            "CurvCoorY": [],
            "CurvCoorZ": []
        },
        {
            "No": "",
            "Class": "ClassInfo",
            "Type": "Unit",
            "Description": "",
            "Name": "Axis",
            "InActive": "",
            "StaionValue": ["[m]"],
            "CurvCoorX": ["[m]"],
            "CurvCoorY": ["[m]"],
            "CurvCoorZ": ["[m]"]
        },
        {
            "No": "1",
            "Class": "Axis",
            "Type": "RoadAligment",
            "Description": "MianDeckAxis",
            "Name": "RA",
            "InActive": "",
            "StaionValue": ["0", "100", "300"],
            "CurvCoorX": ["0", "100", "300"],
            "CurvCoorY": ["0", "100", "100"],
            "CurvCoorZ": ["0", "0", "0"]
        }
    ]

    axis_data_path          = r'C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\RCZ\_Axis_JSON.json'
    with open(axis_data_path, 'r') as f:
        axis_data = json.load(f)
    
    # Load CrossSection objects
    cross_section_data_path =  r'C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\RCZ\_CrossSection_JSON.json'
    cross_section_objects, _ = load_from_json(CrossSection, cross_section_data_path, mapping)



    # Load DeckObjects, passing axis_data for Axis creation
    deck_data_path          = r'C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\RCZ\_DeckObject_JSON.json'
    
    object_deck,filtered_object_deck = load_from_json(DeckObject, deck_data_path, mapping, axis_data=axis_data)
    
    # Post-creation: Set axis_variables and axis for each DeckObject
    for i, obj in enumerate(object_deck):
        if hasattr(obj, 'axis_variables') and isinstance(obj.axis_variables, list):
            obj.set_axis_variables(deck_data[i].get('AxisVariables', []), mapping.get(AxisVariable, {}))
        if hasattr(obj, 'axis_name') and axis_data is not None:
            obj.set_axis(obj.axis_name, axis_data)

    aa=object_deck[0].get_object_metada()
    summary_json = json.dumps(object_deck[0].get_object_metada(), indent=2)  
  
    print(summary_json)  # Debug: Print metadata of the first DeckObject
    # # Example with DeckObject
    # deck_obj = object_deck[0]
    # vis_data = deck_obj.get_input_for_visualisation(
    #     axis_data=axis_data,
    #     cross_section_objects=cross_section_objects,
    #     json_file_path="default/path.json"
    # )
    # print(vis_data)  # json_file will be set to the JSON_name from CrossSection with matching NCS (e.g., "MASTER_SECTION\\SectionData.json" for NCS 111)


#     aaa=object_deck[0].axis.get_plane_basis_at_station(99*1000)

#     axis=object_deck[0].axis
#  # Example points: local offsets in millimeters (mm)
#     example_points = [
#         [
#             {'id': 'P1', 'x': -1000.0, 'y': -5000.0},
#             {'id': 'P2', 'x': 1000.0, 'y': -5000.0},
#             {'id': 'P3', 'x': 1000.0, 'y': 5000.0},
#             {'id': 'P4', 'x': -1000.0, 'y': 5000.0}
#         ],
#         [
#             {'id': 'Q1', 'x': -5000.0, 'y': -1000.0},
#             {'id': 'Q2', 'x': 5000.0, 'y': -1000.0},
#             {'id': 'Q3', 'x': 5000.0, 'y': 1000.0},
#             {'id': 'Q4', 'x': -5000.0, 'y': 1000.0}
#         ],
#         [
#             {'id': 'R1', 'x': -8000.0, 'y': -8000.0},
#             {'id': 'R2', 'x': 8000.0, 'y': -8000.0},
#             {'id': 'R3', 'x': 8000.0, 'y': 8000.0},
#             {'id': 'R4', 'x': -8000.0, 'y': 8000.0}
#         ],
#         [
#             {'id': 'S1', 'x': -12000.0, 'y': -12000.0},
#             {'id': 'S2', 'x': 12000.0, 'y': -12000.0},
#             {'id': 'S3', 'x': 12000.0, 'y': 12000.0},
#             {'id': 'S4', 'x': -12000.0, 'y': 12000.0}
#         ],
#         [
#             {'id': 'T1', 'x': 0.0, 'y': 10000.0},
#             {'id': 'T2', 'x': -8660.0, 'y': -5000.0},
#             {'id': 'T3', 'x': 8660.0, 'y': -5000.0}
#         ]
#     ]

#     # Stations to plot example points (converted from meters to mm)
#     stations_to_plot = [0.0, 17.07146616 * 1000.0, 35.64154007 * 1000.0, 60.0 * 1000.0, 90.0 * 1000.0]


#     # Transform with different rotation angles for testing
#     rotation_angle = 45  # Rotate shapes 90 degrees around the longitudinal axis
#     transformed_global_points = axis.transform_points(example_points, stations_to_plot, rotation_angle=rotation_angle)

#     # Plot using transformed data
#     fig = plot_axis_and_points_transformed(axis, transformed_global_points, stations_to_plot)
#     fig.update_layout(scene=dict(aspectmode="data"))  # Safer for unequal axis scaling
#     save_plot(fig, f'axis_and_points_rotated_{rotation_angle}deg.html')

    COND_PIER_FOUNDATION = False  # Set to True if you want to load Pier and Foundation objects
    if COND_PIER_FOUNDATION:
        # Load JSON data from files
        pier_data_path          = r'C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\RCZ\_PierObject_JSON.json'
        foundation_data_path    = r'C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\RCZ\_FoundationObject_JSON.json'

        # Load PierObjects and get filtered data
        object_pier, filtered_pier_data = load_from_json(PierObject, pier_data_path, mapping, axis_data=axis_data)
        for obj in object_pier:
            if hasattr(obj, 'axis_variables') and isinstance(obj.axis_variables, list):
                obj.set_axis_variables(obj.axis_variables, mapping.get(AxisVariable, {}))
            if hasattr(obj, 'axis_name') and axis_data is not None:
                obj.set_axis(obj.axis_name, axis_data)


        # Load FoundationObjects and get filtered data
        object_foundation, filtered_foundation_data = load_from_json(FoundationObject, foundation_data_path, mapping, axis_data=axis_data)
        for obj in object_foundation:
            if hasattr(obj, 'axis_variables') and isinstance(obj.axis_variables, list):
                obj.set_axis_variables(obj.axis_variables, mapping.get(AxisVariable, {}))
            if hasattr(obj, 'axis_name') and axis_data is not None:
                obj.set_axis(obj.axis_name, axis_data)



    # Create visualization data for all objects

    vis_data_all = []
    for obj in object_deck + locals().get("object_pier", []) + locals().get("object_foundation", []):
        vis_data = obj.get_input_for_visualisation(
            axis_data=axis_data,
            cross_section_objects=cross_section_objects,
            json_file_path="default/path.json"
        )
        vis_data_all.append(vis_data)



    # vis_data_all = []
    # for obj in object_deck + object_pier + object_foundation:
    #     vis_data = obj.get_input_for_visualisation(
    #         axis_data=axis_data,
    #         cross_section_objects=cross_section_objects,
    #         json_file_path="default/path.json"
    #     )
    #     vis_data_all.append(vis_data)

    # Print all visualization data
    print("---------------------------")

    with open("input_.json", "w", encoding="utf-8") as f:
        json.dump(vis_data_all, f, indent=2)

    print(vis_data_all)

