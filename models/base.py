from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, ClassVar, Tuple, Union, get_args, get_origin
import logging

from .axis import Axis
from .axis_variable import AxisVariable
from .cross_section import CrossSection

log = logging.getLogger(__name__)

VERBOSE = False

from dataclasses import fields, is_dataclass, asdict
from typing import Any, Iterable, List, Union, get_args, get_origin
import logging

log = logging.getLogger(__name__)

def _get_cls_map(cls, mapping_config: dict) -> dict:
    """Find the mapping block for a class, accepting class object or name."""
    if cls in mapping_config:
        return mapping_config[cls] or {}
    name = getattr(cls, "__name__", str(cls))
    if name in mapping_config:
        return mapping_config[name] or {}
    return {}

def _eval_spec(row: dict, spec):
    """
    Evaluate a mapping spec against a data row.
    - callable -> call(row)
    - str -> row.get(str)
    - (list/tuple) -> first present key in row
    - else -> None
    """
    if callable(spec):
        try:
            return spec(row)
        except Exception as e:
            log.debug("mapping callable raised: %s", e, exc_info=True)
            return None
    if isinstance(spec, (list, tuple)):
        for k in spec:
            if k in row:
                v = row.get(k)
                if v not in (None, ""):
                    return v
        return None
    if isinstance(spec, str):
        return row.get(spec)
    return None

def _coerce_value(val: Any, tp):
    origin = get_origin(tp) or getattr(tp, "__origin__", None)
    args   = get_args(tp)   or getattr(tp, "__args__", ())

    def _to_num(v, caster):
        try:
            if isinstance(v, str) and not v.strip():
                return v
            return caster(v)
        except Exception:
            return v

    if tp in (int, float, bool):
        caster = {int: int, float: float, bool: lambda x: bool(int(x)) if isinstance(x, (str, int)) else bool(x)}[tp]
        return _to_num(val, caster)

    if origin is Union:
        for a in args:
            if a is type(None):
                continue
            return _coerce_value(val, a)

    if origin in (list, set, tuple):
        inner = args[0] if args else Any
        # accept singletons & sequences
        if isinstance(val, (list, tuple, set)):
            seq = list(val)
        elif val is None:
            seq = []
        else:
            seq = [val]
        coerced = [_coerce_value(v, inner) for v in seq]
        # preserve container type if already a sequence
        if isinstance(val, tuple):
            return tuple(coerced)
        if isinstance(val, set):
            return set(coerced)
        return coerced

    return val

def from_dict(cls, row: dict, mapping_config: dict, *, axis_data=None, verbose: bool=False):
    """
    Generic factory that works with your current py->JSON mapping:
    mapping[field] may be a callable(row), a header name, or a list/tuple of header candidates.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    cls_map = _get_cls_map(cls, mapping_config)
    fdefs = {f.name: f for f in fields(cls)}
    kwargs = {}

    for fname, fdef in fdefs.items():
        spec = cls_map.get(fname)
        if spec is None:
            continue
        raw_val = _eval_spec(row, spec)
        kwargs[fname] = _coerce_value(raw_val, fdef.type)

        if verbose and fname in (
            "station_value", "ref_station_offset", "top_cross_section_ncs",
            "bot_cross_section_ncs", "internal_cross_section_ncs",
            "bot_zoffset", "top_zoffset"
        ):
            log.debug("[from_dict] %s.%s <- %r -> %r", getattr(cls, "__name__", cls), fname, raw_val, kwargs[fname])

    # Don’t wire axis_obj here; your VisoContext already does that reliably.
    # Keep axis_name if present in kwargs.

    if verbose:
        log.debug("[from_dict] Creating %s with kwargs keys=%s", getattr(cls, "__name__", cls), list(kwargs.keys()))
    return cls(**kwargs)

def _build_axis_index(axis_data, axis_map=None):
    """
    Minimal compatibility stub used by some modules.
    Returns a dict: {axis_name: axis_dict}
    """
    idx = {}
    if not axis_data:
        return idx
    try:
        for ax in axis_data:
            if isinstance(ax, dict):
                name = ax.get("name")
                if name:
                    idx[str(name)] = ax
            else:
                name = getattr(ax, "name", None)
                if name:
                    idx[str(name)] = asdict(ax)
    except Exception:
        pass
    return idx

@dataclass(slots=True)
class BaseObject:
    """
    Shared behavior & state for DeckObject / PierObject / FoundationObject.
    This class is now storage-agnostic: it only holds python attributes
    and relies on the loader/context for wiring.
    """
    # identity
    name: str = ""
    axis_name: str = ""

    # axis & variables (wired by loader/context)
    axis_obj: Optional[Axis] = None
    axis_variables: List[Dict] = field(default_factory=list)            # raw rows (already python keys)
    axis_variables_obj: List[AxisVariable] = field(default_factory=list)

    # user display / config
    user_stations: List[float] = field(default_factory=list)
    axis_rotation: float = 0.0
    colors: Dict[str, str] = field(default_factory=lambda: {
        "axis": "black",
        "first_station_points": "blue",
        "cross_section_loops": "red",
        "longitudinal_lines": "gray",
    })

    # memo & caches
    _memo_cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _TRANSFORM_CACHE: ClassVar[Dict] = {}
    _TRANSFORM_CACHE_MAX: ClassVar[int] = 256

    # ------- tiny utils -------
    def memo(self, key, builder):
        m = self._memo_cache
        if key not in m:
            m[key] = builder()
        return m[key]

    def debug(self, *a, **k):
        if VERBOSE:
            print(*a, **k)

    # ------- axis variables -------
    # models/base.py (keep/replace your current one if it is weaker)
    def set_axis_variables(self, axis_var_map: dict | None = None):
        """
        Parse self.axis_variables (raw rows) into self.axis_variables_obj (List[AxisVariable]).
        axis_var_map is your field mapping for VariableName, VariableStations, ...
        """
        from .axis_variable import AxisVariable

        raw = getattr(self, "axis_variables", None) or []
        self.axis_variables_obj = AxisVariable.create_axis_variables(raw)




    def evaluate_axis_vars_at_stations(self, stations_m: List[float]) -> List[Dict[str, float]]:
        if not stations_m:
            return []
        self.set_axis_variables(axis_var_map={})
        return AxisVariable.evaluate_at_stations_cached(self.axis_variables_obj or [], stations_m)

    # ------- transform caching -------
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
            raise RuntimeError("axis_obj is not set (loader/context should wire it).")
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

    # ------- visualization input -------
    def build_viso_row(
        self,
        ctx=None,
        *,
        axis: Optional[Axis] = None,
        mainstations: Optional[dict] = None,
        cross_sections_override: Optional[List[CrossSection]] = None,
        mapping_cfg: Dict = None,
    ) -> dict:
        """
        Build one 'viso row' for this object:
        - inject axis by name (if not already on self)
        - resolve cross-sections by NCS/name via ctx
        - parse AxisVariables into axis_variables_obj
        - return the classic dict your plotting adapter expects
        """
        from .vis_adapter import build_viso_object  # local import to avoid cycles
        from mapping import mapping as MAP
        
        # Make sure axis_variables_obj exists
        if hasattr(self, "axis_variables") and isinstance(self.axis_variables, list):
            self.set_axis_variables((mapping_cfg or MAP).get(AxisVariable, {}))

        row = build_viso_object(
            obj=self,
            ctx=ctx,
            axis=axis,
            mainstations=mainstations,
            cross_sections_override=cross_sections_override,
            mapping_cfg=(mapping_cfg or MAP),
        )
        # convenience to carry the back-reference
        row["obj"] = self
        return row

    def get_input_for_visualisation(
        self,
        *,
        cross_section_objects: Optional[List[CrossSection]] = None,
        axis_rotation: Optional[float] = None,
        colors: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Delegate to your existing vis adapter. No raw JSON here.
        Requires axis_obj to be set (or use context helper below).
        """
        from .vis_adapter import create_input_for_visualisation
        return create_input_for_visualisation(
            obj=self,
            axis_data=None,  # not needed anymore
            cross_section_objects=cross_section_objects,
            json_file_path=None,
            axis_rotation=axis_rotation,
            colors=colors,
        )

    # ------- helpers for dependent lookups -------
    def _resolve_cross_sections_from_ncs(self, ctx) -> List[CrossSection]:
        ncs_list = getattr(self, "cross_section_ncs", []) or []
        out: List[CrossSection] = []
        for n in ncs_list:
            try:
                cs = ctx.crosssec_by_ncs.get(int(n))
            except Exception:
                cs = None
            if cs:
                out.append(cs)
        return out
    
    @classmethod
    def from_row(cls, row: dict, mapping_config: dict, *, axis_data=None, verbose: bool=False):
        return from_dict(cls, row, mapping_config, axis_data=axis_data, verbose=verbose)



# ── helpers used above ─────────────────────────────────────────────────────────
def _tiny_hash(arr) -> str:
    import hashlib
    if arr is None:
        arr = []
    data = (','.join(str(float(v)) for v in arr)).encode('utf-8')
    return hashlib.blake2b(data, digest_size=12).hexdigest()

# def _as_floats(val) -> List[float]:
#     out = []
#     if val is None: 
#         return out
#     if not isinstance(val, (list, tuple)):
#         val = [val]
#     for v in val:
#         try:
#             out.append(float(v))
#         except Exception:
#             pass
#     return out

# def _as_str_list(val) -> List[str]:
#     if val is None:
#         return []
#     if isinstance(val, str):
#         return [val]
#     if isinstance(val, (list, tuple)):
#         return [str(x) for x in val]
#     return [str(val)]

# def _axis_vars_signature(rows: List[Dict]) -> str:
#     import hashlib, json
#     try:
#         blob = json.dumps(rows or [], sort_keys=True, ensure_ascii=False, default=str)
#     except Exception:
#         blob = str(rows)
#     return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

# def create_axis_variables(json_data: List[Dict], var_map: Dict) -> List[AxisVariable]:
#     # Uses your existing AxisVariable API; this expects keys already normalized.

#     name_k = var_map.get("VariableName",       "VariableName")
#     xs_k   = var_map.get("StationValue",       "StationValue")      # <-- was "VariableStations"
#     ys_k   = var_map.get("VariableValues",     "VariableValues")
#     typ_k  = var_map.get("VariableIntTypes",   "VariableIntTypes")
#     desc_k = var_map.get("VariableDescription","VariableDescription")
#     ...

#     normalized: List[Dict] = []
#     for row in (json_data or []):
#         mapped = { var_map.get(k, k): v for k, v in row.items() }
#         name = str(mapped.get(name_k, "")).strip()
#         xs   = _as_floats(mapped.get(xs_k))
#         ys   = _as_floats(mapped.get(ys_k))
#         tys  = _as_str_list(mapped.get(typ_k))
#         desc = mapped.get(desc_k, "")
#         if len(xs) != len(ys):
#             n = min(len(xs), len(ys))
#             xs, ys = xs[:n], ys[:n]
#         pairs = sorted(zip(xs, ys), key=lambda t: t[0])
#         xs = [p[0] for p in pairs]
#         ys = [p[1] for p in pairs]
#         if not name or not xs:
#             continue
#         normalized.append({
#             name_k: name,
#             xs_k:   xs,
#             ys_k:   ys,
#             typ_k:  tys or ["linear"],
#             desc_k: desc,
#         })
#     return AxisVariable.create_axis_variables(normalized)

# #── Multi-candidate raw-key resolver ────────────────────────────────────────────
# def _raw_candidates_for(internal: str, cls_map: dict[str, str]) -> list[str]:
#     """
#     All raw keys that map to this internal field (handles aliases like
#     'Top-CrossSection@Ncs' and 'Top-CrossSection@NCS').
#     """
#     return [rk for rk, v in (cls_map or {}).items() if v == internal]

# def _pick_raw_key(internal: str, cls_map: dict[str, str], data: dict, fallback_same_name=True) -> str | None:
#     """
#     Pick the first raw key for 'internal' that exists in 'data'.
#     If none exists and fallback_same_name is True, try the internal name itself.
#     """
#     for k in _raw_candidates_for(internal, cls_map):
#         if k in data:
#             return k
#     if fallback_same_name and internal in data:
#         return internal
#     return None

# # ── Robust Axis type detector ───────────────────────────────────────────────────
# def _is_axis_type(tp) -> bool:
#     """
#     Returns True if the type annotation contains models.axis.Axis anywhere:
#     Axis, Optional[Axis], Union[Axis, None], List[Axis], Tuple[Axis, ...],
#     or a forward-ref string like "Axis" / "models.axis.Axis".
#     """
#     if tp is None:
#         return False

#     # string forward-ref
#     if isinstance(tp, str):
#         return tp.split(".")[-1] == "Axis"

#     AxisClass = None
#     try:
#         from models.axis import Axis as AxisClass  # your real Axis class
#     except Exception:
#         pass

#     if AxisClass is not None and tp is AxisClass:
#         return True

#     origin = get_origin(tp) or getattr(tp, "__origin__", None)
#     args   = get_args(tp)   or getattr(tp, "__args__", ())

#     if origin is None:
#         # tolerate direct class compare by name
#         return getattr(tp, "__name__", "") == "Axis"

#     # containers
#     if origin in (list, set, tuple):
#         return any(_is_axis_type(a) for a in args)

#     # Optional/Union
#     if origin is Union:
#         return any(_is_axis_type(a) for a in args if a is not type(None))

#     return False