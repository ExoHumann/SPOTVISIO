from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# utils must expose these names; same as your existing module
from utils import (
    _compile_expr,        # compiles a safe expression once
    _sanitize_vars,       # sanitizes scalar env
    _SCALAR_FUNCS,        # {'sin': ..., 'cos': ..., 'Pi': ...} (scalar)
    _VECTOR_FUNCS,        # same, but numpy versions
    _RESERVED_FUNC_NAMES, # tokens to ignore as "variables"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers (only your schema: Points + Loops)
# -----------------------------------------------------------------------------

def _normalize_point_row(p: dict) -> dict:
    """
    Canonicalize a point record to:
      {'Id': str, 'Coord': [str, str], 'Reference': [str], 'ReferenceType': 'Euclidean'|... (optional)}
    Accepts:
      - normal JSON points (Id, Coord, Reference?)
      - legacy table points (PointName, CoorY, CoorZ, CoorYVal, CoorZVal)
    """
    if "Id" in p and ("Coord" in p or "coord" in p):
        pid = str(p.get("Id"))
        coord = p.get("Coord") or p.get("coord") or [0, 0]
        ref   = p.get("Reference") or []
        rtype = p.get("ReferenceType") or p.get("Type")
        return {"Id": pid, "Coord": [str(coord[0]), str(coord[1])], "Reference": list(ref or []), "ReferenceType": rtype}

    # legacy row shape (Crossection_JSON rows)
    pid = str(p.get("PointName") or p.get("Id") or p.get("id") or "")
    def _pick(expr, val):
        txt = str(expr) if expr is not None else ""
        if not txt or txt.strip().lower().startswith("error"):
            if val is None or str(val).strip() == "":
                return "0"
            return str(val)
        return txt

    y_expr = _pick(p.get("CoorY"), p.get("CoorYVal"))
    z_expr = _pick(p.get("CoorZ"), p.get("CoorZVal"))
    ref    = p.get("Reference") or []
    rtype  = p.get("ReferenceType") or p.get("Type")
    return {"Id": pid, "Coord": [y_expr, z_expr], "Reference": list(ref or []), "ReferenceType": rtype}


def _collect_all_points(section_json: dict) -> List[dict]:
    """
    Flatten all point-like definitions we care about into canonical rows.
    Only the keys present in your samples are considered.
    """
    pts: List[dict] = []
    seen: set[str] = set()

    def _add(p):
        if not isinstance(p, dict):
            return
        q = _normalize_point_row(p)
        pid = q.get("Id")
        if pid and pid not in seen:
            seen.add(pid)
            pts.append(q)

    jd = section_json or {}

    # top-level Points
    for p in (jd.get("Points") or []):
        _add(p)

    # Loops -> Points
    for lp in (jd.get("Loops") or []):
        for p in (lp.get("Points") or []):
            _add(p)

    # Reinforcements (PointReinforcements / LineReinforcements)
    for pr in (jd.get("PointReinforcements") or []):
        _add(pr.get("Point", {}))
    for lr in (jd.get("LineReinforcements") or []):
        _add(lr.get("PointStart", {}))
        _add(lr.get("PointEnd", {}))

    # NonEffectiveZones
    for nez in (jd.get("NonEffectiveZones") or []):
        _add(nez.get("PointStart", {}))
        _add(nez.get("PointEnd", {}))

    return pts


def _loops_as_id_sequences(section_json: dict) -> List[List[str]]:
    """
    Return Loops as list of list of point Ids, strictly from JSON['Loops'].
    """
    out: List[List[str]] = []
    for lp in (section_json or {}).get("Loops") or []:
        ids = []
        for p in lp.get("Points") or []:
            pid = p.get("Id")
            if pid is not None and str(pid).strip():
                ids.append(str(pid))
        if len(ids) >= 2:
            out.append(ids)
    return out


def _collect_used_variable_names(points: List[dict]) -> set[str]:
    """
    Find variable-like tokens used in Coord expressions (to pre-allocate arrays).
    """
    used: set[str] = set()
    for p in points:
        coord = p.get("Coord") or [0, 0]
        for expr in coord[:2]:
            txt = str(expr)
            token = ""
            for ch in txt:
                if ch.isalnum() or ch == "_":
                    token += ch
                else:
                    if token and token not in _RESERVED_FUNC_NAMES and not token[0].isdigit():
                        used.add(token)
                    token = ""
            if token and token not in _RESERVED_FUNC_NAMES and not token[0].isdigit():
                used.add(token)
    return used


# -----------------------------------------------------------------------------
# CrossSection
# -----------------------------------------------------------------------------

@dataclass(kw_only=True)
class CrossSection:
    # --- user / table fields (kept minimal) ----------------------------------
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

    json_name: Union[str, List[str], None] = None  # "MASTER_SECTION/SectionData.json" or list
    variables: Union[List[Dict[str, Any]], Dict[str, Any], None] = None
    points: Optional[List[Dict[str, Any]]] = None  # canonicalized, filled by inflate
    sofi_code: Optional[str] = None

    # runtime JSON (source of truth)
    json_data: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    json_source: Optional[str] = field(default=None, init=False, repr=False)

    # DAG cache
    _dag_order: List[str] = field(default_factory=list, init=False, repr=False)
    _dag_by_id: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    # -------------------------------------------------------------------------
    # Hygiene
    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        try:
            if self.ncs is not None and str(self.ncs).strip():
                self.ncs = int(self.ncs)
        except Exception:
            logger.debug("CrossSection(%r): failed to int(ncs=%r)", self.name, self.ncs)

        for attr in ("material1", "material2", "material_reinf"):
            v = getattr(self, attr)
            try:
                if v is not None and str(v).strip():
                    setattr(self, attr, int(v))
            except Exception:
                logger.debug("CrossSection(%r): failed to int(%s=%r)", self.name, attr, v)

    # -------------------------------------------------------------------------
    # Geometry attach / inflate (owned here, not in the loader)
    # -------------------------------------------------------------------------
    def attach_json_payload(self, payload: Dict[str, Any], *, source: Optional[str] = None) -> None:
        if isinstance(payload, dict):
            self.json_data = payload
            self.json_source = source

    def inflate_points_from_json(self) -> None:
        """
        Surface a de-duplicated, canonical list of points from json_data (Points + Loop points).
        """
        if not isinstance(self.json_data, dict):
            return
        flat = _collect_all_points(self.json_data)
        self.points = flat

    def prime_caches(self) -> None:
        """
        Ensure points exist and build the point dependency DAG once.
        """
        if not (isinstance(self.points, list) and self.points) and isinstance(self.json_data, dict):
            self.inflate_points_from_json()

        # DAG
        by_id: Dict[str, dict] = {}
        deps: Dict[str, set] = {}
        for p in (self.points or []):
            pid = str(p.get("Id") or p.get("id"))
            by_id[pid] = p
            r = p.get("Reference") or []
            deps[pid] = set(str(x) for x in (r or []) if x is not None)

        # Kahn
        incoming = {k: set(v) for k, v in deps.items()}
        outgoing: Dict[str, set] = {k: set() for k in deps}
        for k, vs in deps.items():
            for v in vs:
                outgoing.setdefault(v, set()).add(k)

        order: List[str] = []
        roots = sorted([k for k, s in incoming.items() if not s])
        from collections import deque
        q = deque(roots)
        while q:
            u = q.popleft()
            order.append(u)
            for w in list(outgoing.get(u, ())):
                incoming[w].discard(u)
                if not incoming[w]:
                    q.append(w)

        remaining = [k for k in deps.keys() if k not in order]
        if remaining:
            logger.warning("CrossSection DAG unresolved deps or cycles: %s", remaining)
            order.extend(sorted(remaining))

        self._dag_order = order
        self._dag_by_id = by_id

    # Optional: let the class load its own JSON, so loader stays dumb.
    def ensure_geometry_loaded(
        self,
        *,
        json_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
        default_json_by_type: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        If no json_data is attached yet, try to load it from json_name (str or list),
        falling back to a small per-Type table (paths relative to repo or absolute).
        """
        if isinstance(self.json_data, dict):
            self.prime_caches()
            return

        # tiny defaults (can be overridden by caller)
        default_table = default_json_by_type or {
            "Deck":        ["MASTER_SECTION/SectionData.json"],
            "Foundation":  ["MASTER_SECTION/MASTER_Foundation.json"],
            "Pier":        ["MASTER_SECTION/MASTER_Pier.json"],
        }

        # default loader that tries the path as-is (or relative to CWD)
        def _default_loader(path: str) -> Dict[str, Any]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        jl = json_loader or _default_loader

        # collect candidates
        candidates: List[str] = []
        jn = self.json_name
        if isinstance(jn, str) and jn.strip():
            candidates.append(jn.strip())
        elif isinstance(jn, (list, tuple)):
            candidates += [s for s in jn if isinstance(s, str) and s.strip()]

        candidates += default_table.get((self.type or "").strip(), [])

        for cand in candidates:
            data = jl(cand) or {}
            if isinstance(data, dict) and (isinstance(data.get("Points"), list) or isinstance(data.get("Loops"), list)):
                self.attach_json_payload(data, source=cand)
                break

        if isinstance(self.json_data, dict):
            self.inflate_points_from_json()
            self.prime_caches()

    # -------------------------------------------------------------------------
    # Variables (defaults) in mm
    # -------------------------------------------------------------------------
    def get_defaults(self) -> Dict[str, float]:
        """
        Read variable defaults either from attached json_data['Variables'] (preferred)
        or from self.variables (list/dict). Values are floats in *mm*.
        """
        jd = getattr(self, "json_data", None)
        if isinstance(jd, dict) and isinstance(jd.get("Variables"), dict):
            out: Dict[str, float] = {}
            for k, v in jd["Variables"].items():
                try: out[str(k)] = float(v)
                except Exception: pass
            if out: return out

        out: Dict[str, float] = {}
        raw = self.variables or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try: out[str(k)] = float(v)
                except Exception: pass
        else:
            for row in raw or []:
                if not isinstance(row, dict): continue
                try:
                    n = str(row.get("VariableName"))
                    out[n] = float(row.get("VariableValue", 0.0) or 0.0)
                except Exception: pass
        return out

    # -------------------------------------------------------------------------
    # Vectorized expression evaluation (local Y/Z; X is your local-Y)
    # -------------------------------------------------------------------------
    @staticmethod
    def _compile_pair(expr_x: str, expr_y: str):
        return _compile_expr(str(expr_x)), _compile_expr(str(expr_y))

    @staticmethod
    def _build_var_arrays_from_results(results: List[Dict[str, float]],
                                       defaults: Dict[str, float],
                                       keep: Optional[set] = None) -> Dict[str, np.ndarray]:
        names = set(keep or [])
        if not names:
            for d in results or []:
                names.update(d.keys())
        out: Dict[str, np.ndarray] = {}
        S = len(results or [])
        for name in names:
            arr = np.full(S, np.nan, dtype=float)
            for i, row in enumerate(results or []):
                try:
                    arr[i] = float(row.get(name, defaults.get(name, 0.0)))
                except Exception:
                    arr[i] = float(defaults.get(name, 0.0))
            out[name] = arr
        return out

    @staticmethod
    def _fix_var_units_inplace(arrs: Dict[str, np.ndarray], defaults: Dict[str, float]) -> None:
        """
        Practical heuristic for your data:
        - most axis variables are *meters* in AxisVariable; convert to mm (×1000).
        - angle-like variables (W_/Q_/INCL_) are typically degrees in JSON defaults;
          AxisVariable often stores values like 0.075 (which is 75e-3 deg) -> ×1000 to get ~75.
        We therefore default to ×1000 unless a default suggests otherwise.
        """
        for name, a in list(arrs.items()):
            a = np.asarray(a, float)
            if a.size == 0:
                continue
            s = 1000.0  # default: meters->mm or milli-deg -> deg
            def_v = defaults.get(name)
            if def_v is not None and np.isfinite(def_v):
                # choose 1 or 1000, whichever matches default better
                med = float(np.nanmedian(np.abs(a))) or 0.0
                cands = (1.0, 1000.0)
                costs = [abs(med*s - def_v)/max(1.0, abs(def_v)) for s in cands]
                s = cands[int(np.argmin(costs))]
            arrs[name] = a * s

    def get_coordinates_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
        stations_count: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Evaluate Coord expressions for all points across all stations.
        Returns (ids, X_mm, Y_mm) with shape (S, N) each.
        """
        ids = list(order)
        S = int(stations_count or (len(next(iter(var_arrays.values()))) if var_arrays else 0))
        N = len(ids)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        env_base = {**_VECTOR_FUNCS}
        for k, arr in var_arrays.items():
            env_base[k] = np.asarray(arr, float)

        for j, pid in enumerate(ids):
            pj = by_id.get(pid) or {}
            coord = pj.get("Coord") or [0, 0]
            cx, cy = self._compile_pair(coord[0], coord[1])
            env = env_base
            try:
                xv = np.asarray(eval(cx, {"__builtins__": {}}, env), float)
            except Exception as e:
                logger.error("Expr error X for point '%s' in section '%s': %s", pid, getattr(self, "name", "?"), e)
                xv = np.full(S, np.nan, float)
            try:
                yv = np.asarray(eval(cy, {"__builtins__": {}}, env), float)
            except Exception as e:
                logger.error("Expr error Y for point '%s' in section '%s': %s", pid, getattr(self, "name", "?"), e)
                yv = np.full(S, np.nan, float)

            X[:, j] = xv
            Y[:, j] = yv

        # Euclidean reference shifts (historic: first ref adds X, second adds Y)
        id2col = {pid: idx for idx, pid in enumerate(ids)}
        for j, pid in enumerate(ids):
            refs = by_id.get(pid, {}).get("Reference") or []
            if not refs:
                continue
            if len(refs) == 1:
                k = id2col.get(str(refs[0]))
                if k is not None:
                    X[:, j] += X[:, k]; Y[:, j] += Y[:, k]
            else:
                kx = id2col.get(str(refs[0]))
                ky = id2col.get(str(refs[1]))
                if kx is not None: X[:, j] += X[:, kx]
                if ky is not None: Y[:, j] += Y[:, ky]

        if negate_x:
            X = -X

        return ids, X, Y

    def _loops_idx(self, ids: List[str]) -> List[np.ndarray]:
        """
        Loops as arrays of column indices in (S, N) matrices.
        """
        id_to_col = {pid: j for j, pid in enumerate(ids)}
        loops_ids = _loops_as_id_sequences(self.json_data or {})
        out: List[np.ndarray] = []
        for seq in loops_ids:
            idx = [id_to_col.get(pid) for pid in seq]
            idx = [i for i in idx if i is not None]
            if len(idx) >= 2:
                out.append(np.asarray(idx, dtype=int))
        return out

    # -------------------------------------------------------------------------
    # Public: compute local YZ (vectorized) and embed to 3D using Axis
    # -------------------------------------------------------------------------
    def compute_local_points(
        self, *, axis_var_results: List[Dict[str, float]], negate_x: bool = True
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Returns:
          ids, X_mm:(S,N), Y_mm:(S,N), loops_idx(list of index arrays)
        """
        # make sure we’re ready
        if not self._dag_order:
            self.prime_caches()

        # minimal JSON "view" for variable scan: points we’ll actually evaluate
        pts_all = self.points or []
        used = _collect_used_variable_names(pts_all)
        defaults = self.get_defaults()

        var_arrays = self._build_var_arrays_from_results(axis_var_results, defaults, keep=used)
        self._fix_var_units_inplace(var_arrays, defaults)

        ids, X_mm, Y_mm = self.get_coordinates_vectorized(
            var_arrays=var_arrays,
            order=self._dag_order,
            by_id=self._dag_by_id,
            negate_x=negate_x,
            stations_count=len(axis_var_results),
        )
        loops_idx = self._loops_idx(ids)
        return ids, X_mm, Y_mm, loops_idx

    def compute_embedded_points(
        self,
        *,
        axis,                          # models.axis.Axis
        axis_var_results: List[Dict[str, float]],
        stations_m: List[float],
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Full path: filter stations to axis domain, vector-eval local points, embed via Axis.

        Returns:
          ids, stations_mm, P_mm:(S,N,3), X_mm:(S,N), Y_mm:(S,N), loops_idx
        """
        if axis is None or not stations_m:
            empty3 = np.zeros((0, 0, 3), float)
            empty2 = np.zeros((0, 0), float)
            return [], np.array([], float), empty3, empty2, empty2, []

        stations_mm_all = np.asarray(stations_m, float) * 1000.0
        smin = float(np.min(axis.stations))
        smax = float(np.max(axis.stations))
        keep = (stations_mm_all >= smin) & (stations_mm_all <= smax)
        if not np.any(keep):
            empty3 = np.zeros((0, 0, 3), float)
            empty2 = np.zeros((0, 0), float)
            return [], np.array([], float), empty3, empty2, empty2, []

        stations_mm = stations_mm_all[keep]
        kept_results = [axis_var_results[i] for i, k in enumerate(keep) if k]

        ids, X_mm, Y_mm, loops_idx = self.compute_local_points(
            axis_var_results=kept_results, negate_x=negate_x
        )

        # shape for Axis.embed_section_points_world: (S, N, 2)
        if X_mm.size and Y_mm.size:
            local_yz = np.stack([X_mm, Y_mm], axis=2)  # (S, N, 2)
        else:
            local_yz = np.zeros((len(stations_mm), 0, 2), float)

        P_mm = axis.embed_section_points_world(
            stations_mm, yz_points_mm=local_yz, x_offsets_mm=None, rotation_deg=float(twist_deg or 0.0)
        )

        # helpful logs
        S, N = X_mm.shape if X_mm.size else (len(kept_results), 0)
        logger.info("Section %s (ncs=%s): stations=%d, points=%d",
                    getattr(self, "name", "?"), getattr(self, "ncs", "?"), S, N)

        if "UL" in ids and S > 0:
            j = ids.index("UL")
            logger.info("UL @ first station (Y,Z mm): %.3f, %.3f", X_mm[0, j], Y_mm[0, j])

        nan_xy = int(np.isnan(X_mm).sum() + np.isnan(Y_mm).sum())
        if nan_xy:
            logger.warning("NaNs in section %s: %d total", getattr(self, "name", "?"), nan_xy)

        return ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx

    # -------------------------------------------------------------------------
    # Legacy scalar preview (kept small; optional)
    # -------------------------------------------------------------------------
    class ReferenceFrame:
        """Small scalar path; useful for quick checks."""
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
                logger.debug("Scalar eval error %r: %s", string_equation, e)
                return float("nan")

        def _euclid(self, coords):
            x = self.eval_equation(coords[0]); y = self.eval_equation(coords[1])
            return {'coords': {'x': x, 'y': y}}

        def get_coordinates(self, coords):
            rt = (self.reference_type or '').lower()
            if rt in ("c", "carthesian", "e", "euclidean"): return self._euclid(coords)
            if rt in ("p", "polar"):
                r = self.eval_equation(coords[0]); a = math.radians(self.eval_equation(coords[1]))
                return {'coords': {'x': r*math.cos(a), 'y': r*math.sin(a)}}
            return self._euclid(coords)

    def compute_local_points_scalar(self, env_vars: Dict[str, float]) -> List[Dict[str, float]]:
        out = []
        for p in (self.points or []):
            coord = p.get("Coord") or [0, 0]
            rf = CrossSection.ReferenceFrame(reference_type='euclidean', reference=p.get("Reference"), points=out, variables=env_vars)
            xy = rf.get_coordinates(coord)["coords"]
            out.append({"id": p.get("Id") or p.get("id"), "x": float(xy["x"]), "y": float(xy["y"])})
        return out

    # -------------------------------------------------------------------------
    # Friendly constructor from row (uses mapping elsewhere; loader stays simple)
    # -------------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        row: Dict[str, Any],
        *,
        json_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
        default_json_by_type: Optional[Dict[str, List[str]]] = None,
        debug: bool | None = None,
    ) -> "CrossSection":
        obj = cls(
            no=row.get("No"),
            class_name=row.get("Class"),
            type=row.get("Type"),
            description=row.get("Description"),
            name=row.get("Name"),
            inactive=row.get("InActive"),
            ncs=row.get("NCS") or row.get("Ncs") or row.get("ncs"),
            material1=row.get("Material1"),
            material2=row.get("Material2"),
            material_reinf=row.get("Material_Reinf"),
            json_name=row.get("JSON_name") or row.get("JsonName") or row.get("json_name"),
            variables=row.get("Variables"),
            sofi_code=row.get("SofiCode") or row.get("SOFiCode") or row.get("SOFiSTiKCustomCode"),
        )

        # normalize json_name to list[str] | str | None (leave as-is; ensure_geometry_loaded will handle it)
        jn = obj.json_name
        if isinstance(jn, list) and len(jn) == 1:
            if isinstance(jn[0], str) and jn[0].strip():
                obj.json_name = jn  # keep list
        elif isinstance(jn, str):
            obj.json_name = jn

        # If you want the class to auto-load geometry, call:
        obj.ensure_geometry_loaded(json_loader=json_loader, default_json_by_type=default_json_by_type)
        return obj


# ----------------------------------------------------------------------------- #
# Minimal test (runs if you execute this file directly)
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # tiny axis stub: stations in mm, embed copies Y/Z into world at x=0 (no twist)
    class _AxisStub:
        def __init__(self, stations_mm):
            self.stations = np.asarray(stations_mm, float)

        def embed_section_points_world(self, stations_mm, yz_points_mm, x_offsets_mm=None, rotation_deg=0.0):
            # expected shape: (S, N, 2) -> return (S, N, 3)
            if yz_points_mm.ndim != 3 or yz_points_mm.shape[2] != 2:
                raise ValueError("yz_points_mm must be (S, N, 2).")
            S, N, _ = yz_points_mm.shape
            P = np.zeros((S, N, 3), float)
            # simple: world X=0, Y=local_Y, Z=local_Z
            P[:, :, 1] = yz_points_mm[:, :, 0]
            P[:, :, 2] = yz_points_mm[:, :, 1]
            return P

    # sample JSON (short)
    json_data = {
        "Points": [
            {"Id": "O",  "Coord": ["0", "0"]},
            {"Id": "UM", "Coord": ["0", "H_QS"]},
            {"Id": "UL", "Coord": ["1000", "H_QS"]},
        ],
        "Loops": [
            {"Points": [{"Id": "O"}, {"Id": "UM"}, {"Id": "UL"}]}
        ],
        "Variables": {"H_QS": 2750.0}
    }

    cs = CrossSection.from_dict({
        "No": "1", "Class": "CrossSection", "Type": "Deck", "Name": "MASTER_Deck",
        "NCS": 111, "JSON_name": [], "Variables": [{"VariableName": "H_QS", "VariableValue": 2750.0}],
    })
    cs.attach_json_payload(json_data)
    cs.inflate_points_from_json()
    cs.prime_caches()

    # two stations; axis variables in meters here to exercise ×1000
    axis_vars = [{"H_QS": 2.75}, {"H_QS": 2.75}]
    ids, X, Y, loops = cs.compute_local_points(axis_var_results=axis_vars, negate_x=True)
    print("ids:", ids)
    print("X[0,:]:", X[0])
    print("Y[0,:]:", Y[0])
    print("loops:", loops)

    axis = _AxisStub(stations_mm=[0.0, 1000.0, 2000.0])
    ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = cs.compute_embedded_points(
        axis=axis, axis_var_results=axis_vars, stations_m=[0.0, 1.0], twist_deg=0.0, negate_x=True
    )
    print("embedded shape:", P_mm.shape)
