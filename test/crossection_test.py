# test_cross_section.py
# This script allows loading a sectiondata.json file (or in-memory string),
# creates a CrossSection object, attaches the JSON payload,
# and runs tests on compute_local_points with single and multiple stations.

from __future__ import annotations
import ast
import json
import os
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque

# scalar (math) and vector (numpy) function maps
_SCALAR_FUNCS = {
    'COS': math.cos, 'SIN': math.sin, 'TAN': math.tan,
    'ACOS': math.acos, 'ASIN': math.asin, 'ATAN': math.atan,
    'cos': math.cos, 'sin': math.sin, 'tan': math.tan,
    'acos': math.acos, 'asin': math.asin, 'atan': math.atan,
    'LOG': math.log, 'EXP': math.exp, 'SQRT': math.sqrt, 'ABS': abs,
    'log': math.log, 'exp': math.exp, 'sqrt': math.sqrt, 'abs': abs,
    'PI': math.pi, 'Pi': math.pi, 'pi': math.pi,
}
_VECTOR_FUNCS = {
    'COS': np.cos, 'SIN': np.sin, 'TAN': np.tan,
    'ACOS': np.arccos, 'ASIN': np.arcsin, 'ATAN': np.arctan,
    'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
    'acos': np.arccos, 'asin': np.arcsin, 'atan': np.arctan,
    'LOG': np.log, 'EXP': np.exp, 'SQRT': np.sqrt, 'ABS': np.abs,
    'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
    'PI': np.pi, 'Pi': np.pi, 'pi': np.pi, "deg2rad": np.deg2rad, "rad2deg": np.rad2deg,
}

# prevent variable names from shadowing function names
_RESERVED_FUNC_NAMES = set(_SCALAR_FUNCS.keys()) | set(_VECTOR_FUNCS.keys())

_ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd, ast.Call
)

def _clean_expr(expr: str) -> str:
    s = str(expr).strip()
    # 1) allow a leading unary '+'
    if s.startswith('+'):
        s = s[1:].lstrip()
    # 2) caret → python power (if you ever see ^)
    s = s.replace('^', '**')
    return s

_ALLOWED_FUNC_NAMES = set(_SCALAR_FUNCS.keys()) | set(_VECTOR_FUNCS.keys())

def _compile_expr(expr_text: str):
    expr_text = _clean_expr(expr_text)
    node = ast.parse(str(expr_text), mode='eval')
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_AST):
            raise ValueError(f"Disallowed expression: {expr_text}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _ALLOWED_FUNC_NAMES:
                raise ValueError(f"Disallowed function: {getattr(n.func, 'id', '?')}")
    return compile(node, "<expr>", "eval")

def _sanitize_vars(variables: dict) -> dict:
    # drop keys that would shadow functions/constants
    return {k: v for k, v in (variables or {}).items() if k not in _RESERVED_FUNC_NAMES}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more output

# Helper functions
def _deep_find_first(obj, key_names):
    key_names = {k.lower() for k in key_names}
    dq = deque([([], obj)])
    while dq:
        path, cur = dq.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() in key_names and isinstance(v, list) and v:
                    return v, ".".join(path + [k])
                dq.append((path + [k], v))
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                dq.append((path + [f"[{i}]"], v))
    return None, None

def _as_point_id_list(seq):
    out = []
    for p in (seq or []):
        if isinstance(p, dict):
            for k in ("Id","id","PointName","Name","PointId","PointID","Point"):
                if k in p and p[k] not in (None, ""):
                    out.append(str(p[k]))
                    break
        else:
            out.append(str(p))
    return [s for s in out if s]

def _normalize_loops_value(value):
    loops = []
    if not isinstance(value, list):
        return loops
    if value and all(isinstance(x, dict) for x in value):
        for item in value:
            pts = item.get("Points") or item.get("Point") or item.get("Vertices") or item.get("Ids")
            if pts:
                loops.append(_as_point_id_list(pts))
            elif isinstance(item.get("Loop"), list):
                loops.append(_as_point_id_list(item["Loop"]))
            elif isinstance(item.get("Ring"), list):
                loops.append(_as_point_id_list(item["Ring"]))
        if loops:
            return loops
    if value and all(isinstance(x, list) for x in value):
        for seq in value:
            loops.append(_as_point_id_list(seq))
        return loops
    if value and all(isinstance(x, dict) for x in value):
        for item in value:
            pts = item.get("Points") or item.get("Point") or item.get("Vertices") or item.get("Ids")
            if pts:
                loops.append(_as_point_id_list(pts))
    return loops

def _points_to_table(value: Any) -> Dict[str, Dict[str, Any]]:
    out = {}
    if isinstance(value, dict):
        for pid, row in value.items():
            out[str(pid)] = row if isinstance(row, dict) else {"Raw": row}
        return out
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            pid = (row.get("Id") or row.get("id") or row.get("Name") or
                   row.get("PointName") or row.get("PointId") or row.get("PointID"))
            if pid is not None:
                out[str(pid)] = row
    return out

# CrossSection class
@dataclass(kw_only=True)
class CrossSection:
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
    points: Union[List[Dict[str, Any]], Dict[str, Any], None] = None
    variables: Union[List[Dict[str, Any]], Dict[str, Any], None] = None
    sofi_code: str | None = None
    cross_section_types: list[str] | None = None
    axis_variables: list | None = None

    json_data: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    json_source: Optional[str] = field(default=None, init=False, repr=False)

    _loops_cache: Dict[Tuple[str, ...], List[np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _dag_cache_key: Optional[str] = field(default=None, init=False, repr=False)
    _dag_cache_val: Optional[Tuple[List[str], Dict[str, dict]]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            if self.ncs is not None and str(self.ncs).strip() != "":
                self.ncs = int(self.ncs)
        except Exception:
            logger.debug("CrossSection(%r): failed to int(ncs=%r)", self.name, self.ncs)

        for attr in ("material1", "material2", "material_reinf"):
            v = getattr(self, attr)
            try:
                if v is not None and str(v).strip() != "":
                    setattr(self, attr, int(v))
            except Exception:
                logger.debug("CrossSection(%r): failed to int(%s=%r)", self.name, attr, v)

    def get_defaults(self) -> Dict[str, float]:
        if isinstance(self.json_data, dict):
            vars_dict = self.json_data.get("Variables")
            if isinstance(vars_dict, dict):
                out = {}
                for k, v in vars_dict.items():
                    try:
                        out[str(k)] = float(v)
                    except Exception:
                        pass
                if out:
                    return out

        out = {}
        raw = self.variables or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
        else:
            for row in raw or []:
                if not isinstance(row, dict):
                    continue
                try:
                    n = str(row.get("VariableName"))
                    out[n] = float(row.get("VariableValue", 0.0) or 0.0)
                except Exception:
                    pass
        return out

    def attach_json_payload(self, payload: Dict[str, Any], source: Optional[str] = None) -> None:
        if not isinstance(payload, dict):
            return
        self.json_data = payload
        self.json_source = source
        logger.debug(
            "CrossSection.attach_json_payload: name=%r ncs=%r source=%r",
            self.name, self.ncs, source
        )

    def _raw_points_and_loops(self) -> Tuple[Dict[str, Dict[str, Any]], List[List[str]]]:
        root = self.json_data if isinstance(self.json_data, dict) else (self.points or {})
        pts_val, _ = _deep_find_first(root, ["Points", "points"])
        lps_val, _ = _deep_find_first(root, ["Loops", "Contours", "Polygons", "Rings", "Boundaries"])

        points_table = _points_to_table(pts_val or {})
        loops_ids = _normalize_loops_value(lps_val or [])

        return points_table, loops_ids

    def has_geometry(self) -> bool:
        points_table, loops_ids = self._raw_points_and_loops()
        return bool(points_table) and bool(loops_ids)

    def geometry_counts(self) -> Tuple[int, int]:
        points_table, loops_ids = self._raw_points_and_loops()
        return len(points_table), len(loops_ids)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "CrossSection":
        kw = {
            "no": row.get("No"),
            "class_name": row.get("Class") or row.get("class_name"),
            "type": row.get("Type"),
            "description": row.get("Description"),
            "name": row.get("Name"),
            "inactive": row.get("InActive"),
            "ncs": row.get("NCS") or row.get("Ncs") or row.get("ncs"),
            "material1": row.get("Material1"),
            "material2": row.get("Material2"),
            "material_reinf": row.get("Material_Reinf"),
            "json_name": row.get("JSON_name") or row.get("JsonName") or row.get("json_name"),
            "sofi_code": row.get("SofiCode") or row.get("SOFiCode") or row.get("SOFiSTiKCustomCode"),
            "points": row.get("Points"),
            "variables": row.get("Variables"),
            "cross_section_types": row.get("cross_section_types") or row.get("CrossSection_Types"),
            "axis_variables": row.get("axis_variables") or row.get("AxisVariables"),
        }
        return cls(**{k: v for k, v in kw.items() if v is not None})

    @staticmethod
    def _pick_expr(expr_val, numeric_val):
        txt = str(expr_val) if expr_val is not None else ""
        if not txt or txt.strip().lower().startswith("error"):
            if numeric_val is None or str(numeric_val).strip() == "":
                return "0"
            return str(numeric_val)
        return txt

    def _normalize_point_row(self, p: dict) -> dict:
        if "Id" in p and ("Coord" in p or "coord" in p):
            pid = str(p.get("Id"))
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            ref = p.get("Reference") or p.get("reference") or []
            return {"Id": pid, "Coord": [str(coord[0]), str(coord[1])], "Reference": list(ref)}

        pid = str(p.get("PointName") or p.get("Id") or p.get("id") or "")
        y_expr = self._pick_expr(p.get("CoorY"), p.get("CoorYVal"))
        z_expr = self._pick_expr(p.get("CoorZ"), p.get("CoorZVal"))
        ref = p.get("Reference") or p.get("reference") or []
        return {"Id": pid, "Coord": [y_expr, z_expr], "Reference": list(ref)}

    def _safe_points_list(self):
        pts = getattr(self, "points", None)
        if isinstance(pts, list) and pts:
            return pts
        jd = getattr(self, "json_data", None)
        if isinstance(jd, dict):
            pv = jd.get("Points") or jd.get("points")
            if isinstance(pv, list) and pv:
                return pv
        return []

    def _safe_loops_list(self):
        loops_attr = getattr(self, "loops", None)
        if isinstance(loops_attr, list) and loops_attr:
            return loops_attr

        jd = getattr(self, "json_data", None)
        if jd is None:
            return []

        candidates = ("Loops","Loop","Contours","Polygons","Rings","Boundaries","LoopPoints","LoopList")
        raw_loops, picked_path = _deep_find_first(jd, candidates)
        if not raw_loops:
            keys = list(jd.keys()) if isinstance(jd, dict) else type(jd).__name__
            logger.info(f"[loops] No loops found for {getattr(self,'name','?')} in json_data. Top-level keys: {keys}")
            return []

        loop_lists = _normalize_loops_value(raw_loops)
        if not loop_lists:
            logger.info(f"[loops] Found '{picked_path}' for {getattr(self,'name','?')}, but could not normalize.")
            return []

        out = []
        for seq in loop_lists:
            out.append({"Points": [{"Id": pid} for pid in seq]})
        logger.info(f"[loops] {getattr(self,'name','?')} -> picked '{picked_path}', loops={len(out)}")
        return out

    def _collect_all_points(self) -> List[dict]:
        out = []
        for p in self._safe_points_list():
            try:
                norm = self._normalize_point_row(p)
                if norm.get("Id"):
                    out.append(norm)
            except Exception:
                pass

        for lp in self._safe_loops_list():
            for p in (lp.get("Points") or []):
                try:
                    norm = self._normalize_point_row(p)
                    if norm.get("Id"):
                        out.append(norm)
                except Exception:
                        pass

        logger.info("Collected %d points in section %s", len(out), getattr(self, "name", "?"))
        logger.info(" Points: %s", [p.get("Id") for p in out])
        logger.info(" Loops: %d in section %s", len(self._safe_loops_list()), getattr(self, "name", "?"))

        seen = set()
        uniq = []
        for p in out:
            pid = p.get("Id")
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)
        return uniq

    def _dag_identity_tuple(self):
        ncs = int(getattr(self, "ncs", -1))
        pts = self._collect_all_points()
        pts_tup = tuple(sorted(
            (
                str(p.get("Id")),
                str((p.get("Coord") or ["", ""])[0]),
                str((p.get("Coord") or ["", ""])[1]),
                tuple(str(r) for r in (p.get("Reference") or [])),
            )
            for p in pts
        ))
        loops = self._safe_loops_list()
        loops_tup = tuple(sorted(
            (tuple(str(pp.get("Id")) for pp in (lp.get("Points") or [])) for lp in loops),
            key=lambda ids: (ids[0] if ids else "", len(ids))
        ))
        return (ncs, pts_tup, loops_tup)

    def _dag_key_for_identity(self, identity) -> str:
        import hashlib
        try:
            blob = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            blob = repr(identity)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

    def build_dag(self) -> Tuple[List[str], Dict[str, dict]]:
        identity = self._dag_identity_tuple()
        key = self._dag_key_for_identity(identity)

        cache_key = getattr(self, "_dag_cache_key", None)
        cache_val = getattr(self, "_dag_cache_val", None)
        if cache_key == key and cache_val is not None:
            return cache_val

        all_points = self._collect_all_points()
        by_id = {}
        deps = {}
        for p in all_points:
            pid = str(p.get("Id") or p.get("id"))
            by_id[pid] = p
            r = p.get("Reference") or p.get("reference") or []
            if isinstance(r, (list, tuple)):
                deps[pid] = set(str(x) for x in r if x is not None)
            else:
                deps[pid] = set()

        incoming = {k: set(v) for k, v in deps.items()}
        outgoing = {k: set() for k in deps}
        for k, vs in deps.items():
            for v in vs:
                if v in outgoing:
                    outgoing[v].add(k)
                else:
                    outgoing[v] = {k}

        order = []
        roots = [k for k, s in incoming.items() if not s]
        roots.sort()
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
            logger.warning("CrossSection DAG has unresolved dependencies or cycles: %s", remaining)
            order.extend(sorted(remaining))

        self._dag_cache_key = key
        self._dag_cache_val = (order, by_id)
        return order, by_id

    @staticmethod
    def _results_signature(stations_m, axis_var_rows):
        import hashlib
        blob = json.dumps({
            "stations": [float(s) for s in (stations_m or [])],
            "vars": axis_var_rows or [],
        }, sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def _build_var_arrays_from_results(results: List[Dict[str, float]],
                                       defaults: Dict[str, float],
                                       keep: Optional[set] = None) -> Dict[str, np.ndarray]:
        names = keep or set()
        if not names:
            for d in results or []:
                names.update(d.keys())
        out = {}
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
    def _fix_var_units_inplace(var_arrays: Dict[str, np.ndarray], defaults: Dict[str, float]) -> None:
        def looks_angle(name: str) -> bool:
            n = name.upper()
            return n.startswith("W_") or n.startswith("Q_") or n.startswith("INCL_")

        def looks_length(name: str) -> bool:
            n = name.upper()
            return n.startswith(("B_", "T_", "BEFF_", "EX"))

        for name, arr in list(var_arrays.items()):
            a = np.asarray(arr, float)
            if a.size == 0:
                continue

            def_v = defaults.get(name)
            scale = 1.0

            if def_v is not None and np.isfinite(def_v) and def_v != 0:
                med = float(np.nanmedian(np.abs(a))) or 0.0
                candidates = (1.0, 1000.0, 0.001)
                costs = [abs(med * s - def_v) / max(1.0, abs(def_v)) for s in candidates]
                scale = candidates[int(np.argmin(costs))]
            else:
                if looks_angle(name) and np.nanmedian(np.abs(a)) < 0.5:
                    scale = 1000.0
                if looks_length(name) and 0 < np.nanmedian(np.abs(a)) < 100:
                    scale = 1000.0

            if scale != 1.0:
                var_arrays[name] = a * scale

    @staticmethod
    def _collect_used_variable_names(section_json: dict) -> set:
        used = set()
        pts = (section_json or {}).get("Points") or section_json.get("points") or []
        for p in pts or []:
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            for expr in (coord[:2] or []):
                try:
                    txt = str(expr)
                except Exception:
                    continue
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

    @staticmethod
    def _compile_pair(expr_x: str, expr_y: str):
        return _compile_expr(str(expr_x)), _compile_expr(str(expr_y))

    @staticmethod
    def _euclid_vectorized(X: np.ndarray, Y: np.ndarray,
                           ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        if not ref_pts:
            return X, Y
        r = [p for p in ref_pts if p is not None]
        if len(r) == 1:
            px = float(r[0].get("x", 0.0) or r[0].get("X", 0.0))
            py = float(r[0].get("y", 0.0) or r[0].get("Y", 0.0))
            return X + px, Y + py
        if len(r) >= 2:
            p1, p2 = r[0], r[1]
            px = float(p1.get("x", 0.0) or p1.get("X", 0.0))
            py = float(p2.get("y", 0.0) or p2.get("Y", 0.0))
            return X + px, Y + py
        return X, Y

    @staticmethod
    def _polar_vectorized(R: np.ndarray, A_deg: np.ndarray,
                          ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        A = np.deg2rad(A_deg)
        x = R * _VECTOR_FUNCS["cos"](A)
        y = R * _VECTOR_FUNCS["sin"](A)
        return CrossSection._euclid_vectorized(x, y, ref_pts)

    @staticmethod
    def _cy_vectorized(ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        if not ref_pts:
            return np.zeros(1, float), np.zeros(1, float)
        p = ref_pts[0]
        px = float(p.get("x", 0.0) or p.get("X", 0.0))
        py = float(p.get("y", 0.0) or p.get("Y", 0.0))
        return np.asarray([px]), np.asarray([py])

    @staticmethod
    def _cz_vectorized(ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        return CrossSection._cy_vectorized(ref_pts)

    def eval_expressions_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
        stations_count: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:

        ids = list(order)
        if stations_count is not None:
            S = int(stations_count)
        else:
            S = len(next(iter(var_arrays.values()))) if var_arrays else 0

        N = len(ids)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        env_base = {**_VECTOR_FUNCS}
        for k, arr in var_arrays.items():
            env_base[k] = np.asarray(arr, float)

        for j, pid in enumerate(ids):
            pj = by_id.get(pid) or {}
            coord = pj.get("Coord") or pj.get("coord") or [0, 0]
            x_expr = str(coord[0]) if len(coord) > 0 else "0"
            y_expr = str(coord[1]) if len(coord) > 1 else "0"

            cx, cy = self._compile_pair(x_expr, y_expr)

            env = env_base.copy()

            try:
                x_val = eval(cx, {"__builtins__": {}}, env)
            except Exception as e:
                logger.error("Error in X for point '%s' in section '%s': %s", pid, self.name, e)
                x_val = np.full(S, np.nan, float)
            try:
                y_val = eval(cy, {"__builtins__": {}}, env)
            except Exception as e:
                logger.error("Error in Y for point '%s' in section '%s': %s", pid, self.name, e)
                y_val = np.full(S, np.nan, float)

            X[:, j] = np.asarray(x_val, float)
            Y[:, j] = np.asarray(y_val, float)

        id_to_col = {pid: idx for idx, pid in enumerate(ids)}
        for j, pid in enumerate(ids):
            refs = by_id.get(pid, {}).get("Reference") or []
            if not refs:
                continue
            if len(refs) == 1:
                k = id_to_col.get(str(refs[0]))
                if k is not None:
                    X[:, j] += X[:, k]
                    Y[:, j] += Y[:, k]
            else:
                kx = id_to_col.get(str(refs[0]))
                ky = id_to_col.get(str(refs[1]))
                if kx is not None: X[:, j] += X[:, kx]
                if ky is not None: Y[:, j] += Y[:, ky]

        if negate_x:
            X = -X

        return ids, X, Y

    def get_coordinates_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
        stations_count: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        return self.eval_expressions_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id,
            negate_x=negate_x, stations_count=stations_count
        )

    def _loops_idx(self, ids):
        loops_raw = self._safe_loops_list()
        if not loops_raw:
            return []

        id_to_col = {pid: j for j, pid in enumerate(ids)}

        def _pid_from_loop_point(p):
            if isinstance(p, dict):
                for k in ("Id","id","PointName","Name","PointId","PointID","Point"):
                    v = p.get(k)
                    if v is not None and str(v).strip():
                        return str(v)
            elif p is not None:
                return str(p)
            return None

        loops = []
        for lp in loops_raw:
            pts = lp.get("Points") or []
            idxs = []
            for p in pts:
                pid = _pid_from_loop_point(p)
                if pid is None:
                    continue
                j = id_to_col.get(pid)
                if j is not None:
                    idxs.append(j)
            if len(idxs) >= 2:
                loops.append(np.asarray(idxs, dtype=int))
        return loops

    def compute_local_points(
        self, *, axis_var_results: List[Dict[str, float]], negate_x: bool = True
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
        pts_all = self._collect_all_points()
        section_json = {"Points": pts_all}
        used_names = self._collect_used_variable_names(section_json)
        defaults = self.get_defaults()

        var_arrays = self._build_var_arrays_from_results(axis_var_results, defaults, keep=used_names)
        self._fix_var_units_inplace(var_arrays, defaults)

        order, by_id = self.build_dag()

        ids, X_mm, Y_mm = self.get_coordinates_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id, negate_x=negate_x, stations_count=len(axis_var_results)
        )

        loops_idx = self._loops_idx(ids)

        S, N = X_mm.shape if X_mm.size else (len(axis_var_results), 0)
        logger.info("Section %s (ncs=%s): stations=%d, points=%d",
                    getattr(self, "name", "?"), getattr(self, "ncs", "?"), S, N)

        if "UL" in ids and S > 0:
            j = ids.index("UL")
            logger.info("UL @ first station (Y,Z mm): %.3f, %.3f", X_mm[0, j], Y_mm[0, j])

        nan_xy = int(np.isnan(X_mm).sum() + np.isnan(Y_mm).sum())
        if nan_xy:
            logger.warning("NaNs in section %s: %d total", getattr(self, "name", "?"), nan_xy)

        return ids, X_mm, Y_mm, loops_idx

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
                logger.debug("Scalar eval error %r: %s", string_equation, e)
                return float("nan")

        def get_coordinates(self, coords):
            rt = (self.reference_type or '').lower()
            if rt in ("c", "carthesian", "e", "euclidean"): return self._euclid(coords)
            if rt in ("p", "polar"):                         return self._polar(coords)
            if rt in ("constructionaly", "cy"):              return self._cy()
            if rt in ("constructionalz", "cz"):              return self._cz()
            return self._euclid(coords)

        def _euclid(self, coords):
            x = self.eval_equation(coords[0]); y = self.eval_equation(coords[1])
            return {'coords': {'x': x, 'y': y}, 'guides': None}

        def _polar(self, coords):
            r = self.eval_equation(coords[0])
            a = math.radians(self.eval_equation(coords[1]))
            return {'coords': {'x': r*math.cos(a), 'y': r*math.sin(a)}, 'guides': None}

        def _cy(self):
            return {'coords': {'x': 0.0, 'y': 0.0}, 'guides': None}

        def _cz(self):
            return {'coords': {'x': 0.0, 'y': 0.0}, 'guides': None}

    def compute_local_points_scalar(self, env_vars: Dict[str, float]) -> List[Dict[str, float]]:
        pts = self._collect_all_points()
        out = []
        for p in pts:
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            rf = CrossSection.ReferenceFrame(reference_type='euclidean', reference=p.get("Reference"), points=out, variables=env_vars)
            xy = rf.get_coordinates(coord)["coords"]
            out.append({"id": p.get("Id") or p.get("id"), "x": float(xy["x"]), "y": float(xy["y"])})
        return out

# Function to load sectiondata.json from file or string
def load_section_data(json_path_or_str: str) -> Dict[str, Any]:
    if os.path.isfile(json_path_or_str):
        with open(json_path_or_str, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Assume it's a JSON string
        return json.loads(json_path_or_str)

# Example usage and tests
if __name__ == "__main__":
    # Provide your sectiondata.json path or content here
    # Example: json_source = 'path/to/sectiondata.json'
    # Or inline JSON string
    
    json_data = {
   "Name":"section_0100.cdb",
   "Id":100,
   "Unit":"mm",
   "MaterialId":1,
   "Points":
   [
      {"Id":"O", "Coord":["0", "0"], "PointType":"StressPoint"},
      {"Id":"UM", "Coord":["0", "H_QS"], "ReferenceType":"Euclidean", "PointType":"ConstructionPoint"},
      {"Id":"UL", "Coord":["-37+B_TR/2-(1/TAN((W_ST)*(Pi/180))*(H_QS-T_FBA/COS((Q_NG/100)*(Pi/180))-B_TR/2*Q_NG/100))", "H_QS"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"UR", "Coord":["-37-B_TR/2+(1/TAN((W_ST)*(Pi/180))*(H_QS-T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100))", "H_QS"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"AKL", "Coord":["B_TR/2-37", "+T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"AKR", "Coord":["-B_TR/2-37", "+T_FBA/COS((Q_NG/100)*(Pi/180))-B_TR/2*Q_NG/100"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"OL", "Coord":["(B_FB/2)+EX", "((B_FB/2+EX-1900)*Q_NG/100)-1900*0.025"], "ReferenceType":"Euclidean", "PointType":"ConstructionPoint"},
      {"Id":"OR", "Coord":["(-B_FB/2)+EX-75", "(-B_FB/2+EX)*Q_NG/100"], "ReferenceType":"Euclidean", "PointType":"ConstructionPoint"},
      {"Id":"1L", "Coord":["-299.9999523162842", "100.00002384185791"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"2L", "Coord":["-400.00009536743164", "400.00003576278687"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"3L", "Coord":["-500", "699.999988079071"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"4L", "Coord":["-600.0001430511475", "1100.0000834465027"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
      {"Id":"5L", "Coord":["-700.0000476837158", "1400.0000357627869"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"}
   ],
   "Loops":
   [
      {
         "Points":
         [
            {"Id":"100", "Coord":["0", "0"], "Reference":["UM"], "ReferenceType":"Euclidean"},
            {"Id":"115", "Coord":["0", "0"], "Reference":["UR"], "ReferenceType":"Euclidean"},
            {"Id":"114", "Coord":["0", "0"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
            {"Id":"113", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
            {"Id":"112", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-T_BA*TAN((90-W_ST)*(Pi/180))", "-T_BA"], "Reference":["UR"], "ReferenceType":"Euclidean"},
            {"Id":"111", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-T_BA*TAN((90-W_ST)*(Pi/180))+B_BV", "-T_BM"], "Reference":["UR"], "ReferenceType":"Euclidean"},
            {"Id":"106", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+T_BA*TAN((90-W_ST)*(Pi/180))-B_BV", "-T_BM"], "Reference":["UL"], "ReferenceType":"Euclidean"},
            {"Id":"105", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+T_BA*TAN((90-W_ST)*(Pi/180))", "-T_BA"], "Reference":["UL"], "ReferenceType":"Euclidean"},
            {"Id":"104", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(-T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
            {"Id":"103", "Coord":["0", "0"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
            {"Id":"102", "Coord":["0", "0"], "Reference":["UL"], "ReferenceType":"Euclidean"}
         ],
         "Id":"L1",
         "MaterialId":"1"
      },
      {
         "Points":
         [
            {"Id":"120", "Coord":["0", "0"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
            {"Id":"136", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(-T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
            {"Id":"135", "Coord":["B_FBV/2", "(T_FBM/COS((Q_NG/100)*(Pi/180)))+B_FBV/2*Q_NG/100"], "ReferenceType":"Euclidean"},
            {"Id":"134", "Coord":["-B_FBV/2", "(T_FBM/COS((Q_NG/100)*(Pi/180)))-B_FBV/2*Q_NG/100"], "ReferenceType":"Euclidean"},
            {"Id":"132", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
            {"Id":"131", "Coord":["0", "0"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
            {"Id":"130", "Coord":["B_KK+1", "T_KP+B_KK*Q_NG/100"], "Reference":["OR"], "ReferenceType":"Euclidean"},
            {"Id":"128", "Coord":["0", "T_KP"], "Reference":["OR"], "ReferenceType":"Euclidean"},
            {"Id":"127", "Coord":["0", "0"], "Reference":["OR"], "ReferenceType":"Euclidean"},
            {"Id":"126", "Coord":["0", "0"]},
            {"Id":"125", "Coord":["-1900.0000953674316", "47.49999940395355"], "Reference":["OL"], "ReferenceType":"Euclidean"},
            {"Id":"124", "Coord":["0", "0"], "Reference":["OL"], "ReferenceType":"Euclidean"},
            {"Id":"123", "Coord":["0", "T_KP"], "Reference":["OL"], "ReferenceType":"Euclidean"},
            {"Id":"122", "Coord":["-B_KK-1", "T_KP+B_KK*0.025"], "Reference":["OL"], "ReferenceType":"Euclidean"}
         ],
         "Id":"L2",
         "MaterialId":"2"
      }
   ],
   "PointReinforcements":
   [
      {
         "Point": {"Id":"PRF150", "Coord":["-82.49998092651367", "-82.49998092651367"], "Reference":["UL"], "ReferenceType":"Euclidean"},
         "Id":"150",
         "Layer":"7",
         "MaterialId":"15",
         "TorsionalContribution":"Acti",
         "Diameter":"12"
      },
      {
         "Point": {"Id":"PRF151", "Coord":["82.49998092651367", "-82.49998092651367"], "Reference":["UR"], "ReferenceType":"Euclidean"},
         "Id":"151",
         "Layer":"7",
         "MaterialId":"15",
         "TorsionalContribution":"Acti",
         "Diameter":"12"
      },
      {
         "Point": {"Id":"PRF152", "Coord":["51.49984359741211", "-517.500028014183"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
         "Id":"152",
         "Layer":"7",
         "MaterialId":"15",
         "TorsionalContribution":"Acti",
         "Diameter":"12"
      },
      {
         "Point": {"Id":"PRF153", "Coord":["-51.49984359741211", "-517.5000242888927"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
         "Id":"153",
         "Layer":"7",
         "MaterialId":"15",
         "TorsionalContribution":"Acti",
         "Diameter":"12"
      }
   ],
   "LineReinforcements":
   [
      {
         "BarDistribution":"Even",
         "Spacing":"150",
         "BarCount":"0",
         "PointStart": {"Id":"LRF100Start", "Coord":["-174.99995231628418", "-174.99995231628418"], "Reference":["UL"], "ReferenceType":"Euclidean"},
         "PointEnd": {"Id":"LRF100End", "Coord":["174.99995231628418", "-174.99995231628418"], "Reference":["UR"], "ReferenceType":"Euclidean"},
         "Id":"100",
         "Layer":"1",
         "MaterialId":"10",
         "TorsionalContribution":"Pass",
         "Diameter":"25"
      },
      {
         "BarDistribution":"Even",
         "Spacing":"150",
         "BarCount":"0",
         "PointStart": {"Id":"LRF300Start", "Coord":["-250", "-174.99995231628418"], "Reference":["UL"], "ReferenceType":"Euclidean"},
         "PointEnd": {"Id":"LRF300End", "Coord":["-250", "19.999980926513672"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
         "Id":"300",
         "Layer":"3",
         "MaterialId":"10",
         "TorsionalContribution":"Pass",
         "Diameter":"20"
      },
      {
         "BarDistribution":"Even",
         "Spacing":"150",
         "BarCount":"0",
         "PointStart": {"Id":"LRF310Start", "Coord":["250", "-174.99995231628418"], "Reference":["UR"], "ReferenceType":"Euclidean"},
         "PointEnd": {"Id":"LRF310End", "Coord":["250", "20.00001072883606"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
         "Id":"310",
         "Layer":"4",
         "MaterialId":"10",
         "TorsionalContribution":"Pass",
         "Diameter":"20"
      },
      {
         "BarDistribution":"Even",
         "Spacing":"150",
         "BarCount":"0",
         "PointStart": {"Id":"LRF200Start", "Coord":["-82.49998092651367", "125"], "Reference":["OL"], "ReferenceType":"Euclidean"},
         "PointEnd": {"Id":"LRF200End", "Coord":["-1724.9999046325684", "225.0000238418579"], "Reference":["OL"], "ReferenceType":"Euclidean"},
         "Id":"200",
         "Layer":"2",
         "MaterialId":"11",
         "TorsionalContribution":"Pass",
         "Diameter":"20"
      },
      {
         "BarDistribution":"Even",
         "Spacing":"150",
         "BarCount":"0",
         "PointStart": {"Id":"LRF250Start", "Coord":["-1724.9999046325684", "225.0000238418579"], "Reference":["OL"], "ReferenceType":"Euclidean"},
         "PointEnd": {"Id":"LRF250End", "Coord":["82.49998092651367", "125"], "Reference":["OR"], "ReferenceType":"Euclidean"},
         "Id":"250",
         "Layer":"2",
         "MaterialId":"11",
         "TorsionalContribution":"Pass",
         "Diameter":"20"
      }
   ],
   "NonEffectiveZones":[{
      "Id":"BP1",
      "Type":"ZV",
      "PointStart": {"Id":"NEFFBP1Min", "Coord":["-BEFF_BP/2", "-1000"], "Reference":["UM"], "ReferenceType":"Euclidean"},
      "PointEnd": {"Id":"NEFFBP1Max", "Coord":["BEFF_BP/2", "1000"], "Reference":["UM"], "ReferenceType":"Euclidean"}
   },
   {
      "Id":"FPL",
      "Type":"ZV",
      "PointStart": {"Id":"NEFFFPLMin", "Coord":["-BEFF_KR", "-1000.0000596046448"], "Reference":["OL"], "ReferenceType":"Euclidean"},
      "PointEnd": {"Id":"NEFFFPLMax", "Coord":["BEFF_KR", "1000.0000596046448"], "Reference":["OL"], "ReferenceType":"Euclidean"}
   },
   {
      "Id":"FPR",
      "Type":"ZV",
      "PointStart": {"Id":"NEFFFPRMin", "Coord":["-BEFF_KR", "1000.000074505806"], "Reference":["OR"], "ReferenceType":"Euclidean"},
      "PointEnd": {"Id":"NEFFFPRMax", "Coord":["BEFF_KR", "-1000.0001043081284"], "Reference":["OR"], "ReferenceType":"Euclidean"}
   },
   {
      "Id":"FPM",
      "Type":"ZV",
      "PointStart": {"Id":"NEFFFPMMin", "Coord":["-BEFF_FB/2", "-700.0000476837158"], "Reference":["O"], "ReferenceType":"Euclidean"},
      "PointEnd": {"Id":"NEFFFPMMax", "Coord":["BEFF_FB/2", "700.0000476837158"], "Reference":["O"], "ReferenceType":"Euclidean"}
   }],
   "Variables": {"H_QS":2750.000238418579, "B_TR":8575.000762939453, "W_ST":75.0, "T_FBA":600.0000238418579, "Q_NG":3.0, "B_FB":15900.00057220459, "EX":0.0, "T_ST":500.0, "T_FBI":600.0000238418579, "T_BA":600.0000238418579, "B_BV":2000.0, "T_BM":350.0000238418579, "B_KK":500.0, "T_KP":250.0, "B_FBV":3000.000238418579, "T_FBM":350.0000238418579, "BEFF_BP":100.00000149011612, "BEFF_KR":2500.0, "BEFF_FB":100.00000149011612},
   "VariableDescriptions":
   {
      "H_QS":"H_QS:Axis Variable",
      "B_TR":"B_TR:Axis Variable",
      "W_ST":"W_ST:Axis Variable",
      "T_FBA":"T_FBA:Axis Variable",
      "Q_NG":"Q_NG:Axis Variable",
      "B_FB":"B_FB:Axis Variable",
      "EX":"EX:Axis Variable",
      "T_ST":"T_ST:Axis Variable",
      "T_FBI":"T_FBI:Axis Variable",
      "T_BA":"T_BA:Axis Variable",
      "B_BV":"B_BV:Axis Variable",
      "T_BM":"T_BM:Axis Variable",
      "B_KK":"B_KK:Axis Variable",
      "T_KP":"T_KP:Axis Variable",
      "B_FBV":"B_FBV:Axis Variable",
      "T_FBM":"T_FBM:Axis Variable",
      "BEFF_BP":"BEFF_BP:Axis Variable",
      "BEFF_KR":"BEFF_KR:Axis Variable",
      "BEFF_FB":"BEFF_FB:Axis Variable"
   }
}
    raw_crosssection = {'No': '1', 'Class': 'CrossSection', 'Type': 'Deck', 'Description': 'test', 'Name': 'MASTER_Deck', 'InActive': '', 'NCS': 111, 'Material1': 101, 'Material2': 0, 'Material_Reinf': 200, 'JSON_name': ['MASTER_SECTION\\SectionData.json'], 'SofiCode': [], 'Points': [{'PointName': 'O', 'CoorY': '0', 'CoorZ': '0', 'CoorYVal': '0', 'CoorZVal': '0'}, {'PointName': 'UM', 'CoorY': '0', 'CoorZ': 'H_QS', 'CoorYVal': '0', 'CoorZVal': 2750.00023841857}, {'PointName': 'UL', 'CoorY': '-37+B_TR/2-(1/TAN((W_ST)*(Pi/180))*(H_QS-T_FBA/COS((Q_NG/100)*(Pi/180))-B_TR/2*Q_NG/100))', 'CoorZ': 'H_QS', 'CoorYVal': 'Error 2015', 'CoorZVal': 2750.00023841857}, {'PointName': 'UR', 'CoorY': '-37-B_TR/2+(1/TAN((W_ST)*(Pi/180))*(H_QS-T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100))', 'CoorZ': 'H_QS', 'CoorYVal': 'Error 2015', 'CoorZVal': 2750.00023841857}, {'PointName': 'AKL', 'CoorY': 'B_TR/2-37', 'CoorZ': '+T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100', 'CoorYVal': 4250.50038146972, 'CoorZVal': 'Error 2015'}, {'PointName': 'AKR', 'CoorY': '-B_TR/2-37', 'CoorZ': '+T_FBA/COS((Q_NG/100)*(Pi/180))-B_TR/2*Q_NG/100', 'CoorYVal': '-4324.50038146972', 'CoorZVal': 'Error 2015'}, {'PointName': 'OL', 'CoorY': '(B_FB/2)+EX', 'CoorZ': '((B_FB/2+EX-1900)*Q_NG/100)-1900*0.025', 'CoorYVal': 7950.00028610225, 'CoorZVal': 134.000008583068}, {'PointName': 'OR', 'CoorY': '(-B_FB/2)+EX-75', 'CoorZ': '(-B_FB/2+EX)*Q_NG/100', 'CoorYVal': '-8025.00028610225', 'CoorZVal': '-238.500008583068'}, {'PointName': '1L', 'CoorY': '-299.999952316284', 'CoorZ': 100.000023841857, 'CoorYVal': '-299.999952316284', 'CoorZVal': 100.000023841857}, {'PointName': '2L', 'CoorY': '-400.000095367431', 'CoorZ': 400.000035762786, 'CoorYVal': '-400.000095367431', 'CoorZVal': 400.000035762786}, {'PointName': '3L', 'CoorY': '-500', 'CoorZ': 699.999988079071, 'CoorYVal': '-500', 'CoorZVal': 699.999988079071}, {'PointName': '4L', 'CoorY': '-600.000143051147', 'CoorZ': '1100.0000834465', 'CoorYVal': '-600.000143051147', 'CoorZVal': '1100.0000834465'}, {'PointName': '5L', 'CoorY': '-700.000047683715', 'CoorZ': 1400.00003576278, 'CoorYVal': '-700.000047683715', 'CoorZVal': 1400.00003576278}], 'Variables': [{'VariableName': 'TEST', 'VariableValue': 2750.00023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'H_QS:Axis Variable'}, {'VariableName': 'H_QS', 'VariableValue': 2750.00023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'H_QS:Axis Variable'}, {'VariableName': 'B_TR', 'VariableValue': 8575.00076293945, 'VariableUnit': '[mm]', 'VariableDescription': 'B_TR:Axis Variable'}, {'VariableName': 'W_ST', 'VariableValue': '75', 'VariableUnit': '[mm]', 'VariableDescription': 'W_ST:Axis Variable'}, {'VariableName': 'T_FBA', 'VariableValue': 600.000023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'T_FBA:Axis Variable'}, {'VariableName': 'Q_NG', 'VariableValue': '3', 'VariableUnit': '[mm]', 'VariableDescription': 'Q_NG:Axis Variable'}, {'VariableName': 'B_FB', 'VariableValue': 15900.0005722045, 'VariableUnit': '[mm]', 'VariableDescription': 'B_FB:Axis Variable'}, {'VariableName': 'EX', 'VariableValue': '0', 'VariableUnit': '[mm]', 'VariableDescription': 'EX:Axis Variable'}, {'VariableName': 'T_ST', 'VariableValue': '500', 'VariableUnit': '[mm]', 'VariableDescription': 'T_ST:Axis Variable'}, {'VariableName': 'T_FBI', 'VariableValue': 600.000023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'T_FBI:Axis Variable'}, {'VariableName': 'T_BA', 'VariableValue': 600.000023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'T_BA:Axis Variable'}, {'VariableName': 'B_BV', 'VariableValue': '2000', 'VariableUnit': '[mm]', 'VariableDescription': 'B_BV:Axis Variable'}, {'VariableName': 'T_BM', 'VariableValue': 350.000023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'T_BM:Axis Variable'}, {'VariableName': 'B_KK', 'VariableValue': '500', 'VariableUnit': '[mm]', 'VariableDescription': 'B_KK:Axis Variable'}, {'VariableName': 'T_KP', 'VariableValue': '250', 'VariableUnit': '[mm]', 'VariableDescription': 'T_KP:Axis Variable'}, {'VariableName': 'B_FBV', 'VariableValue': 3000.00023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'B_FBV:Axis Variable'}, {'VariableName': 'T_FBM', 'VariableValue': 350.000023841857, 'VariableUnit': '[mm]', 'VariableDescription': 'T_FBM:Axis Variable'}, {'VariableName': 'BEFF_BP', 'VariableValue': 100.000001490116, 'VariableUnit': '[mm]', 'VariableDescription': 'BEFF_BP:Axis Variable'}, {'VariableName': 'BEFF_KR', 'VariableValue': '2500', 'VariableUnit': '[mm]', 'VariableDescription': 'BEFF_KR:Axis Variable'}, {'VariableName': 'BEFF_FB', 'VariableValue': 100.000001490116, 'VariableUnit': '[mm]', 'VariableDescription': 'BEFF_FB:Axis Variable'}]}

    cs = CrossSection.from_row(raw_crosssection)
    cs.attach_json_payload(json_data, source='test_source')

    # Test 1: Single station with default variables
    axis_var_results = [cs.get_defaults()]
    ids, X_mm, Y_mm, loops_idx = cs.compute_local_points(axis_var_results=axis_var_results)