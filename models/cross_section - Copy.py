from __future__ import annotations
# models/cross_section.py
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import logging

# Reuse your safe-eval & math maps
from utils import (
    _compile_expr,
    _sanitize_vars,
    _SCALAR_FUNCS,
    _VECTOR_FUNCS,
    _RESERVED_FUNC_NAMES,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CrossSection:
    """
    CrossSection now *owns* all 2D point evaluation:

    - Variable defaults
    - DAG build & topological sort
    - Vectorized expression evaluation
    - (Deprecated) scalar reference frame paths
    - Vectorized local frame transforms (C/Euclid, P/Polar, CY, CZ)
    - Public API:
        get_defaults()
        build_dag()
        eval_expressions_vectorized(...)
        get_coordinates_vectorized(...)
        compute_local_points(...)              -> (ids, X_mm, Y_mm, loops_idx)
        compute_local_points_scalar(...)       -> legacy single-slice path
        compute_embedded_points(...)           -> (ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx)
    """

    # ---- existing mapped fields (keep names to avoid breaking mapping/from_dict) ----
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

    # cached parse (by object identity)
    _dag_cache_key: int = field(default=0, init=False, repr=False)
    _dag_order: List[str] = field(default_factory=list, init=False, repr=False)
    _dag_by_id: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _loops_cache: Dict[Tuple[str, ...], List[np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    # -------------------------------------------------------------------------
    # Defaults / variables
    # -------------------------------------------------------------------------

    def get_defaults(self) -> Dict[str, float]:
        """
        Read "Variables" from the section JSON (supports dict or list-of-dicts forms).
        Values are returned as floats (best-effort), untouched units.
        """
        defaults: Dict[str, float] = {}
        raw = (self.variables or {}) or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    defaults[str(k)] = float(v)
                except Exception:
                    pass
        else:
            # list-of-dicts [{VariableName, VariableValue}, ...]
            for row in raw or []:
                try:
                    n = str(row.get("VariableName"))
                    defaults[n] = float(row.get("VariableValue", 0.0) or 0.0)
                except Exception:
                    pass
        return defaults

    # -------------------------------------------------------------------------
    # DAG build (topological sort for point dependencies)
    # -------------------------------------------------------------------------

    def _collect_all_points(self) -> List[dict]:
        pts = self.points or []
        if isinstance(pts, dict):
            # tolerate {"Points":[...] } or similar
            pts = pts.get("Points") or pts.get("points") or []
        out = []
        for p in pts or []:
            try:
                pid = str(p.get("Id") or p.get("id"))
                if not pid:
                    continue
                out.append(p)
            except Exception:
                pass
        return out

    @lru_cache(maxsize=1024)
    def _dag_key_for_identity(self, key: int) -> int:
        # thin wrapper to make lru_cache happy with identity int
        return key

    def build_dag(self) -> Tuple[List[str], Dict[str, dict]]:
        """
        Build topological order of points based on "Reference" dependencies.
        Caches by object identity (id(self.points)).
        """
        base_key = id(self.points)
        self._dag_cache_key = self._dag_key_for_identity(base_key)

        all_points = self._collect_all_points()
        by_id = {}
        deps: Dict[str, set] = {}
        for p in all_points:
            pid = str(p.get("Id") or p.get("id"))
            by_id[pid] = p
            r = p.get("Reference") or p.get("reference") or []
            if isinstance(r, (list, tuple)):
                deps[pid] = set(str(x) for x in r if x is not None)
            else:
                deps[pid] = set()

        # Kahn topological sort
        incoming = {k: set(v) for k, v in deps.items()}
        outgoing: Dict[str, set] = {k: set() for k in deps}
        for k, vs in deps.items():
            for v in vs:
                if v in outgoing:
                    outgoing[v].add(k)
                else:
                    outgoing[v] = {k}

        order: List[str] = []
        roots = [k for k, s in incoming.items() if not s]
        roots.sort()
        from collections import deque
        q = deque(roots)
        while q:
            u = q.popleft()
            order.append(u)
            for w in list(outgoing.get(u, ())):
                incoming[w].discard(u)
                if not incoming[w]:
                    q.append(w)

        # if cycles or missing deps -> append remaining in a stable order
        remaining = [k for k in deps.keys() if k not in order]
        if remaining:
            logger.warning("CrossSection DAG has unresolved dependencies or cycles: %s", remaining)
            order.extend(sorted(remaining))

        # keep for quick reuse
        self._dag_order = order
        self._dag_by_id = by_id
        return order, by_id

    # -------------------------------------------------------------------------
    # Variable array preparation & unit harmonization (vector env)
    # -------------------------------------------------------------------------

    @staticmethod
    def _results_signature(results: List[Dict[str, float]], used_names: set) -> Tuple:
        """
        Small signature so embeddings cache reflects meaningful var changes.
        Samples up to (first, middle, last) for each used variable.
        """
        if not results:
            return ()
        idxs = [0, len(results)//2, len(results)-1] if len(results) > 2 else [0, len(results)-1]
        sig = []
        for name in sorted(used_names or []):
            vals = []
            for i in idxs:
                v = results[i].get(name, 0.0)
                try:
                    vals.append(round(float(v), 6))
                except Exception:
                    vals.append(0.0)
            sig.append((name, tuple(vals)))
        return tuple(sig)

    @staticmethod
    def _build_var_arrays_from_results(results: List[Dict[str, float]],
                                       defaults: Dict[str, float],
                                       keep: Optional[set] = None) -> Dict[str, np.ndarray]:
        """
        Make a name->array map (float64) for all stations.
        """
        names = keep or set()
        if not names:
            # if keep is empty, collect across all result dicts
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
    def _fix_var_units_inplace(var_arrays: Dict[str, np.ndarray], defaults: Dict[str, float]) -> None:
        """
        Heuristics identical to the engine:

        - Angles (W_/Q_/INCL_...) near 0.0 likely radians -> scale ×1000 to degrees (legacy convention).
        - Lengths (B_/T_/BEFF_/EX...) near <100 -> likely meters -> scale ×1000 to mm.
        If default is available in JSON, pick the scale that best matches the default.
        """
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

    # -------------------------------------------------------------------------
    # Expression collection & evaluation
    # -------------------------------------------------------------------------

    @staticmethod
    def _collect_used_variable_names(section_json: dict) -> set:
        """
        Find variable names referenced by point Coord expressions.
        """
        used = set()
        pts = (section_json or {}).get("Points") or section_json.get("points") or []
        for p in pts or []:
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            for expr in (coord[:2] or []):
                try:
                    txt = str(expr)
                except Exception:
                    continue
                # very simple parse: consider A..Z_0..9 tokens as potential names
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

    # -------------------------------------------------------------------------
    # Vectorized local-frame transforms
    # (XY here mean "local-Y" and "local-Z" in your convention)
    # -------------------------------------------------------------------------

    @staticmethod
    def _euclid_vectorized(X: np.ndarray, Y: np.ndarray,
                           ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euclidean: if 0 ref -> (X,Y)
                   if 1 ref -> add (px, py)
                   if 2 ref -> add (p1.x, p2.y)  [historic behavior]
        """
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
        # construction-al Y: origin at first reference point, X-axis along +X
        if not ref_pts:
            return np.zeros(1, float), np.zeros(1, float)
        p = ref_pts[0]
        px = float(p.get("x", 0.0) or p.get("X", 0.0))
        py = float(p.get("y", 0.0) or p.get("Y", 0.0))
        return np.asarray([px]), np.asarray([py])

    @staticmethod
    def _cz_vectorized(ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        # construction-al Z: same as CY in this minimal port (can be specialized)
        return CrossSection._cy_vectorized(ref_pts)

    # -------------------------------------------------------------------------
    # Public: vectorized evaluation of section points
    # -------------------------------------------------------------------------

    def eval_expressions_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Evaluate point Coord expressions for all stations at once.

        Returns:
            ids     : length-N point ids
            X_mm    : (S,N) local "Y" in mm    (historic naming)
            Y_mm    : (S,N) local "Z" in mm
        """
        ids = list(order)
        S = len(next(iter(var_arrays.values()))) if var_arrays else 0
        N = len(ids)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        # env for vector-eval
        env_base = {**_VECTOR_FUNCS}
        for j, pid in enumerate(ids):
            pj = by_id.get(pid) or {}
            coord = pj.get("Coord") or pj.get("coord") or [0, 0]
            x_expr = str(coord[0]) if len(coord) > 0 else "0"
            y_expr = str(coord[1]) if len(coord) > 1 else "0"

            # compile once per point
            cx, cy = self._compile_pair(x_expr, y_expr)

            # build env per point
            env = dict(env_base)
            for k, arr in var_arrays.items():
                # arrays stay arrays (vectorized)
                env[k] = arr

            # safe eval with arrays
            try:
                x_val = eval(cx, {"__builtins__": {}}, env)
            except Exception:
                x_val = np.full(S, np.nan, float)
            try:
                y_val = eval(cy, {"__builtins__": {}}, env)
            except Exception:
                y_val = np.full(S, np.nan, float)

            # to float arrays
            X[:, j] = np.asarray(x_val, float)
            Y[:, j] = np.asarray(y_val, float)

        if negate_x:
            X = -X

        # Interpret coordinates in the chosen local system (point-specific)
        # For the vectorized path we assume primary use is Euclidean coords already,
        # because reference-based mixes are uncommon across stations; users can
        # encode ref shifts in expressions. If needed, extend per-point ref frames here.

        return ids, X, Y

    def get_coordinates_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        # currently the same as eval_expressions_vectorized, separated for extensibility
        return self.eval_expressions_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id, negate_x=negate_x
        )

    # -------------------------------------------------------------------------
    # Loops index (cached per ids order)
    # -------------------------------------------------------------------------

    def _loops_idx(self, ids: List[str]) -> List[np.ndarray]:
        key = tuple(ids)
        hit = self._loops_cache.get(key)
        if hit is not None:
            return hit

        pts = self._collect_all_points()
        by_id = {str(p.get("Id") or p.get("id")): p for p in pts}
        loops: List[np.ndarray] = []
        for p in pts:
            poly = p.get("Polyline") or p.get("polyline")
            if not poly:
                continue
            try:
                # Polyline is stored as list of Ids, we convert to index array
                idxs = []
                for ref in (poly or []):
                    rid = str(ref)
                    if rid in by_id and rid in ids:
                        idxs.append(ids.index(rid))
                if len(idxs) >= 2:
                    loops.append(np.asarray(idxs, int))
            except Exception:
                pass

        self._loops_cache[key] = loops
        return loops

    # -------------------------------------------------------------------------
    # Public: compute local YZ points (vectorized)
    # -------------------------------------------------------------------------

    def compute_local_points(
        self,
        *,
        axis_var_results: List[Dict[str, float]],
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Vectorized main: returns (ids, X_mm, Y_mm, loops_idx).
        """
        section_json = {"Points": self._collect_all_points()}  # minimal view
        used_names = self._collect_used_variable_names(section_json)
        defaults = self.get_defaults()

        var_arrays = self._build_var_arrays_from_results(axis_var_results, defaults, keep=used_names)
        self._fix_var_units_inplace(var_arrays, defaults)

        order, by_id = self.build_dag()
        ids, X_mm, Y_mm = self.get_coordinates_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id, negate_x=negate_x
        )
        loops_idx = self._loops_idx(ids)
        return ids, X_mm, Y_mm, loops_idx

    # -------------------------------------------------------------------------
    # Public: embed local YZ as 3D using Axis (parallel-transport frames)
    # -------------------------------------------------------------------------

    def compute_embedded_points(
        self,
        *,
        axis,                          # models.axis.Axis (expects mm internal)
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
            return [], np.array([], float), np.zeros((0, 0, 3), float), np.zeros((0, 0), float), np.zeros((0, 0), float), []

        stations_mm_all = np.asarray(stations_m, float) * 1000.0
        smin = float(np.min(axis.stations))
        smax = float(np.max(axis.stations))
        keep = (stations_mm_all >= smin) & (stations_mm_all <= smax)
        if not np.any(keep):
            return [], np.array([], float), np.zeros((0, 0, 3), float), np.zeros((0, 0), float), np.zeros((0, 0), float), []

        stations_mm = stations_mm_all[keep]
        kept_results = [axis_var_results[i] for i, k in enumerate(keep) if k]

        ids, X_mm, Y_mm, loops_idx = self.compute_local_points(
            axis_var_results=kept_results, negate_x=negate_x
        )

        local_yz = np.dstack([X_mm, Y_mm])  # (S,N,2)   X==localY, Y==localZ (historic)
        P_mm = axis.embed_section_points_world(
            stations_mm, yz_points_mm=local_yz, x_offsets_mm=None, rotation_deg=float(twist_deg)
        )
        return ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx

    # -------------------------------------------------------------------------
    # Legacy scalar ReferenceFrame (kept for compatibility, marked deprecated)
    # -------------------------------------------------------------------------

    class ReferenceFrame:
        """Deprecated scalar path; kept for older preview tools."""
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

    # Convenience: single-station scalar compute (legacy preview)
    def compute_local_points_scalar(self, env_vars: Dict[str, float]) -> List[Dict[str, float]]:
        pts = self._collect_all_points()
        out = []
        for p in pts:
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            rf = CrossSection.ReferenceFrame(reference_type='euclidean', reference=p.get("Reference"), points=out, variables=env_vars)
            xy = rf.get_coordinates(coord)["coords"]
            out.append({"id": p.get("Id") or p.get("id"), "x": float(xy["x"]), "y": float(xy["y"])})
        return out
