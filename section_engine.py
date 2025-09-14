from __future__ import annotations
# models/cross_section.py

import ast
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from utils import (
    _compile_expr,
    _sanitize_vars,
    _SCALAR_FUNCS,     # unused here but kept for parity if you bring scalar path back
    _VECTOR_FUNCS,
    _RESERVED_FUNC_NAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Minimal helpers for your JSON shapes
# ---------------------------------------------------------------------
def _normalize_point_row(p: dict) -> dict:
    """
    Canonicalize a point row to:
      {Id:str, Coord:[str,str], Reference:list[str], ReferenceType:str?}

    Supports:
      - JSON points: {"Id","Coord",[optional]"Reference","ReferenceType"}
      - Row points:  {"PointName","CoorY","CoorZ",[optional]"Reference","ReferenceType"}
    """
    if "Id" in p and ("Coord" in p or "coord" in p):
        pid = str(p["Id"])
        coord = p.get("Coord") or p.get("coord") or [0, 0]
        ref = list(p.get("Reference") or [])
        rtype = p.get("ReferenceType") or p.get("Type")
        return {"Id": pid, "Coord": [str(coord[0]), str(coord[1])], "Reference": ref, "ReferenceType": rtype}

    # Row fallback
    pid = str(p.get("PointName") or p.get("Id") or p.get("id") or "")
    def pick(expr, val):
        txt = str(expr) if expr is not None else ""
        if not txt or txt.strip().lower().startswith("error"):
            return "0" if (val is None or str(val).strip() == "") else str(val)
        return txt
    y_expr = pick(p.get("CoorY"), p.get("CoorYVal"))
    z_expr = pick(p.get("CoorZ"), p.get("CoorZVal"))
    ref = list(p.get("Reference") or [])
    rtype = p.get("ReferenceType") or p.get("Type")
    return {"Id": pid, "Coord": [y_expr, z_expr], "Reference": ref, "ReferenceType": rtype}


def _points_table(value: Any) -> Dict[str, Dict[str, Any]]:
    """Create {id: point_dict} from a list/dict."""
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(value, dict):
        for pid, row in value.items():
            out[str(pid)] = row if isinstance(row, dict) else {"Raw": row}
    elif isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                pid = (row.get("Id") or row.get("id") or row.get("PointName") or row.get("Name"))
                if pid is not None:
                    out[str(pid)] = row
    return out


# ---------------------------------------------------------------------
# CrossSection (lean)
# ---------------------------------------------------------------------
@dataclass(kw_only=True)
class CrossSection:
    # Metadata you pass in from your rows
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

    # Attached section JSON (Points/Loops/etc.)
    json_data: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    json_source: Optional[str] = field(default=None, init=False, repr=False)

    # Simple DAG cache
    _dag_cache_key: Optional[str] = field(default=None, init=False, repr=False)
    _dag_cache_val: Optional[Tuple[List[str], Dict[str, dict]]] = field(default=None, init=False, repr=False)

    # --------------------- attach / defaults ---------------------
    def attach_json_payload(self, payload: Dict[str, Any], source: Optional[str] = None) -> None:
        if isinstance(payload, dict):
            self.json_data = payload
            self.json_source = source

    def get_defaults(self) -> Dict[str, float]:
        """
        Read variable defaults in mm. Prefer json_data["Variables"] (dict),
        else accept row-style list [{VariableName, VariableValue}, ...].
        """
        # JSON dict
        if isinstance(self.json_data, dict) and isinstance(self.json_data.get("Variables"), dict):
            out = {}
            for k, v in self.json_data["Variables"].items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
            if out:
                return out

        # Row fallback
        out: Dict[str, float] = {}
        raw = self.variables or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try: out[str(k)] = float(v)
                except Exception: pass
        else:
            for row in raw or []:
                if isinstance(row, dict):
                    n = str(row.get("VariableName"))
                    try: out[n] = float(row.get("VariableValue", 0.0) or 0.0)
                    except Exception: pass
        return out

    # ------------------- gather points from JSON -------------------
    def _collect_all_points(self) -> List[dict]:
        """
        Pull all point-like definitions your files actually use:
        - Points
        - Loops[].Points
        - PointReinforcements[].Point
        - LineReinforcements[].PointStart / .PointEnd
        - NonEffectiveZones[].PointStart / .PointEnd
        """
        pts, seen = [], set()
        jd = self.json_data or {}

        def add(p):
            if isinstance(p, dict):
                q = _normalize_point_row(p)
                pid = q.get("Id")
                if pid and pid not in seen:
                    seen.add(pid); pts.append(q)

        for p in jd.get("Points") or []: add(p)
        for lp in jd.get("Loops") or []:
            for p in lp.get("Points") or []: add(p)
        for pr in jd.get("PointReinforcements") or []:
            add(pr.get("Point", {}))
        for lr in jd.get("LineReinforcements") or []:
            add(lr.get("PointStart", {})); add(lr.get("PointEnd", {}))
        for nz in jd.get("NonEffectiveZones") or []:
            add(nz.get("PointStart", {})); add(nz.get("PointEnd", {}))

        return pts

    # ---------------------- DAG (by Reference) ----------------------
    def _dag_key(self) -> str:
        pts = self._collect_all_points() if self.json_data else [ _normalize_point_row(p) for p in (self.points or []) ]
        # small stable fingerprint string
        def row_sig(p):
            c = p.get("Coord") or ["0","0"]
            r = tuple(p.get("Reference") or [])
            return (p.get("Id"), str(c[0]), str(c[1]), r)
        return json.dumps(sorted([row_sig(p) for p in pts]), sort_keys=True, default=str)

    def build_dag(self) -> Tuple[List[str], Dict[str, dict]]:
        key = self._dag_key()
        if self._dag_cache_key == key and self._dag_cache_val is not None:
            return self._dag_cache_val

        all_pts = self._collect_all_points() if self.json_data else [ _normalize_point_row(p) for p in (self.points or []) ]
        by_id: Dict[str, dict] = {}
        deps: Dict[str, set] = {}
        for p in all_pts:
            pid = str(p.get("Id") or "")
            if not pid: continue
            by_id[pid] = p
            refs = p.get("Reference") or []
            deps[pid] = set(str(x) for x in refs if x is not None)

        # Kahn
        incoming = {k: set(v) for k, v in deps.items()}
        outgoing: Dict[str, set] = {}
        for k, vs in incoming.items():
            for v in vs:
                outgoing.setdefault(v, set()).add(k)

        order: List[str] = []
        from collections import deque
        q = deque(sorted([k for k, s in incoming.items() if not s]))
        while q:
            u = q.popleft()
            order.append(u)
            for w in list(outgoing.get(u, ())):
                incoming[w].discard(u)
                if not incoming[w]:
                    q.append(w)

        # Fallback in case of cycles/missing refs
        remain = [k for k in deps.keys() if k not in order]
        if remain:
            logger.warning("DAG unresolved (cycles/missing refs): %s", remain)
            order.extend(sorted(remain))

        self._dag_cache_key = key
        self._dag_cache_val = (order, by_id)
        return order, by_id

    # ---------------- variable prep & unit nudge ----------------
    @staticmethod
    def _collect_used_variable_names(points_like: List[dict]) -> set:
        """AST-scan of Coord expressions to find variable names (minus reserved)."""
        used = set()
        for p in points_like:
            coord = p.get("Coord") or ["0", "0"]
            for s in coord[:2]:
                txt = str(s)
                try:
                    node = ast.parse(txt, mode="eval")
                    for n in ast.walk(node):
                        if isinstance(n, ast.Name): used.add(n.id)
                except Exception:
                    # simple fallback
                    for token in filter(None, [t.strip() for t in "".join(ch if (ch.isalnum() or ch == "_") else " " for ch in txt).split(" ")]):
                        if not token[0].isdigit(): used.add(token)
        return used - _RESERVED_FUNC_NAMES

    @staticmethod
    def _build_var_arrays(results: List[Dict[str, float]], defaults: Dict[str, float], only: Optional[set]) -> Dict[str, np.ndarray]:
        names = set(only or [])
        if not names:
            for d in results or []: names.update(d.keys())
        S = len(results or [])
        out: Dict[str, np.ndarray] = {}
        for n in names:
            a = np.empty(S, float)
            for i, row in enumerate(results or []):
                val = row.get(n, defaults.get(n, 0.0))
                try: a[i] = float(val)
                except Exception: a[i] = float(defaults.get(n, 0.0))
            out[n] = a
        return out

    @staticmethod
    def _fix_units_inplace(var_arrays: Dict[str, np.ndarray], defaults: Dict[str, float]) -> None:
        """Tiny heuristic: angle-ish names (W_/Q_/INCL_) close to 0 → likely radians*1000; length-ish small → meters*1000."""
        def looks_angle(n): n = n.upper(); return n.startswith(("W_","Q_","INCL_"))
        def looks_len(n):   n = n.upper(); return n.startswith(("B_","T_","BEFF_","EX"))
        for name, arr in list(var_arrays.items()):
            a = np.asarray(arr, float); 
            if a.size == 0: continue
            d = defaults.get(name)
            scale = 1.0
            if d is not None and d != 0:
                med = float(np.nanmedian(np.abs(a))) or 0.0
                cands = (1.0, 1000.0, 0.001)
                pick = min(cands, key=lambda s: abs(med*s - d) / max(1.0, abs(d)))
                scale = pick
            else:
                if looks_angle(name) and np.nanmedian(np.abs(a)) < 0.5: scale = 1000.0
                if looks_len(name) and 0 < np.nanmedian(np.abs(a)) < 100: scale = 1000.0
            if scale != 1.0: var_arrays[name] = a * scale

    # ---------------- vector evaluation (Euclid + small extras) ----------------
    def _eval_points_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        S = next((len(a) for a in var_arrays.values() if isinstance(a, np.ndarray)), 1)
        N = len(order)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        idx_of = {pid: j for j, pid in enumerate(order)}
        env = {**_VECTOR_FUNCS, **_sanitize_vars(var_arrays)}

        def eval_vec(expr: Any) -> np.ndarray:
            try:
                v = float(expr)
                return np.full(S, v, float)
            except Exception:
                pass
            try:
                code = _compile_expr(expr)
                out = eval(code, {"__builtins__": {}}, env)
                out = np.asarray(out, float)
                return np.full(S, float(out), float) if out.ndim == 0 else out
            except Exception as e:
                logger.debug("eval %r failed: %s", expr, e)
                return np.full(S, np.nan, float)

        for j, pid in enumerate(order):
            p = by_id[pid]
            c = p.get("Coord") or ["0", "0"]
            r = p.get("Reference") or []
            t = (p.get("ReferenceType") or "Euclidean").strip().lower()

            x = eval_vec(c[0]); y = eval_vec(c[1])
            if negate_x: x = -x

            if t in ("euclidean","e","c","carthesian"):
                if len(r) == 0:
                    X[:, j], Y[:, j] = x, y
                elif len(r) == 1:
                    j0 = idx_of.get(str(r[0]))
                    if j0 is None: X[:, j], Y[:, j] = x, y
                    else:          X[:, j], Y[:, j] = x + X[:, j0], y + Y[:, j0]
                else:
                    jx = idx_of.get(str(r[0])); jy = idx_of.get(str(r[1]))
                    X[:, j] = x + (X[:, jx] if jx is not None else 0.0)
                    Y[:, j] = y + (Y[:, jy] if jy is not None else 0.0)

            elif t in ("polar","p"):
                if len(r) < 2:
                    X[:, j], Y[:, j] = x, y
                else:
                    j0, j1 = idx_of.get(str(r[0])), idx_of.get(str(r[1]))
                    if j0 is None or j1 is None:
                        X[:, j], Y[:, j] = x, y
                    else:
                        dx = X[:, j1] - X[:, j0]; dy = Y[:, j1] - Y[:, j0]
                        L = np.hypot(dx, dy); Ls = np.where(L == 0.0, 1.0, L)
                        ux, uy = dx/Ls, dy/Ls; vx, vy = -uy, ux
                        X[:, j] = X[:, j0] + x*ux + y*vx
                        Y[:, j] = Y[:, j0] + x*uy + y*vy

            elif t in ("constructionaly","cy"):
                if len(r) == 3:
                    j1, j2, j3 = idx_of.get(str(r[0])), idx_of.get(str(r[1])), idx_of.get(str(r[2]))
                    if None not in (j1, j2, j3):
                        dx = X[:, j2] - X[:, j1]; dy = Y[:, j2] - Y[:, j1]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            m = np.where(dx != 0.0, dy/np.where(dx==0.0,1.0,dx), 0.0)
                        c0 = Y[:, j1] - m*X[:, j1]
                        Y[:, j] = m*X[:, j3] + c0; X[:, j] = X[:, j3]
                    else:
                        X[:, j], Y[:, j] = x, y
                else:
                    X[:, j], Y[:, j] = x, y

            elif t in ("constructionalz","cz"):
                if len(r) == 3:
                    j1, j2, j3 = idx_of.get(str(r[0])), idx_of.get(str(r[1])), idx_of.get(str(r[2]))
                    if None not in (j1, j2, j3):
                        dx = X[:, j2] - X[:, j1]; dy = Y[:, j2] - Y[:, j1]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            m = np.where(dx != 0.0, dy/np.where(dx==0.0,1.0,dx), 0.0)
                        c0 = Y[:, j1] - m*X[:, j1]
                        X[:, j] = np.where(m != 0.0, (Y[:, j3] - c0)/m, X[:, j3]); Y[:, j] = Y[:, j3]
                    else:
                        X[:, j], Y[:, j] = x, y
                else:
                    X[:, j], Y[:, j] = x, y

            else:
                X[:, j], Y[:, j] = x, y

        return order, X, Y

    # ---------------- loops index ----------------
    def _loops_idx(self, ids: List[str]) -> List[np.ndarray]:
        """Map Loops[].Points Ids to column indices."""
        jd = self.json_data or {}
        loops = []
        id_to_col = {pid: i for i, pid in enumerate(ids)}
        for lp in jd.get("Loops") or []:
            idxs = []
            for p in lp.get("Points") or []:
                pid = str((p.get("Id") or p.get("id") or p.get("PointName") or "").strip())
                if pid and pid in id_to_col: idxs.append(id_to_col[pid])
            if len(idxs) >= 2: loops.append(np.asarray(idxs, dtype=int))
        return loops

    # ---------------- public: compute local ----------------
    def compute_local_points(
        self,
        *,
        axis_var_results: List[Dict[str, float]],
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Vector pipeline:
          1) Gather point-like rows used in your JSONs
          2) Build DAG (References)
          3) Build station arrays for used variables (+ light unit fix)
          4) Evaluate coordinates
          5) Return (ids, X_mm, Y_mm, loop_indices)
        """
        if self.json_data:
            pts = self._collect_all_points()
        else:
            pts = [ _normalize_point_row(p) for p in (self.points or []) ]

        used = self._collect_used_variable_names(pts)
        defaults = self.get_defaults()
        var_arrays = self._build_var_arrays(axis_var_results, defaults, used)
        self._fix_units_inplace(var_arrays, defaults)

        order, by_id = self.build_dag()
        ids, X_mm, Y_mm = self._eval_points_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id, negate_x=negate_x
        )
        loops_idx = self._loops_idx(ids)
        return ids, X_mm, Y_mm, loops_idx

    # ---------------- minimal from_dict for your rows ----------------
    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "CrossSection":
        """
        Minimal constructor: map fields that appear in your Crossection_JSON rows.
        Does not read files; use attach_json_payload to add JSON content.
        """
        return cls(
            no=row.get("No"),
            class_name=row.get("Class"),
            type=row.get("Type"),
            description=row.get("Description"),
            name=row.get("Name"),
            inactive=row.get("InActive"),
            ncs=row.get("NCS"),
            material1=row.get("Material1"),
            material2=row.get("Material2"),
            material_reinf=row.get("Material_Reinf"),
            json_name=row.get("JSON_name"),
            points=row.get("Points"),
            variables=row.get("Variables"),
        )


# ---------------------------------------------------------------------
# Tiny self-test: runs purely on your SectionData shape (no Axis needed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # == Minimal sample based on your SectionData ==
    section_json = {
        "Points": [
            {"Id": "O",  "Coord": ["0", "0"], "ReferenceType": "Euclidean"},
            {"Id": "UM", "Coord": ["0", "H_QS"], "ReferenceType": "Euclidean"},
            {"Id": "AKL","Coord": ["B_TR/2-37", "T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100"], "ReferenceType": "Euclidean"},
            {"Id": "1L","Coord":["-300","100"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
        ],
        "Loops": [
            {"Id":"L1","Points":[{"Id":"O"},{"Id":"UM"},{"Id":"AKL"},{"Id":"1L"}]}
        ],
        "Variables": {
            "H_QS": 2750.000238418579,
            "B_TR": 8575.000762939453,
            "T_FBA": 600.0000238418579,
            "Q_NG": 3.0
        }
    }

    # Row info (only to fill metadata; not used for geometry when JSON is attached)
    row = {
        "No":"1","Class":"CrossSection","Type":"Deck","Description":"test","Name":"MASTER_Deck",
        "NCS":111,"Material1":101,"Material2":0,"Material_Reinf":200,
        "JSON_name":["MASTER_SECTION\\SectionData.json"],
    }

    cs = CrossSection.from_dict(row)
    cs.attach_json_payload(section_json)

    # Two stations: station 0 = defaults; station 1 bumps H_QS
    st0 = cs.get_defaults()
    st1 = dict(st0); st1["H_QS"] = st0["H_QS"] + 250.0
    ids, X_mm, Y_mm, loops_idx = cs.compute_local_points(axis_var_results=[st0, st1])

    print("ids:", ids[:8], "… (total:", len(ids), ")")
    print("X shape:", X_mm.shape, "Y shape:", Y_mm.shape)
    print("station0 first 3:", list(zip(X_mm[0,:3], Y_mm[0,:3])))
    print("station1 first 3:", list(zip(X_mm[1,:3], Y_mm[1,:3])))
    print("loops:", [idx.tolist() for idx in loops_idx])
