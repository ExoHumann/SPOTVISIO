# utils.py
# Complete shared utilities, with all general functions moved from codebase.
# Includes _tiny_hash, cached_axis, load_json, build_axis_index, build_cross_section_index, rotation/math from Axis.py, embed_points_to_global_mm, clean_numbers (inferred from main.py).
# Readability: Grouped by category, docstrings, type hints, logging.
# No logic removed.

import functools
import hashlib
import json
import logging
import math

import numpy as np
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from models.axis import Axis


try:
    import orjson as fastjson
except ImportError:
    fastjson = None

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re
import numpy as np
import ast, math
from functools import lru_cache
import numpy as np

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

# utils.py
_ALLOWED_FUNC_NAMES = set(_SCALAR_FUNCS.keys()) | set(_VECTOR_FUNCS.keys())

@lru_cache(maxsize=4096)
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

def safe_eval_scalar(expr: str, vars_: Dict[str, float]) -> float:
    """
    Safe scalar eval for loaders / inspectors. Returns float('nan') on error.
    """
    try:
        return float(expr)  # numeric fast-path
    except Exception:
        pass
    try:
        code = _compile_expr(expr)
        env = {k: float(v) for k, v in vars_.items() if k not in _RESERVED_FUNC_NAMES}
        env.update(_SCALAR_FUNCS)
        return float(eval(code, {"__builtins__": {}}, env))
    except Exception:
        return float('nan')


# Hashing Utilities
def tiny_hash(arr: Optional[List[float]]) -> str:
    """Generate small hash for arrays (from AddClasses.py)."""
    if arr is None:
        arr = []
    data = (','.join(str(float(v)) for v in arr)).encode('utf-8')
    return hashlib.blake2b(data, digest_size=12).hexdigest()

# Caching Utilities
@lru_cache(maxsize=128)
def cached_axis(stations: Tuple[float, ...], x: Tuple[float, ...], y: Tuple[float, ...], z: Tuple[float, ...]) -> 'Axis':
    """Cached Axis creation (from AddClasses.py, main.py)."""
    from models.axis import Axis
    return Axis(stations=list(stations), x_coords=list(x), y_coords=list(y), z_coords=list(z), units='m')

# JSON Utilities
def load_json(path: str) -> Dict:
    """Load JSON with fastjson fallback (from spot_loader.py, AddClasses.py)."""
    logger.debug(f"Loading JSON from {path}")
    try:
        if fastjson:
            with open(path, 'rb') as f:
                return fastjson.loads(f.read())
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise

# Index Builders
def build_axis_index(axis_data: List[Dict], mapping: Dict[str, str]) -> Dict[str, Dict]:
    """Build axis index (from AddClasses.py, spot_loader.py)."""
    class_key = mapping.get("class", "Class")
    name_key = mapping.get("name", "Name")
    index = {}
    for d in axis_data or []:
        if d.get(class_key) == "Axis":
            name = str(d.get(name_key, "")).strip().lower()
            if name:
                index[name] = d
    return index

def build_cross_section_index(cross_sections: List[Any]) -> Dict[int, Any]:
    """Build cross-section index (from AddClasses.py)."""
    index = {}
    for cs in cross_sections or []:
        try:
            index[int(cs.ncs)] = cs
        except Exception as e:
            logger.warning(f"Invalid NCS: {e}")
    return index

# Math and Geometry Utilities (from Axis.py)
def rotate_vecs_about_axis(V, U, theta):
    """Rotate vectors (from Axis.py)."""
    theta = np.asarray(theta, float).reshape(-1, 1)
    c = np.cos(theta); s = np.sin(theta)
    udv = (U * V).sum(axis=1, keepdims=True)
    return V * c + np.cross(U, V) * s + U * udv * (1.0 - c)

def rotate_180_about_axis(P, A, U):
    """180° rotation (from Axis.py)."""
    R = P - A[:, None, :]
    dot = np.einsum('sni,si->sn', R, U)
    Rnew = 2.0*dot[:, :, None]*U[:, None, :] - R
    return A[:, None, :] + Rnew

def section_basis_from_tangent(U):
    """Basis from tangent (from Axis.py)."""
    U = np.asarray(U, float)
    up = np.array([0.0, 0.0, 1.0])
    dot_up = (U * up[None, :]).sum(axis=1)
    near_up = np.abs(dot_up) > 0.9
    ref = np.tile(up, (len(U), 1))
    ref[near_up] = np.array([1.0, 0.0, 0.0])
    Y = np.cross(ref, U)
    Yn = np.linalg.norm(Y, axis=1, keepdims=True); Yn[Yn == 0.0] = 1.0
    Y /= Yn
    Z = np.cross(U, Y)
    Zn = np.linalg.norm(Z, axis=1, keepdims=True); Zn[Zn == 0.0] = 1.0
    Z /= Zn
    mask = Z[:, 2] < 0.0
    if np.any(mask):
        Y[mask] *= -1.0
        Z[mask] *= -1.0
    return Y, Z

def embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=0.0):
    """Embed points to global (from Axis.py)."""
    stations_mm = np.asarray(stations_mm, float)
    X_mm = np.asarray(X_mm, float)
    Y_mm = np.asarray(Y_mm, float)

    A, U = axis.frames_for_stations(tuple(np.round(stations_mm, 6)))

    Yb, Zb = section_basis_from_tangent(U)

    if np.any(np.asarray(twist_deg) != 0.0):
        theta = np.deg2rad(np.asarray(twist_deg, float)).reshape(len(stations_mm))
        Yb = rotate_vecs_about_axis(Yb, U, theta)
        Zb = rotate_vecs_about_axis(Zb, U, theta)

    P = (A[:, None, :] +
         X_mm[:, :, None] * Yb[:, None, :] +
         Y_mm[:, :, None] * Zb[:, None, :])
    P = rotate_180_about_axis(P, A, U)
    return P

# Data Cleaning Utilities (from main.py)
def clean_numbers(nums: List[float]) -> List[float]:
    """Clean list by removing None/NaN (from main.py _clean_numbers)."""
    return [n for n in nums if n is not None and not math.isnan(n)]

def extend_numeric(base: List[float], addition: List[float]) -> None:
    """Extend list with addition (inferred from main.py _extend_numeric)."""
    base.extend(addition)

# Other general functions (e.g., from main.py solver_for, but keep in main if specific)

if __name__ == "__main__":
    logger.info("Utils test.")
    # Test hash
    print(tiny_hash([1.0, 2.0]))
    # Test axis
    test_stations = (0.0, 100.0)
    test_x = (0.0, 100.0)
    test_y = (0.0, 100.0)
    test_z = (0.0, 0.0)
    ax = cached_axis(test_stations, test_x, test_y, test_z)
    print(ax)
    # Add more as needed