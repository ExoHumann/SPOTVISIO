from __future__ import annotations
# models/cross_section.py
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Union
from utils import _compile_expr, _sanitize_vars, _SCALAR_FUNCS, _VECTOR_FUNCS

import logging
logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
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
