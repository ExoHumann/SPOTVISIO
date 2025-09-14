# models/axis_variable.py
import bisect
import logging
import math
from functools import lru_cache
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

from collections import OrderedDict
import hashlib

# Global LRU cache: (fingerprint, stations_tuple) -> List[float]
_GLOBAL_AXISVAR_CACHE = OrderedDict()
_GLOBAL_AXISVAR_CACHE_MAX = 512  # tune as you like

def _axisvar_fingerprint(v: "AxisVariable") -> str:
    xs = tuple(round(float(x), 6) for x in v.xs)
    ys = tuple(round(float(y), 6) for y in v.ys)
    ts = tuple(str(t) for t in v.types)
    # Curve shape is what matters; name/description don’t affect values
    return hashlib.blake2b(repr((xs, ys, ts)).encode("utf-8"), digest_size=16).hexdigest()

def _global_cache_get(key):
    try:
        val = _GLOBAL_AXISVAR_CACHE.pop(key)
        _GLOBAL_AXISVAR_CACHE[key] = val  # move to MRU
        return val
    except KeyError:
        return None

def _global_cache_set(key, value):
    if key in _GLOBAL_AXISVAR_CACHE:
        _GLOBAL_AXISVAR_CACHE.pop(key)
    _GLOBAL_AXISVAR_CACHE[key] = value
    if len(_GLOBAL_AXISVAR_CACHE) > _GLOBAL_AXISVAR_CACHE_MAX:
        print(len(_GLOBAL_AXISVAR_CACHE) > _GLOBAL_AXISVAR_CACHE_MAX), _GLOBAL_AXISVAR_CACHE.popitem(last=False)  # LRU eviction


class AxisVariable:
    """Axis variable interpolation (refactored from AxisVariables.py)."""
    def __init__(self, values: List[Dict], delta: float = 0.0001, name: str = None, description: str = None):
        self.raw_points = values
        self.delta = delta
        self.name = name
        self.description = description
        self.xs = [v['X'] for v in self.raw_points]
        self.ys = [v['Value'] for v in self.raw_points]
        self.types = [v.get('Type', '#') for v in self.raw_points]
        self._adjust_discontinuities()
        self._prepare_segments()
        self.validate()

    def validate(self):
        """Validate points."""
        if len(self.raw_points) < 1:
            raise ValueError("AxisVariable requires at least 1 point.")

    def _adjust_discontinuities(self):
        adjusted_xs = self.xs.copy()
        for i in range(len(self.xs) - 1):
            if abs(self.xs[i] - self.xs[i + 1]) < 1e-10:
                adjusted_xs[i] = self.xs[i] - self.delta
                adjusted_xs[i + 1] = self.xs[i] + self.delta
        self.xs = adjusted_xs

    def _prepare_segments(self):
        self.segments = []
        for i in range(len(self.raw_points) - 1):
            x0, y0 = self.xs[i], self.ys[i]
            x1, y1 = self.xs[i + 1], self.ys[i + 1]
            is_discontinuity = abs(self.raw_points[i]['X'] - self.raw_points[i + 1]['X']) < 1e-10
            if is_discontinuity:
                m = (y1 - y0) / (x1 - x0)
                b = y0 - m * x0
                fn = lambda x, m=m, b=b: m * x + b
            else:
                interp_type = self.types[i]
                if abs(x0 - x1) < 1e-10:
                    fn = lambda x, y=y0: y
                else:
                    if interp_type == '#':
                        m = (y1 - y0) / (x1 - x0)
                        b = y0 - m * x0
                        fn = lambda x, m=m, b=b: m * x + b
                    elif interp_type == 'C':
                        fn = lambda x, y=y0: y
                    elif interp_type == '*':
                        spline = CubicSpline([x0, x1], [y0, y1], bc_type=((1, 0), (2, 0)))
                        fn = lambda x, spline=spline: float(spline(x))
                    else:
                        raise NotImplementedError(f"Type '{interp_type}' not supported.")
            self.segments.append((x0, x1, fn))

    def evaluate(self, station: float) -> float:
        for i in range(len(self.raw_points) - 1, -1, -1):
            if abs(station - self.raw_points[i]['X']) < 1e-6:
                return self.raw_points[i]['Value']
        if station < self.xs[0]:
            return self.ys[0]
        if station > self.xs[-1]:
            return self.ys[-1]
        idx = bisect.bisect_right(self.xs, station) - 1
        if idx < 0 or idx >= len(self.segments):
            return 0.0
        x0, x1, fn = self.segments[idx]
        return fn(station)

    def plot(self, start: float = 0, end: float = 100, step: float = 1):
        stations = np.arange(start, end + 1, step)
        values = np.array([self.evaluate(s) for s in stations])
        plt.figure(figsize=(10, 6))
        plt.scatter(self.xs, self.ys, color='red', zorder=5, label='Input Points')
        discontinuity_indices = [
            i for i in range(len(self.raw_points) - 1)
            if abs(self.raw_points[i]['X'] - self.raw_points[i + 1]['X']) < 1e-10
        ]
        seg_start = 0
        for idx in discontinuity_indices + [len(self.xs) - 1]:
            segment_xs = self.xs[seg_start:idx + 1]
            segment_ys = self.ys[seg_start:idx + 1]
            plt.plot(segment_xs, segment_ys, color='blue', label='Interpolated Line' if seg_start == 0 else None, zorder=1)
            seg_start = idx + 1
        for i in discontinuity_indices:
            plt.plot(
                [self.xs[i], self.xs[i + 1]],
                [self.ys[i], self.ys[i + 1]],
                color='purple',
                linestyle='dashed',
                linewidth=2,
                label='Discontinuity' if i == discontinuity_indices[0] else None,
                zorder=10
            )
        title = f'Axis Variable {self.name} Distribution (with Discontinuities): {self.description}' if self.name and self.description else f'Axis Variable {self.name or "Unknown"} Distribution (with Discontinuities)'
        plt.title(title)
        plt.xlabel('Station')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def create_axis_variables(var_list: List[Dict]) -> List["AxisVariable"]:
        axis_variables = []
        for var_data in var_list:
            if not all(key in var_data for key in ['VariableName', 'StationValue', 'VariableValues', 'VariableIntTypes']):
                continue
            points = [
                {'X': float(station), 'Value': float(value) if str(value).replace('.', '', 1).replace('-', '', 1).isdigit() else 0.0, 'Type': interp_type}
                for station, value, interp_type in zip(var_data['StationValue'], var_data['VariableValues'], var_data['VariableIntTypes'])
            ]
            axis_var = AxisVariable(
                points,
                name=var_data['VariableName'],
                description=var_data.get('VariableDescription', '')
            )
            axis_variables.append(axis_var)
        return axis_variables

    @staticmethod
    def evaluate_at_stations_cached(axis_variables: List["AxisVariable"], stations: List[float]) -> List[Dict[str, float]]:
        """
        Fast, cross-run cache: identical curves (same points/types) + same stations
        reuse computed columns. Falls back to the object’s evaluate() for the first build.
        """
        stations_tuple = tuple(float(s) for s in stations)
        cols = {}  # name -> list of values

        for v in axis_variables:
            if not v.name:
                continue
            fp = _axisvar_fingerprint(v)
            key = (fp, stations_tuple)

            vals = _global_cache_get(key)
            if vals is None:
                # Compute once using this instance, then cache by fingerprint
                vals = [v.evaluate(s) for s in stations_tuple]
                _global_cache_set(key, vals)

            cols[v.name] = vals

        # Assemble rows
        out = []
        for i in range(len(stations_tuple)):
            row = {}
            for name, col in cols.items():
                row[name] = col[i]
            out.append(row)
        return out


    @lru_cache(maxsize=256)
    def _evaluate_single_axis_var_on_tuple(axis_var_id, stations_tuple):
        raise RuntimeError("Do not call directly")

if __name__ == "__main__":
    test_data = [
        {"VariableName": "CSHEIGHT", "VariableStations": [0, 5, 10], "VariableValues": ["5", "20", "1"], "VariableIntTypes": ["#", "#", "#"], "VariableDescription": "CSHEIGHT:Axis Variable"}
    ]
    vars = AxisVariable.create_axis_variables(test_data)
    assert len(vars) == 1
    logger.info("axis_variable.py test passed.")