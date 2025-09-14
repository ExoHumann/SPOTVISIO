# models/axis.py
from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------- small numeric helpers (module-level) ----------

def _normalize_rows(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row vector; guard against zero-length rows.
    """
    a = np.asarray(a, float)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return a / n


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    3x3 rotation matrix rotating around 'axis' (shape (3,)) by angle 'theta' (radians).
    'axis' does not need to be unit (we normalize internally).
    """
    ax = np.asarray(axis, float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]], dtype=float)
    I = np.eye(3, dtype=float)
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


# ---------- Axis class (single embedding path) ----------

class Axis:
    """
    Geometric axis sampled by stations; units stored internally as millimeters.

    Provides:
      - `get_position_at_station(station_mm)` -> (x,y,z) in mm
      - `_positions_and_tangents(stations_mm)` -> (P,T) where P:(N,3), T:(N,3) unit
      - `parallel_transport_frames(stations_mm)` -> (N,B) minimal-twist normals/binormals
      - `embed_section_points_world(stations_mm, yz_points_mm, ...)` -> (N,M,3) world coords

    Notes:
      * All public inputs/outputs are in **mm** (except rotation_deg).
      * `yz_points_mm` are local section coordinates (Y,Z) per named point.
      * Optional `rotation_deg` applies an in-plane rotation inside each local YZ plane.
    """

    def __init__(
        self,
        stations: Optional[List[float]] = None,
        x_coords: Optional[List[float]] = None,
        y_coords: Optional[List[float]] = None,
        z_coords: Optional[List[float]] = None,
        units: str = "m",
    ):
        if any(v is None for v in (stations, x_coords, y_coords, z_coords)):
            raise TypeError("Axis requires stations and x/y/z coords.")

        s = np.asarray(stations, dtype=float)
        x = np.asarray(x_coords, dtype=float)
        y = np.asarray(y_coords, dtype=float)
        z = np.asarray(z_coords, dtype=float)
        if not (len(s) == len(x) == len(y) == len(z)):
            raise ValueError("Axis: stations, x, y, z must have the same length.")

        # Convert to mm; sort by station
        factor = 1.0 if units.lower() == "mm" else 1000.0
        s_mm = s * factor
        x_mm = x * factor
        y_mm = y * factor
        z_mm = z * factor
        order = np.argsort(s_mm)

        self.stations = s_mm[order]
        self.x_coords = x_mm[order]
        self.y_coords = y_mm[order]
        self.z_coords = z_mm[order]

        self.validate()

    # ----- basic introspection -----

    def validate(self) -> None:
        """Minimal validation of axis data."""
        if len(self.stations) < 2:
            raise ValueError("Axis requires at least 2 sample points.")
        # Stations are already sorted in __init__; warn on duplicates (flat segments).
        if np.any(np.diff(self.stations) == 0.0):
            logger.warning("Axis: duplicate station values detected; tangents may be undefined locally.")

    def __len__(self) -> int:
        return len(self.stations)

    def __repr__(self) -> str:
        n = len(self)
        rng = (float(self.stations[0]), float(self.stations[-1])) if n else (None, None)
        return f"Axis(n={n}, stations_mm=[{rng[0]}, {rng[1]}])"
    
    def clamp_station_m(self, s_m: float) -> float:
        """
        Clamp a station given in meters to the axis domain, return meters.
        Axis stores stations in mm internally.
        """
        s_mm = float(s_m) * 1000.0
        s_mm = float(np.clip(s_mm, float(self.stations[0]), float(self.stations[-1])))
        return s_mm / 1000.0

    def point_at_m(self, s_m: float) -> np.ndarray:
        """
        Interpolated world position (mm) at a station given in meters.
        """
        s_mm = self.clamp_station_m(float(s_m)) * 1000.0
        x, y, z = self.get_position_at_station(s_mm)
        return np.array([x, y, z], dtype=float)


    # ----- sampling -----

    def get_segment_for_station(self, station: float) -> Tuple[Optional[int], Optional[float]]:
        """
        Return (i, t) such that station lies on segment i..i+1 and
        position = P[i] * (1-t) + P[i+1] * t. Results are in [0,1].

        Returns (None, None) if station is out of the axis range.
        """
        s = float(station)
        s0, s1 = float(self.stations[0]), float(self.stations[-1])
        if s < s0 or s > s1:
            logger.warning("Station %.3f mm is outside [%.3f, %.3f] mm.", s, s0, s1)
            return None, None

        i = int(np.searchsorted(self.stations, s, side="right") - 1)
        i = max(0, min(i, len(self.stations) - 2))
        ds = self.stations[i + 1] - self.stations[i]
        if ds == 0.0:
            return i, 0.0
        t = (s - self.stations[i]) / ds
        return i, float(t)

    def get_position_at_station(self, station: float) -> Tuple[float, float, float]:
        """
        Interpolate global position (X,Y,Z) at the given station (mm).
        """
        s = float(station)
        # Fast vector interpolation along the stored polyline:
        x = float(np.interp(s, self.stations, self.x_coords))
        y = float(np.interp(s, self.stations, self.y_coords))
        z = float(np.interp(s, self.stations, self.z_coords))
        return x, y, z

    def _positions_and_tangents(self, stations_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample world positions P and unit tangents T at the given stations (mm).
        P:(N,3) in mm, T:(N,3) unitless.
        """
        stations_mm = np.asarray(stations_mm, float)
        # Positions via 1D interpolation on each component
        Px = np.interp(stations_mm, self.stations, self.x_coords)
        Py = np.interp(stations_mm, self.stations, self.y_coords)
        Pz = np.interp(stations_mm, self.stations, self.z_coords)
        P = np.stack([Px, Py, Pz], axis=1)  # (N,3)

        # Tangents via centered finite differences (forward/backward at ends)
        T = np.zeros_like(P)
        if len(P) >= 2:
            T[1:-1] = P[2:] - P[:-2]
            T[0]    = P[1] - P[0]
            T[-1]   = P[-1] - P[-2]
        T = _normalize_rows(T)
        return P, T

    # ----- frames & embedding -----

    def parallel_transport_frames(self, stations_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build rotation-minimizing frames along the axis.
        Returns (T, N, B), each (N,3), where:
          T: unit tangent,
          N: transported normal (minimal twist),
          B: binormal = T x N.
        """
        P, T = self._positions_and_tangents(stations_mm)
        N = np.zeros_like(P)
        B = np.zeros_like(P)

        # Initial normal from an up-hint projected orthogonal to T[0]
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        n0 = up - T[0] * float(up @ T[0])
        if np.linalg.norm(n0) < 1e-9:
            # Fallback if up is (near) parallel to T0
            up = np.array([0.0, 1.0, 0.0], dtype=float)
            n0 = up - T[0] * float(up @ T[0])
        N[0] = _normalize_rows(n0)
        B[0] = _normalize_rows(np.cross(T[0], N[0]))

        # Parallel-transport along the curve using Rodrigues rotations
        for i in range(1, len(P)):
            t_prev, t_cur = T[i - 1], T[i]
            v = np.cross(t_prev, t_cur)
            s = float(np.linalg.norm(v))
            c = float(np.clip(t_prev @ t_cur, -1.0, 1.0))

            if s < 1e-12:
                # No significant change in tangent: keep previous normal/binormal
                N[i] = N[i - 1]
                B[i] = B[i - 1]
                continue

            axis = v / s
            theta = math.atan2(s, c)
            R = _rodrigues(axis, theta)
            Ni = R @ N[i - 1]

            # Re-orthonormalize (accumulated drift guard)
            Ni = Ni - (Ni @ t_cur) * t_cur
            Ni = Ni / (np.linalg.norm(Ni) + 1e-12)
            N[i] = Ni
            B[i] = np.cross(t_cur, Ni)

        return T, N, B

    def embed_section_points_world(
        self,
        stations_mm: np.ndarray,
        yz_points_mm: np.ndarray,
        x_offsets_mm: np.ndarray | None = None,
        rotation_deg: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """
        Transform local (Y,Z) section points into world coordinates using
        rotation-minimizing frames at each requested station.

        Args
        ----
        stations_mm : (N,) array
            Stations (mm) at which to place the section.
        yz_points_mm : (M,2) or (N,M,2) array
            Local section points (Y,Z) in mm. If (M,2), broadcast to all stations.
        x_offsets_mm : optional (N,) array
            Forward offsets along the tangent (local X) in mm. Default 0.
        rotation_deg : float or (N,) array
            Additional in-plane rotation (degrees) applied within each local YZ plane.

        Returns
        -------
        W : (N,M,3) array
            World coordinates in mm. W[i, k] corresponds to point k of the section at station i.
        """
        stations_mm = np.asarray(stations_mm, float)
        if stations_mm.ndim != 1:
            raise ValueError("stations_mm must be a 1D array of stations in mm.")

        # Axis frames
        P, T = self._positions_and_tangents(stations_mm)   # (N,3), (N,3)
        _, N, B = self.parallel_transport_frames(stations_mm)  # (T,N,B) -> we only need N,B here

        # Normalize/expand section arrays
        yz = np.asarray(yz_points_mm, float)
        if yz.ndim == 2:
            # (M,2) -> broadcast to (N,M,2)
            yz = np.broadcast_to(yz, (len(stations_mm),) + yz.shape)
        elif yz.ndim != 3 or yz.shape[0] != len(stations_mm) or yz.shape[2] != 2:
            raise ValueError("yz_points_mm must be (M,2) or (N,M,2).")

        # Optional in-plane rotation (per-station)
        rot = np.asarray(rotation_deg, float)
        if rot.ndim == 0:
            rot = np.full(len(stations_mm), float(rotation_deg), dtype=float)
        theta = np.deg2rad(rot)
        c, s = np.cos(theta), np.sin(theta)  # (N,)

        y = yz[..., 0]                       # (N,M)
        z = yz[..., 1]                       # (N,M)
        y_rot = c[:, None] * y - s[:, None] * z
        z_rot = s[:, None] * y + c[:, None] * z

        # Optional longitudinal offsets (local X)
        if x_offsets_mm is None:
            x_offsets_mm = np.zeros(len(stations_mm), dtype=float)
        x_offsets_mm = np.asarray(x_offsets_mm, float)
        if x_offsets_mm.shape != (len(stations_mm),):
            raise ValueError("x_offsets_mm must be None or shape (N,) to match stations_mm.")

        # Compose world coordinates
        W = (P[:, None, :]                         # (N,1,3)
             + x_offsets_mm[:, None, None] * T[:, None, :]   # local X along T
             + y_rot[:, :, None] * N[:, None, :]             # local Y along N
             + z_rot[:, :, None] * B[:, None, :])            # local Z along B
        return W


# ---------- tiny self-test ----------

if __name__ == "__main__":
    test_stations = [0, 100, 200]
    test_x = [0, 100, 200]
    test_y = [0, 100, 100]
    test_z = [0, 100, 200]
    axis = Axis(test_stations, test_x, test_y, test_z, units="m")
    assert len(axis) == 3
    # simple embed smoke-test
    stations_mm = np.array([0.0, 100000.0, 200000.0])  # mm (since units='m' -> stored as mm)
    yz = np.array([[0.0, 0.0], [1000.0, 0.0]])         # two points in the section (Y,Z) mm
    W = axis.embed_section_points_world(stations_mm, yz, rotation_deg=0.0)
    print("Embedded shape:", W.shape)
    logger.info("axis.py test passed.")
