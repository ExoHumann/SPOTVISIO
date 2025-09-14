

# import numpy as np
# import plotly.graph_objects as go
# import os
# import webbrowser
# import math
# from functools import lru_cache
# import numpy as np

# class Axis:
#     """
#     Axis stationing and coordinates.
#     Internally stored in millimeters (mm) for the rest of the pipeline.

#     You may pass either positionally:
#         Axis(stations_m, x_m, y_m, z_m)

#     or via keywords used elsewhere in your code:
#         Axis(stations=..., x_coords=..., y_coords=..., z_coords=...)
#         Axis(stations_axis=..., x_coords=..., y_coords=..., z_coords=...)

#     Units:
#       - By default we assume inputs are in meters and convert to mm (units='m').
#       - If your inputs are already in millimeters, pass units='mm'.
#     """

#     def __init__(self,
#                  stations_m=None, x_m=None, y_m=None, z_m=None,
#                  **kwargs):
#         # ---- Accept legacy/alternate keyword names ----
#         # stations
#         if stations_m is None:
#             stations_m = kwargs.pop('stations', None)
#         if stations_m is None:
#             stations_m = kwargs.pop('stations_axis', None)

#         # coords
#         if x_m is None: x_m = kwargs.pop('x_coords', None) or kwargs.pop('x', None)
#         if y_m is None: y_m = kwargs.pop('y_coords', None) or kwargs.pop('y', None)
#         if z_m is None: z_m = kwargs.pop('z_coords', None) or kwargs.pop('z', None)

#         # units (default meters -> convert to mm)
#         units = kwargs.pop('units', 'm')  # 'm' or 'mm'

#         # Any unexpected kwargs left?
#         if kwargs:
#             # Not fatal, but useful to surface
#             # print(f"[Axis] Unused kwargs: {list(kwargs.keys())}")
#             pass

#         # ---- Validate presence ----
#         if any(v is None for v in (stations_m, x_m, y_m, z_m)):
#             raise TypeError(
#                 "Axis(...) requires stations and x/y/z coordinate arrays.\n"
#                 "Accepted arg names: "
#                 "  positional: (stations_m, x_m, y_m, z_m) "
#                 "  or keywords: stations / stations_axis, x_coords, y_coords, z_coords. "
#                 "Optionally pass units='mm' if arrays are already in millimeters."
#             )

#         # ---- To numpy ----
#         s = np.asarray(stations_m, dtype=float)
#         x = np.asarray(x_m,       dtype=float)
#         y = np.asarray(y_m,       dtype=float)
#         z = np.asarray(z_m,       dtype=float)

#         # ---- Length check ----
#         if not (len(s) == len(x) == len(y) == len(z)):
#             raise ValueError(
#                 f"Axis: Stations/X/Y/Z lengths differ: {len(s)}, {len(x)}, {len(y)}, {len(z)}"
#             )

#         # ---- Units -> mm ----
#         if str(units).lower() in ('mm', 'millimeter', 'millimeters'):
#             factor = 1.0
#         else:
#             # default: meters to millimeters
#             factor = 1000.0

#         s_mm = s * factor
#         x_mm = x * factor
#         y_mm = y * factor
#         z_mm = z * factor

#         # ---- Ensure increasing stations for interp/searchsorted ----
#         order = np.argsort(s_mm)
#         self.stations  = s_mm[order]
#         self.x_coords  = x_mm[order]
#         self.y_coords  = y_mm[order]
#         self.z_coords  = z_mm[order]
    
#     def __len__(self):
#         return len(self.stations)

#     def __repr__(self):
#         n = len(self)
#         rng = (float(self.stations[0]), float(self.stations[-1])) if n else (None, None)
#         return f"Axis(n={n}, stations_mm=[{rng[0]}, {rng[1]}])"

    
#     @lru_cache(maxsize=64)
#     def frames_for_stations(self, stations_mm_tuple):
#         stations_mm = np.asarray(stations_mm_tuple, float)
#         s = np.asarray(self.stations, float)
#         x = np.asarray(self.x_coords, float)
#         y = np.asarray(self.y_coords, float)
#         z = np.asarray(self.z_coords, float)
#         Ax = np.interp(stations_mm, s, x)
#         Ay = np.interp(stations_mm, s, y)
#         Az = np.interp(stations_mm, s, z)
#         idx = np.searchsorted(s, stations_mm, side='right') - 1
#         idx = np.clip(idx, 0, len(s) - 2)
#         dx = x[idx + 1] - x[idx]; dy = y[idx + 1] - y[idx]; dz = z[idx + 1] - z[idx]
#         L = np.sqrt(dx*dx + dy*dy + dz*dz); L[L == 0.0] = 1.0
#         U = np.stack([dx/L, dy/L, dz/L], axis=1)
#         A = np.stack([Ax, Ay, Az], axis=1)
#         return A, U
    
    
#     @staticmethod
#     def _rotate_vecs_about_axis(V, U, theta):
#         """Rotate each row of V about corresponding U by angle theta (radians). V,U:(S,3), theta:(S,)"""
#         theta = np.asarray(theta, float).reshape(-1, 1)
#         c = np.cos(theta); s = np.sin(theta)
#         udv = (U * V).sum(axis=1, keepdims=True)
#         return V * c + np.cross(U, V) * s + U * udv * (1.0 - c)

#     @staticmethod
#     def _rotate_180_about_axis(P, A, U):
#         R = P - A[:, None, :]
#         dot = np.einsum('sni,si->sn', R, U)
#         Rnew = 2.0*dot[:, :, None]*U[:, None, :] - R
#         return A[:, None, :] + Rnew

#     @staticmethod
#     def _section_basis_from_tangent(U):
#         U = np.asarray(U, float)
#         up = np.array([0.0, 0.0, 1.0])
#         dot_up = (U * up[None, :]).sum(axis=1)
#         near_up = np.abs(dot_up) > 0.9
#         ref = np.tile(up, (len(U), 1))
#         ref[near_up] = np.array([1.0, 0.0, 0.0])
#         Y = np.cross(ref, U)
#         Yn = np.linalg.norm(Y, axis=1, keepdims=True); Yn[Yn == 0.0] = 1.0
#         Y /= Yn
#         Z = np.cross(U, Y)
#         Zn = np.linalg.norm(Z, axis=1, keepdims=True); Zn[Zn == 0.0] = 1.0
#         Z /= Zn
#         mask = Z[:, 2] < 0.0
#         if np.any(mask):
#             Y[mask] *= -1.0
#             Z[mask] *= -1.0
#         return Y, Z

#     @staticmethod
#     def embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=0.0):
#         stations_mm = np.asarray(stations_mm, float)
#         X_mm = np.asarray(X_mm, float)
#         Y_mm = np.asarray(Y_mm, float)

#         # cached frames (lru_cache on the Axis instance)
#         A, U = axis.frames_for_stations(tuple(np.round(stations_mm, 6)))

#         # basis in the normal plane
#         Yb, Zb = Axis._section_basis_from_tangent(U)

#         # optional twist
#         if np.any(np.asarray(twist_deg) != 0.0):
#             theta = np.deg2rad(np.asarray(twist_deg, float)).reshape(len(stations_mm))
#             Yb = Axis._rotate_vecs_about_axis(Yb, U, theta)
#             Zb = Axis._rotate_vecs_about_axis(Zb, U, theta)

#         # assemble + legacy 180° flip
#         P = (A[:, None, :] +
#              X_mm[:, :, None] * Yb[:, None, :] +
#              Y_mm[:, :, None] * Zb[:, None, :])
#         P = Axis._rotate_180_about_axis(P, A, U)
#         return P

#     def get_segment_for_station(self, station):
#         """
#         Find the segment containing the station and compute interpolation factor.
        
#         Args:
#             station (float): Station value in mm.
            
#         Returns:
#             tuple: (i, t) where i is the index of the segment's start point,
#                    and t is the interpolation factor (0 <= t <= 1).
#                    Returns (None, None) if station is out of range.
#         """
#         for i in range(len(self.stations) - 1):
#             if self.stations[i] <= station <= self.stations[i + 1]:
#                 t = (station - self.stations[i]) / (self.stations[i + 1] - self.stations[i]) if self.stations[i + 1] != self.stations[i] else 0
#                 return i, t
#         print(f"Warning: Station {station} mm is outside the range [{self.stations[0]}, {self.stations[-1]}] mm")
#         return None, None

#     def get_position_at_station(self, station):
#         """
#         Interpolate global position (X, Y, Z) at the given station in mm.
        
#         Args:
#             station (float): Station value in mm.
            
#         Returns:
#             tuple: (X, Y, Z) global coordinates in mm, or (0, 0, 0) if out of range.
#         """
#         i, t = self.get_segment_for_station(station)
#         if i is None:
#             return 0.0, 0.0, 0.0
#         x = self.x_coords[i] + t * (self.x_coords[i + 1] - self.x_coords[i])
#         y = self.y_coords[i] + t * (self.y_coords[i + 1] - self.y_coords[i])
#         z = self.z_coords[i] + t * (self.z_coords[i + 1] - self.z_coords[i])
#         return x, y, z

#     def get_plane_basis_at_station(self, station):
#         """
#         Compute orthonormal basis vectors for the plane perpendicular to the axis segment at the interpolated position.
        
#         Args:
#             station (float): Station value in mm.
            
#         Returns:
#             tuple: (u, v, tangent) where u and v are orthonormal vectors in the plane,
#                 and tangent is the segment's direction vector (unitless).
#                 Returns (None, None, None) if out of range or invalid segment.
#         """
#         i, t = self.get_segment_for_station(station)
#         if i is None:
#             return None, None, None

#         pos0 = np.array([self.x_coords[i], self.y_coords[i], self.z_coords[i]])
#         pos1 = np.array([self.x_coords[i + 1], self.y_coords[i + 1], self.z_coords[i + 1]])
#         position = pos0 + t * (pos1 - pos0)
#         tangent = pos1 - pos0
#         tangent_norm = np.linalg.norm(tangent)
#         if tangent_norm == 0:
#             print(f"Warning: Zero-length segment at station {station} mm")
#             return None, None, None
#         tangent = tangent / tangent_norm

#         # Use a reference vector perpendicular to the tangent, ensure v is positive Z
#         ref_vector = np.array([0, 0, 1]) if abs(np.dot(tangent, [0, 0, 1])) < 0.9 else np.array([0, 1, 0])
#         u = np.cross(tangent, ref_vector)
#         u = u / np.linalg.norm(u)
#         v = np.cross(u, tangent)  # Ensure v aligns with positive Z preference
#         v = v / np.linalg.norm(v)
#         if v[2] < 0:
#             v = -v  # Flip v to ensure positive Z component

#         return u, v, tangent

#     def transform_points(self, list_dicts_eval_points, stations, rotation_angle=0):
#         """
#         Transform lists of points at given stations to global coordinates in mm, with optional rotation around the longitudinal axis.
        
#         Args:
#             list_dicts_eval_points (list): List of lists of dictionaries with 'id', 'x', 'y', and possibly other keys,
#                                         where 'x' and 'y' are in mm.
#             stations (list): List of station values in mm corresponding to each point list.
#             rotation_angle (float, optional): Rotation angle in degrees around the longitudinal axis (default: 0).
            
#         Returns:
#             list: List of lists of dictionaries with all original keys plus 'x_global', 'y_global', 'z_global' keys,
#                 where 'x_global', 'y_global', 'z_global' are in mm.
#         """
#         if len(list_dicts_eval_points) != len(stations):
#             print(f"Warning: Number of point sets ({len(list_dicts_eval_points)}) does not match number of stations ({len(stations)})")
#             return []

#         transformed_points = []
#         for station, points in zip(stations, list_dicts_eval_points):
#             station_points = []
#             for point in points:
#                 try:
#                     global_x, global_y, global_z = self.transform_to_global(station, point['x'], point['y'], rotation_angle)
#                     # Create a new dictionary that copies all original keys and adds global coordinates
#                     transformed_point = dict(point)  # Shallow copy of the original point dictionary
#                     transformed_point.update({
#                         'x_global': float(global_x),  # Global X coordinate (in mm)
#                         'y_global': float(global_y),  # Global Y coordinate (in mm)
#                         'z_global': float(global_z)   # Global Z coordinate (in mm)
#                     })
#                     station_points.append(transformed_point)
#                 except (KeyError, TypeError) as e:
#                     print(f"Warning: Invalid point format at station {station} mm: {point}. Error: {e}")
#                     continue
#             transformed_points.append(station_points)
#         return transformed_points

#     def transform_to_global(self, station, local_x, local_y, rotation_angle=0):
#         """
#         Transform local (x, y) coordinates to global (X, Y, Z) coordinates in mm,
#         ensuring the plane is perpendicular to the axis segment at the station, with optional rotation around the longitudinal axis.
        
#         Args:
#             station (float): Station value in mm.
#             local_x (float): Local x coordinate in mm (maps to width direction in the plane).
#             local_y (float): Local y coordinate in mm (maps to height direction in the plane).
#             rotation_angle (float, optional): Rotation angle in degrees around the longitudinal axis (default: 0).
            
#         Returns:
#             tuple: (X, Y, Z) global coordinates in mm, or (0, 0, 0) if transformation fails.
#         """
#         # Get position and plane basis
#         origin_x, origin_y, origin_z = self.get_position_at_station(station)
#         u, v, tangent = self.get_plane_basis_at_station(station)
#         if u is None or v is None:
#             return 0.0, 0.0, 0.0

#         # Apply rotation to local coordinates
#         theta = math.radians(rotation_angle)  # Convert degrees to radians
#         cos_theta = math.cos(theta)
#         sin_theta = math.sin(theta)
#         # 2D rotation matrix: [cos θ, -sin θ; sin θ, cos θ]
#         x_rotated = local_x * cos_theta - local_y * sin_theta
#         y_rotated = local_x * sin_theta + local_y * cos_theta

#         # Debug basis and rotation
#         print(f"Station {station} mm: tangent={tangent}, u={u}, v={v}, rotation_angle={rotation_angle} deg")
#         print(f"Local coords: ({local_x}, {local_y}) -> Rotated local coords: ({x_rotated}, {y_rotated})")

#         # Transform rotated local coordinates to global
#         global_coords = (
#             origin_x + x_rotated * u[0] + y_rotated * v[0],
#             origin_y + x_rotated * u[1] + y_rotated * v[1],
#             origin_z + x_rotated * u[2] + y_rotated * v[2]
#         )
#         return global_coords

# def plot_axis_and_points_transformed(axis, global_points_sets, stations_to_plot):
#     """
#     Plot the axis and already transformed global points at specified stations in 3D.
    
#     Args:
#         axis (Axis): Axis object with stationing and coordinates in mm.
#         global_points_sets (list): List of lists of dictionaries with 'id', 'x_global', 'y_global', 'z_global'.
#         stations_to_plot (list): List of station values in mm to label each group.
    
#     Returns:
#         go.Figure: Plotly figure object.
#     """
#     if len(global_points_sets) != len(stations_to_plot):
#         print(f"Warning: Number of point sets ({len(global_points_sets)}) does not match number of stations ({len(stations_to_plot)})")
#         return go.Figure()

#     fig = go.Figure()

#     # Plot axis line
#     fig.add_trace(
#         go.Scatter3d(
#             x=axis.x_coords,
#             y=axis.y_coords,
#             z=axis.z_coords,
#             mode='lines+markers',
#             line=dict(color='black', width=4),
#             marker=dict(size=3, color='black'),
#             name='Axis'
#         )
#     )

#     # Plot each set of already transformed global points
#     for i, (station, global_points) in enumerate(zip(stations_to_plot, global_points_sets)):
#         if not global_points:
#             print(f"Warning: No points for station {station} mm")
#             continue

#         point_names = [p['id'] for p in global_points]
#         x_global = [p['x_global'] for p in global_points]
#         y_global = [p['y_global'] for p in global_points]
#         z_global = [p['z_global'] for p in global_points]

#         # Check for all-zero coordinates
#         if all(x == 0 and y == 0 and z == 0 for x, y, z in zip(x_global, y_global, z_global)):
#             print(f"Warning: All coordinates are (0, 0, 0) for station {station} mm")
#             continue

#         # Scatter points
#         fig.add_trace(
#             go.Scatter3d(
#                 x=x_global,
#                 y=y_global,
#                 z=z_global,
#                 mode='markers+text',
#                 text=point_names,
#                 textposition='top center',
#                 marker=dict(size=4, color='blue' if i == 0 else 'green', opacity=0.8),
#                 name=f'Station {station} mm Points'
#             )
#         )

#         # Loop line if points form a shape (only for sets with 3 or more points to form a valid loop)
#         if len(global_points) >= 3:
#             x_loop = x_global + [x_global[0]]  # Close the loop
#             y_loop = y_global + [y_global[0]]
#             z_loop = z_global + [z_global[0]]
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=x_loop,
#                     y=y_loop,
#                     z=z_loop,
#                     mode='lines',
#                     line=dict(color='red' if i == 0 else 'orange', width=2),
#                     name=f'Station {station} mm Loop'
#                 )
#             )
#         else:
#             print(f"Warning: Not enough points ({len(global_points)}) to form a loop at station {station} mm")

#     fig.update_layout(
#         title='3D Plot of Axis and Transformed Global Points (mm)',
#         scene=dict(
#             xaxis_title='X (mm)',
#             yaxis_title='Y (mm)',
#             zaxis_title='Z (mm)',
#             aspectmode='manual',
#             aspectratio=dict(x=1, y=1, z=1)
#         ),
#         showlegend=True,
#         template='plotly_white',
#         width=1200,
#         height=800
#     )

#     return fig

# def save_plot(fig, filename='axis_plot.html'):
#     """
#     Save the Plotly figure as an HTML file. Only auto-open the browser if
#     AUTO_OPEN_PLOT=1 is set in the environment to avoid blocking the run.
#     """
#     import os, webbrowser
#     try:
#         fig.write_html(filename, auto_open=False, include_plotlyjs='cdn')
#         print(f"Saved plot to {filename}")
#         if os.environ.get("AUTO_OPEN_PLOT") == "1":
#             file_url = f'file://{os.path.abspath(filename)}'
#             print(f"Opening {file_url} in default browser.")
#             webbrowser.open(file_url)
#     except Exception as e:
#         print(f"Error saving {filename}: {e}")

# # ---- Simple Axis factory cache (reuse identical axes across objects) ----
# # _AXIS_CACHE = {}

# # def _axis_signature(stations_m, x_m, y_m, z_m, units='m'):
# #     import numpy as np, hashlib
# #     # Use small hashes of arrays to keep memory low
# #     def h(a):
# #         a = np.asarray(a, dtype=float)
# #         m = hashlib.blake2b(a.tobytes(), digest_size=12)
# #         return m.hexdigest()
# #     return (h(stations_m), h(x_m), h(y_m), h(z_m), str(units))

# # def get_axis_cached(stations_m, x_m, y_m, z_m, units='m'):
# #     """
# #     Return a cached Axis with identical station/X/Y/Z arrays (and units), or create it.
# #     """
# #     sig = _axis_signature(stations_m, x_m, y_m, z_m, units)
# #     ax = _AXIS_CACHE.get(sig)
# #     if ax is None:
# #         ax = Axis(stations_m, x_m, y_m, z_m, units=units)
# #         # keep cache from growing unbounded
# #         if len(_AXIS_CACHE) > 128:
# #             _AXIS_CACHE.clear()
# #         _AXIS_CACHE[sig] = ax
# #     return ax


    



# if __name__ == "__main__":
#     # Axis data in meters (converted to mm internally)
#     # stations = [
#     #     0.0, 0.0, 0.0, 17.07146616, 30.25001202, 35.64154007, 40.37966326, 44.57551956,
#     #     48.33968332, 51.77837435, 54.98858306, 61.04079984, 66.92361742, 72.68890043,
#     #     83.724616, 93.76874931, 102.53174792, 110.44052051, 118.0344286, 125.62696359,
#     #     132.90491325, 136.30658391, 139.5177052, 142.55379955, 145.45846156, 148.25543104,
#     #     150.94322847, 155.89162008, 160.20606252, 164.55843999, 169.83656123, 173.10209351,
#     #     176.93842698, 181.50082272, 186.99074439, 193.64518536, 201.72270893
#     # ]
#     # x_coords = [
#     #     35.12741971, 29.52194867, 19.5449909, 7.66375165, -0.10279412, -4.8447777,
#     #     -8.9849704, -12.60351802, -15.77901765, -19.4592564, -23.47275142, -27.80651585,
#     #     -32.93943334, -38.28643754, -43.22623424, -46.72478212, -49.07361534, -50.80988934,
#     #     -52.44776618, -54.12827168, -55.90176428, -57.44756466, -59.2951689, -61.35243534,
#     #     -63.53576349, -66.4305168, -69.94049633, -73.97972611, -78.04983767, -81.68354956,
#     #     -85.08619842, -88.09233062, -91.320241, -94.70372066, -98.18268603, -100.29700096,
#     #     -101.36537046
#     # ]
#     # y_coords = [
#     #     -26.99148801, -27.98358686, -29.46701799, -30.15276335, -29.7896157, -29.19355608,
#     #     -28.29087415, -27.11543471, -25.70302937, -23.59796269, -20.51660336, -16.50753773,
#     #     -10.94717431, -3.74821282, 4.90571928, 13.11096103, 20.8597267, 28.36116823,
#     #     35.66692829, 41.53096788, 45.81337942, 48.64198087, 51.07979957, 53.14925925,
#     #     54.90203052, 56.83487865, 58.725155, 60.79295349, 63.04142256, 65.34087005,
#     #     67.67813807, 70.15234717, 73.4820699, 77.91687037, 83.70164785, 88.13699941,
#     #     90.61011916
#     # ]
#     # z_coords = [0.0] * len(stations)  # All Z coordinates are 0.0 meters

#     stations = [
#         0.0, 100, 200.0
#     ]
#     x_coords = [
#         0, 100, 200
#     ]
#     y_coords = [
#         0,100,100
#     ]
#     z_coords = [
#          0.0, 100, 200.0
#         ]




#     # Create Axis object (converts inputs to mm internally)
#     axis = Axis(stations, x_coords, y_coords, z_coords)

#     # Example points: local offsets in millimeters (mm)
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