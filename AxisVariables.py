# import numpy as np
# from scipy.interpolate import CubicSpline
# import bisect
# import matplotlib.pyplot as plt

# class AxisVariable:
#     def __init__(self, values, delta=0.0001, name=None, description=None):
#         self.raw_points = values
#         self.delta = delta
#         self.name = name  # Store variable name for identification
#         self.description = description  # Store variable description
#         self.xs = [v['X'] for v in self.raw_points]
#         self.ys = [v['Value'] for v in self.raw_points]
#         self.types = [v.get('Type', '#') for v in self.raw_points]
#         self._adjust_discontinuities()
#         self._prepare_segments()

#     def _adjust_discontinuities(self):
#         adjusted_xs = self.xs.copy()
#         for i in range(len(self.xs) - 1):
#             if abs(self.xs[i] - self.xs[i + 1]) < 1e-10:
#                 adjusted_xs[i] = self.xs[i] - self.delta
#                 adjusted_xs[i + 1] = self.xs[i] + self.delta
#         self.xs = adjusted_xs

#     def _prepare_segments(self):
#         self.segments = []
#         for i in range(len(self.raw_points) - 1):
#             x0 = self.xs[i]
#             y0 = self.ys[i]
#             x1 = self.xs[i + 1]
#             y1 = self.ys[i + 1]
#             is_discontinuity = abs(self.raw_points[i]['X'] - self.raw_points[i + 1]['X']) < 1e-10

#             if is_discontinuity:
#                 m = (y1 - y0) / (x1 - x0)
#                 b = y0 - m * x0
#                 fn = lambda x, m=m, b=b: m * x + b
#             else:
#                 interp_type = self.types[i]
#                 if abs(x0 - x1) < 1e-10:
#                     fn = lambda x, y=y0: y
#                 else:
#                     if interp_type == '#':
#                         m = (y1 - y0) / (x1 - x0)
#                         b = y0 - m * x0
#                         fn = lambda x, m=m, b=b: m * x + b
#                     elif interp_type == 'C':
#                         fn = lambda x, y=y0: y
#                     elif interp_type == '*':
#                         spline = CubicSpline([x0, x1], [y0, y1], bc_type=((1, 0), (2, 0)))
#                         fn = lambda x, spline=spline: float(spline(x))
#                     else:
#                         raise NotImplementedError(f"Interpolation type '{interp_type}' not supported.")
#             self.segments.append((x0, x1, fn))

#     def evaluate(self, station):
#         # Check if station matches an original X value
#         for i in range(len(self.raw_points) - 1, -1, -1):
#             if abs(station - self.raw_points[i]['X']) < 1e-6:
#                 return self.raw_points[i]['Value']

#         # Handle stations outside the defined range
#         if station < self.xs[0]:
#             return self.ys[0]  # Return the value at the first station
#         if station > self.xs[-1]:
#             return self.ys[-1]  # Return the value at the last station

#         # Handle stations within the defined range
#         idx = bisect.bisect_right(self.xs, station) - 1
#         if idx < 0 or idx >= len(self.segments):
#             return 0.0
#         x0, x1, fn = self.segments[idx]
#         return fn(station)

#     def plot(self, start=0, end=100, step=1):
#         stations = np.arange(start, end + 1, step)
#         values = np.array([self.evaluate(s) for s in stations])
#         plt.figure(figsize=(10, 6))
#         plt.scatter(self.xs, self.ys, color='red', zorder=5, label='Input Points')
#         discontinuity_indices = [
#             i for i in range(len(self.raw_points) - 1)
#             if abs(self.raw_points[i]['X'] - self.raw_points[i + 1]['X']) < 1e-10
#         ]
#         seg_start = 0
#         for idx in discontinuity_indices + [len(self.xs) - 1]:
#             segment_xs = self.xs[seg_start:idx + 1]
#             segment_ys = self.ys[seg_start:idx + 1]
#             plt.plot(segment_xs, segment_ys, color='blue', label='Interpolated Line' if seg_start == 0 else None, zorder=1)
#             seg_start = idx + 1
#         for i in discontinuity_indices:
#             plt.plot(
#                 [self.xs[i], self.xs[i + 1]],
#                 [self.ys[i], self.ys[i + 1]],
#                 color='purple',
#                 linestyle='dashed',
#                 linewidth=2,
#                 label='Discontinuity' if i == discontinuity_indices[0] else None,
#                 zorder=10
#             )
#         title = f'Axis Variable {self.name} Distribution (with Discontinuities): {self.description}' if self.name and self.description else f'Axis Variable {self.name or "Unknown"} Distribution (with Discontinuities)'
#         plt.title(title)
#         plt.xlabel('Station')
#         plt.ylabel('Value')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

#     @staticmethod
#     def create_axis_variables(var_list):
#         """
#         Create a list of AxisVariable objects from a list of variable definitions.
        
#         Args:
#             var_list (list): List of dictionaries, each containing 'VariableName', 'VariableStations',
#                             'VariableValues', 'VariableIntTypes', and 'VariableDescription'.
        
#         Returns:
#             list: List of AxisVariable objects, each with name and description attributes.
#         """
#         axis_variables = []
#         for var_data in var_list:
#             if not all(key in var_data for key in ['VariableName', 'VariableStations', 'VariableValues', 'VariableIntTypes']):
#                 continue  # Skip invalid entries
#             points = [
#                 {
#                     'X': float(station),
#                     'Value': float(value) if isinstance(value, (str, int, float)) and str(value).replace('.', '', 1).replace('-', '', 1).isdigit() else 0.0,
#                     'Type': interp_type
#                 }
#                 for station, value, interp_type in zip(
#                     var_data['VariableStations'],
#                     var_data['VariableValues'],
#                     var_data['VariableIntTypes']
#                 )
#             ]
#             axis_var = AxisVariable(
#                 points,
#                 name=var_data['VariableName'],
#                 description=var_data.get('VariableDescription', '')
#             )
#             axis_variables.append(axis_var)
#         return axis_variables

#     @staticmethod
#     def evaluate_at_stations(axis_variables, stations):
#         """
#         Evaluate a list of AxisVariable objects at given stations.
        
#         Args:
#             axis_variables (list): List of AxisVariable objects.
#             stations (list): List of station values to evaluate at.
        
#         Returns:
#             list: List of dictionaries, each containing variable names and their evaluated values at each station.
#         """
#         results = []
#         for station in stations:
#             station_dict = {}
#             for axis_var in axis_variables:
#                 if axis_var.name:
#                     station_dict[axis_var.name] = axis_var.evaluate(station)
#             results.append(station_dict)
#         return results
    
#     from functools import lru_cache
#     import numpy as np

#     def _hash_axis_var_list(axis_variables):
#         # Stable identity for caching: (id per AxisVariable object)
#         return tuple(id(v) for v in axis_variables)

#     @lru_cache(maxsize=256)
#     def _evaluate_single_axis_var_on_tuple(axis_var_id, stations_tuple):
#         # Helper for the cache above; resolves the object back from id.
#         # We keep a side table in the outer function.
#         raise RuntimeError("Do not call directly")

#     def evaluate_at_stations_cached(axis_variables, stations):
#         """
#         Cached & fast version of AxisVariable.evaluate_at_stations.
#         Produces the same structure as your original staticmethod.
#         """
#         stations = np.asarray(stations, dtype=float)
#         results = []

#         # build a side-table to map ids -> objects (to use inside inner cached calls)
#         id_to_var = {id(v): v for v in axis_variables}
#         var_ids = tuple(id_to_var.keys())
#         stations_tuple = tuple(float(s) for s in stations.tolist())

#         # Monkey-patch the cached worker with closures that see id_to_var
#         def _worker(axis_var_id, stations_tuple_):
#             v = id_to_var[axis_var_id]
#             vals = [v.evaluate(s) for s in stations_tuple_]
#             return vals

#         # Bind once per call
#         global _evaluate_single_axis_var_on_tuple
#         def _evaluate_single_axis_var_on_tuple(axis_var_id, stations_tuple):
#             return _worker(axis_var_id, stations_tuple)

#         # Evaluate all variables; reuse per-(var,stations) results via the cache
#         evaluated = {}
#         for vid in var_ids:
#             evaluated[vid] = _evaluate_single_axis_var_on_tuple(vid, stations_tuple)

#         # stitch back to your original list-of-dicts structure
#         for idx, s in enumerate(stations_tuple):
#             row = {}
#             for v in axis_variables:
#                 if v.name:
#                     row[v.name] = evaluated[id(v)][idx]
#             results.append(row)
#         return results

# # Test code
# if __name__ == "__main__":
#     # Example input in the new format
#     test_data = [
#         {
#             "VariableName": "CSHEIGHT",
#             "VariableStations": [0, 5, 10],
#             "VariableValues": ["5", "20", "1"],
#             "VariableIntTypes": ["#", "#", "#"],
#             "VariableDescription": "CSHEIGHT:Axis Variable"
#         },
#         {
#             "VariableName": "WIDH_1000",
#             "VariableStations": [0, 25, 50],
#             "VariableValues": ["100", "0", "100"],
#             "VariableIntTypes": ["#", "#", "#"],
#             "VariableDescription": "WIDH_1000:Axis Variable"
#         }
#     ]

#     # Create AxisVariable objects
#     axis_vars = AxisVariable.create_axis_variables(test_data)

#     # Verify descriptions
#     for axis_var in axis_vars:
#         print(f"Variable: {axis_var.name}, Description: {axis_var.description}")

#     # Evaluate at specific stations
#     stations = list(range(-10, 60, 1))  # Extended range to test extrapolation
#     results = AxisVariable.evaluate_at_stations(axis_vars, stations)

#     # Print results
#     for i, result in enumerate(results):
#         print(f"Station {stations[i]}: {result}")

#     # Plot each variable
#     for axis_var in axis_vars:
#         axis_var.plot()
#     print("Plotting is disabled for this example. Uncomment the plot section to visualize.")