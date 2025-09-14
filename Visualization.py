# # vis_renderer.py — centralizes ALL plotting, layout, export, and build_point_matrices.

# from __future__ import annotations
# from typing import List, Dict, Any, Optional, Tuple
# import os, sys, json, webbrowser
# import numpy as np
# import plotly.graph_objects as go
# from jinja2 import Template
# from Axis import Axis  # for embed_points_to_global_mm


# class VisRenderer:
#     """
#     Visualization + orchestration:
#       • build_point_matrices(...) — calls your compute helpers (passed in) to avoid circular imports
#       • get_plot_traces_matrix(...) — axis, first-station points, cross-section loops, longitudinal lines
#       • apply_equal_ranges(...) — equal ranges & scene layout
#       • save_as_website(...) — HTML with right-side metadata panel
#     """

#     # --------------------------- BUILD (compute orchestrator) ---------------------------
#     def build_point_matrices(
#         self,
#         axis,
#         json_data: Dict[str, Any],
#         results: List[Dict[str, Any]],
#         stations_m: List[float],
#         *,
#         twist_deg: float = 0.0,
#         helpers: Dict[str, Any],
#     ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
#         """
#         Returns:
#           ids:         list[str]                length N
#           stations_mm: (S,) float               filtered to axis range (mm)
#           P_mm:        (S,N,3) float            global coords in mm
#           X_mm:        (S,N) float              local-Y in mm (naming per your pipeline)
#           Y_mm:        (S,N) float              local-Z in mm
#           loops_idx:   list[list[int]]          index loops over ids
#         """
#         # Unpack helpers (accept both 'fix_var_units_inplace' and '_fix_var_units_inplace', etc.)
#         collect_used_variable_names   = helpers.get("collect_used_variable_names") or helpers["Utility.collect_used_variable_names"]  # type: ignore
#         build_var_arrays_from_results = helpers.get("build_var_arrays_from_results") or helpers["Utility.build_var_arrays_from_results"]  # type: ignore
#         fix_var_units_inplace         = helpers.get("fix_var_units_inplace") or helpers.get("_fix_var_units_inplace") or helpers["fix_var_units_inplace"]  # type: ignore
#         prepare_point_solver_cached   = helpers.get("prepare_point_solver_cached") or helpers["prepare_point_solver_cached"]  # type: ignore
#         get_point_coords_vectorized   = helpers.get("get_point_coords_vectorized") or helpers["Utility.get_point_coords_vectorized"]  # type: ignore

#         # 1) Keep-mask stations to axis range (apply the same mask to results)
#         stations_mm_all = np.asarray(stations_m, dtype=float) * 1000.0
#         smin, smax = float(np.min(axis.stations)), float(np.max(axis.stations))
#         keep = (stations_mm_all >= smin) & (stations_mm_all <= smax)
#         if not np.any(keep):
#             empty = np.array([], dtype=float)
#             return [], empty, np.zeros((0, 0, 3), float), np.zeros((0, 0), float), np.zeros((0, 0), float), []

#         stations_mm = stations_mm_all[keep]
#         kept_results = [results[i] for i, k in enumerate(keep) if k]

#         # 2) Build variable arrays & solve section in 2D (vectorized)
#         variables   = (json_data.get('Variables') or {})
#         used_names  = collect_used_variable_names(json_data)
#         var_arrays  = build_var_arrays_from_results(kept_results, variables, keep=used_names)
#         fix_var_units_inplace(var_arrays)

#         pre = prepare_point_solver_cached(json_data)
#         ids, X_mm, Y_mm = get_point_coords_vectorized(json_data, var_arrays, precomputed=pre)  # shapes (S,N)

#         # 3) Embed to global 3D (mm) and apply axis twist/flip per Axis implementation
#         P_mm = Axis.embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=twist_deg)

#         # 4) Build loop indices (by ids) from JSON (support Geometry/Loops and top-level Loops)
#         loops_idx = self._loops_idx_from_json(json_data, ids)
#         return ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx

#     # ----------------------------- helpers (shared) -----------------------------
#     @staticmethod
#     def _clean_numbers(seq):
#         out = []
#         if not seq:
#             return out
#         for v in seq:
#             if v is None:
#                 continue
#             try:
#                 fv = float(v)
#             except (TypeError, ValueError):
#                 continue
#             if np.isfinite(fv):
#                 out.append(fv)
#         return out

#     @staticmethod
#     def _extend_numeric(dst, src):
#         if not src:
#             return
#         for v in src:
#             if v is None:
#                 dst.append(None)
#             else:
#                 try:
#                     fv = float(v)
#                     if np.isfinite(fv):
#                         dst.append(fv)
#                 except (TypeError, ValueError):
#                     continue

#     @staticmethod
#     def _clean_json_lines(meta_obj) -> List[str]:
#         """Pretty-print object metadata as a list of lines for the right panel."""
#         try:
#             txt = json.dumps(meta_obj, indent=2, ensure_ascii=False)
#         except Exception:
#             return [str(meta_obj)]
#         lines = []
#         for raw in txt.splitlines():
#             s = raw.strip()
#             if s.endswith(','):
#                 s = s[:-1]
#             if s.startswith('"') and '":' in s:
#                 k, rest = s.split('":', 1)
#                 k = k.strip('"')
#                 s = f"{k}:{rest}"
#             if s:
#                 lines.append(s)
#         return lines or ["(no metadata)"]

#     @staticmethod
#     def _loops_idx_from_json(json_data: Dict[str, Any], ids: List[str]) -> List[List[int]]:
#         loops = (json_data or {}).get('Loops')
#         if loops is None:
#             geom = (json_data or {}).get('Geometry') or {}
#             loops = geom.get('Loops') or []
#         id_to_col = {pid: j for j, pid in enumerate(ids)}
#         out: List[List[int]] = []
#         for loop in loops or []:
#             pts = loop.get('Points', []) if isinstance(loop, dict) else []
#             idxs = [id_to_col.get(p.get('Id')) for p in pts if isinstance(p, dict)]
#             idxs = [ix for ix in idxs if ix is not None]
#             if len(idxs) >= 2:
#                 out.append(idxs)
#         return out

#     # ----------------------------- trace builder -----------------------------
#     def get_plot_traces_matrix(
#         self,
#         axis,
#         json_data,
#         stations_mm,      # (S,) mm
#         ids,              # N
#         P_mm,             # (S,N,3) mm
#         X_mm=None,        # (S,N) local-Y mm  (naming retained: X_mm == localY)
#         Y_mm=None,        # (S,N) local-Z mm  (naming retained: Y_mm == localZ)
#         obj_name: str = "Object",
#         colors: Optional[Dict[str, str]] = None,
#         cls_obj: Optional[object] = None,
#         show_labels: bool = False,
#     ) -> Tuple[List[go.Scatter3d], List[float], List[float], List[float]]:
#         """
#         Metadata format (to match your panel):
#           • First-station points, loops: dict with keys
#             { type, obj, id, station_m, localY_m, localZ_m, globalX_m, globalY_m, globalZ_m }
#             (note: localY_m := X_mm/1000; localZ_m := Y_mm/1000)
#           • Longitudinal: list[str] with lines:
#             ["X: .... m", "Y: .... m", "Z: .... m", "—", *object_metadata_lines]
#         """
#         default_colors = {
#             'axis': 'black',
#             'first_station_points': 'blue',
#             'cross_section_loops': 'red',
#             'longitudinal_lines': 'gray',
#         }
#         colors = {**default_colors, **(colors or {})}
#         traces: List[go.Scatter3d] = []

#         S = int(np.asarray(stations_mm).size)
#         if P_mm.ndim != 3:
#             raise ValueError("P_mm must be (S,N,3)")
#         N = P_mm.shape[1]

#         # Convert to meters for plotting/hover
#         P_m = P_mm / 1000.0
#         localY_m = (X_mm / 1000.0) if X_mm is not None else None  # naming per your convention
#         localZ_m = (Y_mm / 1000.0) if Y_mm is not None else None
#         stations_m = np.asarray(stations_mm, dtype=float) / 1000.0

#         # Axis trace
#         axis_x_m = (np.asarray(axis.x_coords, float) / 1000.0).tolist()
#         axis_y_m = (np.asarray(axis.y_coords, float) / 1000.0).tolist()
#         axis_z_m = (np.asarray(axis.z_coords, float) / 1000.0).tolist()
#         axis_meta = [{"type": "axis", "obj": obj_name, "station_m": s}
#                      for s in (np.asarray(axis.stations, float) / 1000.0).tolist()]
#         traces.append(
#             go.Scatter3d(
#                 x=axis_x_m, y=axis_y_m, z=axis_z_m,
#                 mode='lines',
#                 line=dict(color=colors['axis'], width=3, dash='dash'),
#                 name=f'{obj_name} Axis',
#                 customdata=axis_meta,
#                 hovertemplate="<b>%{customdata.obj}</b><br>Axis @ %{customdata.station_m:.3f} m<extra></extra>"
#             )
#         )

#         if S == 0 or N == 0:
#             return traces, axis_x_m, axis_y_m, axis_z_m

#         # First-station points (dict metadata with localY/localZ and global XYZ)
#         P0 = P_m[0]  # (N,3)
#         valid0 = ~np.isnan(P0).any(axis=1)
#         x0 = P0[valid0, 0].tolist()
#         y0 = P0[valid0, 1].tolist()
#         z0 = P0[valid0, 2].tolist()
#         ids0 = [ids[j] for j, ok in enumerate(valid0) if ok]
#         st0 = float(stations_m[0])

#         meta0 = []
#         for j, pid in enumerate(ids):
#             if not valid0[j]:
#                 continue
#             ly = None if localY_m is None or np.isnan(localY_m[0, j]) else float(localY_m[0, j])
#             lz = None if localZ_m is None or np.isnan(localZ_m[0, j]) else float(localZ_m[0, j])
#             meta0.append({
#                 "type": "point",
#                 "obj": obj_name,
#                 "id": pid,
#                 "station_m": st0,
#                 "localY_m": ly,
#                 "localZ_m": lz,
#                 "globalX_m": float(P0[j, 0]),
#                 "globalY_m": float(P0[j, 1]),
#                 "globalZ_m": float(P0[j, 2]),
#             })

#         traces.append(
#             go.Scatter3d(
#                 x=x0, y=y0, z=z0,
#                 mode='markers+text' if show_labels else 'markers',
#                 marker=dict(size=4, opacity=0.9, color=colors['first_station_points']),
#                 text=ids0 if show_labels else None,
#                 textposition='top center',
#                 name=f'{obj_name} Points @ {st0:.3f} m',
#                 customdata=meta0,
#                 hovertemplate=(
#                     "<b>%{customdata.obj}</b><br>"
#                     "Point: %{customdata.id}<br>"
#                     "Station: %{customdata.station_m:.3f} m<br>"
#                     "Local Y: %{customdata.localY_m:.3f} m<br>"
#                     "Local Z: %{customdata.localZ_m:.3f} m<br>"
#                     "X: %{customdata.globalX_m:.3f} m<br>"
#                     "Y: %{customdata.globalY_m:.3f} m<br>"
#                     "Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
#                 )
#             )
#         )

#         # Cross-section loops (dict metadata per vertex)
#         loops_idx = self._loops_idx_from_json(json_data, ids)
#         loop_x: List[Optional[float]] = []
#         loop_y: List[Optional[float]] = []
#         loop_z: List[Optional[float]] = []
#         loop_meta: List[Optional[Dict[str, Any]]] = []

#         def _emit_loop_run(seg_xyz: np.ndarray, run_idx: List[int], st_m: float, si: int):
#             xs = seg_xyz[run_idx, 0].tolist()
#             ys = seg_xyz[run_idx, 1].tolist()
#             zs = seg_xyz[run_idx, 2].tolist()
#             # close if contiguous full loop
#             if len(run_idx) >= 2 and (run_idx[0] != run_idx[-1]):
#                 xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])
#             for k in range(len(xs)):
#                 j = run_idx[min(k, len(run_idx) - 1)]
#                 pid = ids[j]
#                 ly = None if localY_m is None or np.isnan(localY_m[si, j]) else float(localY_m[si, j])
#                 lz = None if localZ_m is None or np.isnan(localZ_m[si, j]) else float(localZ_m[si, j])
#                 loop_x.append(xs[k]); loop_y.append(ys[k]); loop_z.append(zs[k])
#                 loop_meta.append({
#                     "type": "loop",
#                     "obj": obj_name,
#                     "id": pid,
#                     "station_m": st_m,
#                     "localY_m": ly,
#                     "localZ_m": lz,
#                     "globalX_m": xs[k],
#                     "globalY_m": ys[k],
#                     "globalZ_m": zs[k],
#                 })
#             loop_x.append(None); loop_y.append(None); loop_z.append(None); loop_meta.append(None)

#         for si in range(S):
#             st_m = float(stations_m[si])
#             seg = P_m[si]  # (N,3)
#             valid = ~np.isnan(seg).any(axis=1)
#             if not valid.any():
#                 continue
#             if loops_idx:
#                 for lp in loops_idx:
#                     run = [j for j in lp if valid[j]]
#                     if len(run) >= 2:
#                         _emit_loop_run(seg, run, st_m, si)
#             else:
#                 run = [j for j, ok in enumerate(valid) if ok]
#                 if len(run) >= 2:
#                     _emit_loop_run(seg, run, st_m, si)

#         if loop_x:
#             traces.append(
#                 go.Scatter3d(
#                     x=loop_x, y=loop_y, z=loop_z,
#                     mode='lines',
#                     line=dict(color=colors['cross_section_loops'], width=2),
#                     name=f'{obj_name} Cross Sections',
#                     customdata=loop_meta,
#                     hovertemplate=(
#                         "<b>%{customdata.obj}</b><br>"
#                         "Point: %{customdata.id}<br>"
#                         "Station: %{customdata.station_m:.3f} m<br>"
#                         "Local: (%{customdata.localY_m:.3f} m, %{customdata.localZ_m:.3f} m)<br>"
#                         "Global: (%{customdata.globalX_m:.3f}, %{customdata.globalY_m:.3f}, %{customdata.globalZ_m:.3f}) m"
#                         "<extra></extra>"
#                     )
#                 )
#             )

#         # Longitudinal (list[str] metadata: X/Y/Z first, then object meta lines)
#         long_x: List[Optional[float]] = []
#         long_y: List[Optional[float]] = []
#         long_z: List[Optional[float]] = []
#         long_meta: List[Optional[List[str]]] = []

#         obj_lines = self._clean_json_lines(getattr(cls_obj, "get_object_metada", lambda: {})())

#         for j, _pid in enumerate(ids):
#             col = P_m[:, j, :]  # (S,3)
#             valid = ~np.isnan(col).any(axis=1)
#             if not valid.any():
#                 continue
#             first = True
#             for k in range(S):
#                 if valid[k]:
#                     if first:
#                         first = False
#                     else:
#                         prev = k - 1
#                         while prev >= 0 and not valid[prev]:
#                             prev -= 1
#                         if prev >= 0:
#                             p1, p2 = col[prev], col[k]
#                             long_x.extend([p1[0], p2[0], None])
#                             long_y.extend([p1[1], p2[1], None])
#                             long_z.extend([p1[2], p2[2], None])
#                             cmn1 = [f"X: {p1[0]:.3f} m", f"Y: {p1[1]:.3f} m", f"Z: {p1[2]:.3f} m", "—"] + obj_lines
#                             cmn2 = [f"X: {p2[0]:.3f} m", f"Y: {p2[1]:.3f} m", f"Z: {p2[2]:.3f} m", "—"] + obj_lines
#                             long_meta.extend([cmn1, cmn2, None])

#         if long_x:
#             traces.append(
#                 go.Scatter3d(
#                     x=long_x, y=long_y, z=long_z,
#                     mode='lines',
#                     line=dict(color=colors['longitudinal_lines'], width=1),
#                     name=f'{obj_name} Longitudinal',
#                     customdata=long_meta,
#                     hovertemplate=(obj_name + " Longitudinal<br>"
#                                    "X: %{x:.3f} m<br>"
#                                    "Y: %{y:.3f} m<br>"
#                                    "Z: %{z:.3f} m<extra></extra>"),
#                 )
#             )

#         # All coords for layout (meters; may include None)
#         all_x_m = axis_x_m + x0 + [v for v in long_x if v is not None] + [v for v in loop_x if v is not None]
#         all_y_m = axis_y_m + y0 + [v for v in long_y if v is not None] + [v for v in loop_y if v is not None]
#         all_z_m = axis_z_m + z0 + [v for v in long_z if v is not None] + [v for v in loop_z if v is not None]
#         return traces, all_x_m, all_y_m, all_z_m

#     # ----------------------------- layout -----------------------------
#     def apply_equal_ranges(self, fig: go.Figure, all_x: list, all_y: list, all_z: list):
#         all_x = self._clean_numbers(all_x); all_y = self._clean_numbers(all_y); all_z = self._clean_numbers(all_z)
#         if all_x and all_y and all_z:
#             min_x, max_x = min(all_x), max(all_x)
#             min_y, max_y = min(all_y), max(all_y)
#             min_z, max_z = min(all_z), max(all_z)
#             cx, cy, cz = (min_x + max_x)/2.0, (min_y + max_y)/2.0, (min_z + max_z)/2.0
#             rx, ry, rz = (max_x - min_x), (max_y - min_y), (max_z - min_z)
#             r = max(rx, ry, rz)/2.0 or 1.0
#             x_range = [cx - r, cx + r]
#             y_range = [cy - r, cy + r]
#             z_range = [cz - r, cz + r]
#         else:
#             x_range = y_range = z_range = [-1, 1]

#         fig.update_layout(
#             title='3D Scatter Plot (vectorized, with metadata & magnetic hover)',
#             scene=dict(
#                 xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
#                 xaxis=dict(range=x_range), yaxis=dict(range=y_range), zaxis=dict(range=z_range),
#                 aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
#                 camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))
#             ),
#             showlegend=True, template='plotly_white',
#             margin=dict(l=0, r=0, t=50, b=0), hovermode='closest',
#             hoverdistance=-1, spikedistance=-1, uirevision='keep',
#         )
#         fig.update_scenes(xaxis=dict(showspikes=True), yaxis=dict(showspikes=True), zaxis=dict(showspikes=True))

#     # ----------------------------- export -----------------------------
#     def save_as_website(self, fig: go.Figure, file_path: str, open_browser: bool = True):
#         """
#         Save plot + right-side metadata panel.
#         If 'file_path' has no .html, write '<file_path>_3d.html'.
#         """
#         html_file = file_path if file_path.lower().endswith('.html') else f"{file_path}_3d.html"

#         template_str = """
# <!DOCTYPE html>
# <html lang="en"><head>
# <meta charset="utf-8" />
# <meta name="viewport" content="width=device-width, initial-scale=1.0" />
# <title>3D Plot</title>
# <script src="resources/plotly-latest.min.js"></script>
# <script src="_internal/resources/plotly-latest.min.js"></script>
# <style>
#   html,body{margin:0;height:100%}
#   body{display:flex;flex-direction:row;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}
#   #plot-container{flex:2;min-width:320px;height:100vh;background:#f5f6f7}
#   #plot{width:100%;height:100%}
#   #json-panel{flex:1.4;min-width:240px;max-width:560px;padding:14px;background:#fff;border-left:1px solid #ddd;overflow-y:auto}
#   h3{margin:0 0 10px 0;font-size:15px;font-weight:600}
#   pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.35}
#   .hint{color:#666;font-size:12px;margin-top:6px}
#   @media (max-width: 900px){
#     body{flex-direction:column}
#     #plot-container{height:60vh}
#     #json-panel{height:40vh;border-left:none;border-top:1px solid #ddd}
#   }
# </style>
# </head>
# <body>
#   <div id="plot-container"><div id="plot"></div></div>
#   <div id="json-panel">
#     <h3>Hover to view metadata</h3>
#     <pre id="json-output">No item selected.</pre>
#     <div class="hint">Click also sets the panel to that item.</div>
#   </div>

# <script>
# function pretty(x){ try { return JSON.stringify(x, null, 2); } catch(e){ return String(x); } }

# function bootPlot(){
#   var plotDiv = document.getElementById('plot');
#   var figure  = {{ fig | tojson }};
#   Plotly.newPlot(plotDiv, figure.data, figure.layout, {responsive:true}).then(function(){
#     var panel = document.getElementById('json-output');
#     function onEvt(evt){
#       if (!evt || !evt.points || !evt.points.length) return;
#       var p = evt.points[0];
#       var payload = (p.customdata !== undefined) ? p.customdata 
#                     : { trace: (p.fullData && p.fullData.name) || '(unknown)', x: p.x, y: p.y, z: p.z };
#       panel.textContent = (typeof payload === 'object') ? pretty(payload) : String(payload);
#     }
#     plotDiv.on('plotly_hover',  onEvt);
#     plotDiv.on('plotly_click',  onEvt);
#   }).catch(function(err){
#     console.error(err);
#     document.getElementById('json-output').textContent = "Plotly error. See console.";
#   });
# }

# // Try local -> internal -> CDN
# (function ensurePlotly(){
#   if (window.Plotly) return bootPlot();

#   function load(src, onload, onerror){
#     var s = document.createElement('script');
#     s.src = src; s.async = false;
#     s.onload = onload; s.onerror = onerror;
#     document.head.appendChild(s);
#   }

#   load('resources/plotly-latest.min.js', bootPlot, function(){
#     load('_internal/resources/plotly-latest.min.js', bootPlot, function(){
#       load('https://cdn.plot.ly/plotly-2.27.0.min.js', bootPlot, function(){
#         document.getElementById('json-output').textContent = "Could not load Plotly (local or CDN).";
#       });
#     });
#   });
# })();
# </script>

# </body></html>
#         """
#         try:
#             template = Template(template_str)
#             fig_dict = fig.to_dict()
#             final_html = template.render(fig=fig_dict)
#             with open(html_file, 'w', encoding='utf-8') as f:
#                 f.write(final_html)
#             print(f"Saved 3D plot with metadata panel to {html_file}")
#             if open_browser:
#                 url = 'file://' + os.path.abspath(html_file)
#                 print(f"Opening {url}")
#                 webbrowser.open(url)
#         except Exception as e:
#             print(f"Error saving {html_file}: {e}")
