# models/vis_adapter.py
from __future__ import annotations
from typing import Dict, List, Optional
from .mapping import mapping
from .axis import Axis
from .cross_section import CrossSection
from .axis_variable import AxisVariable

# ——— Helpers you already had (trimmed to essentials you showed) ———

def build_viso_object(
    obj,                         # BaseObject (or subclass)
    ctx,                         # VisoContext | None
    *,
    axis: Optional[Axis] = None,
    mainstations: Optional[dict] = None,        # keep as dict mapping if you like
    cross_sections_override: Optional[List[CrossSection]] = None,
    mapping_cfg: Dict[type, Dict[str, str]] | None = None,
) -> dict:
    """
    Resolve dependencies for `obj` and return the classic vis dict
    via create_input_for_visualisation. Pure function (no I/O).
    """
    mapping_cfg = mapping_cfg or mapping

    # Axis: explicit override > ctx by name > existing obj.axis_obj
    ax = axis or getattr(obj, "axis_obj", None)
    if ax is None:
        ax_name = getattr(obj, "axis_name", None) or getattr(obj, "object_axis_name", None)
        if ctx is not None and ax_name:
            ax = ctx.get_axis(ax_name)
    if ax is not None:
        obj.axis_obj = ax

    # (optional) mainstations
    if mainstations is not None:
        setattr(obj, "_mainstations", mainstations)

    # Cross-sections (objects, not filenames)
    if cross_sections_override is not None:
        obj._cross_sections = list(cross_sections_override)
    elif ctx is not None:
        # resolve by NCS/name using the context helpers
        obj._cross_sections = obj._resolve_cross_sections_from_ncs(ctx)
    else:
        obj._cross_sections = getattr(obj, "_cross_sections", None)

    # Axis variables: ensure AxisVariable objects exist (your BaseObject handles this)
    if hasattr(obj, "axis_variables") and isinstance(obj.axis_variables, list):
        obj.set_axis_variables(mapping_cfg.get(AxisVariable, {}))

    return create_input_for_visualisation(
        obj=obj,
        axis_data=None,
        cross_section_objects=obj._cross_sections,
        json_file_path=None,
        axis_rotation=getattr(obj, "axis_rotation", 0),
        colors=getattr(obj, "colors", None),
    )

def create_input_for_visualisation(
    obj: object,
    *,
    axis_data: list | None = None,
    cross_section_objects: list | None = None,
    json_file_path: str | None = None,
    axis_rotation: float | None = None,
    colors: dict | None = None,
) -> dict:
    """
    Your exact routine, unchanged in behavior. Left compact here;
    keep your full version from the message in this module.
    """
    import hashlib, numpy as np
    from .mapping import mapping as _mapping
    from .axis import Axis

    colors = colors or getattr(obj, 'colors', {
        "axis": "black",
        "first_station_points": "blue",
        "cross_section_loops": "red",
        "longitudinal_lines": "gray"
    })
    axis_rotation = axis_rotation if axis_rotation is not None else getattr(obj, 'axis_rotation', 0)

    def _to_list(x):
        try:
            if isinstance(x, np.ndarray):
                return [float(v) for v in x.tolist()]
        except Exception:
            pass
        if x is None or x == "": return []
        if isinstance(x, (list, tuple, set)): return [float(v) for v in x]
        if isinstance(x, (int, float)): return [float(x)]
        return []

    # breakpoints → stations_to_plot (reuse your interpolation rules)
    station_value_main  = _to_list(getattr(obj, 'station_value', None))
    station_value_inner = _to_list(getattr(obj, 'internal_station_value', None))
    user_stations       = _to_list(getattr(obj, 'user_stations', None))
    base_breaks = sorted(set(station_value_main + station_value_inner + user_stations))

    def interpolate_segments(breakpoints):
        if len(breakpoints) < 2: return breakpoints
        out = []
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            dist = b - a
            if dist <= 0: continue
            if   dist <   5: steps = 2
            elif dist <  10: steps = 3
            elif dist <  50: steps = 10
            elif dist < 100: steps = 20
            elif dist < 200: steps = 40
            elif dist < 300: steps = 30
            elif dist < 400: steps = 40
            elif dist < 500: steps = 50
            else:             steps = 500
            step = dist / steps
            out.extend(round(a + j * step, 8) for j in range(steps))
        out.append(round(breakpoints[-1], 8))
        return out

    stations_to_plot = interpolate_segments(base_breaks)

    # Axis from obj.axis_obj if present:
    def _as_float_list(val):
        if val is None: return []
        if isinstance(val, np.ndarray): val = val.ravel().tolist()
        elif isinstance(val, (list, tuple)): val = list(val)
        else: val = [val]
        out = []
        for v in val:
            try: out.append(float(v))
            except Exception: pass
        return out

    if getattr(obj, "axis_obj", None) is not None:
        ax = obj.axis_obj
        stations = _as_float_list(getattr(ax, "stations", None))
        x_coords = _as_float_list(getattr(ax, "x_coords", None))
        y_coords = _as_float_list(getattr(ax, "y_coords", None))
        z_coords = _as_float_list(getattr(ax, "z_coords", None))
    else:
        # raw axis_data fallback (keep as in your version)
        from .mapping import mapping
        axis_map = mapping.get(Axis, {})
        class_key    = axis_map.get('class', 'Class')
        name_key     = axis_map.get('name', 'Name')
        stations_key = axis_map.get('stations', 'StaionValue')
        x_key        = axis_map.get('x_coords', 'CurvCoorX')
        y_key        = axis_map.get('y_coords', 'CurvCoorY')
        z_key        = axis_map.get('z_coords', 'CurvCoorZ')
        stations = x_coords = y_coords = z_coords = []
        axis_dict = None
        axis_name = (getattr(obj, 'axis_name', '') or '').lower()
        for data in (axis_data or []):
            if (data.get(class_key) == 'Axis' and str(data.get(name_key, '')).lower() == axis_name):
                axis_dict = data; break
        if axis_dict:
            stations = _as_float_list(axis_dict.get(stations_key))
            x_coords = _as_float_list(axis_dict.get(x_key))
            y_coords = _as_float_list(axis_dict.get(y_key))
            z_coords = _as_float_list(axis_dict.get(z_key))

    # AxisVariables payload (unchanged)
    axis_variables = [
        {
            "VariableName": var.name,
            "VariableStations": var.xs,
            "VariableValues": [str(v * 1000) for v in var.ys],   # mm convention
            "VariableIntTypes": var.types,
            "VariableDescription": "empty",
        }
        for var in getattr(obj, 'axis_variables_obj', [])
    ]

    # Cross-section JSON path resolution (keep your logic)
    final_json_file = ""
    ncs_list = _to_list(getattr(obj, 'cross_section_ncs', None)) or _to_list(getattr(obj, 'internal_cross_section_ncs', None))
    if cross_section_objects:
        cs_by_ncs  = {getattr(cs, 'ncs', None): cs for cs in cross_section_objects if getattr(cs, 'ncs', None) is not None}
        cs_by_name = {str(getattr(cs, 'name', '')).lower(): cs for cs in cross_section_objects}
        for ncs in ncs_list:
            cs = cs_by_ncs.get(ncs)
            if cs and getattr(cs, 'json_name', None):
                final_json_file = cs.json_name[0]
                break
        if not final_json_file:
            wanted = None
            cname = getattr(obj, 'class_name', '')
            if   cname == 'DeckObject':       wanted = 'MASTER_Deck'
            elif cname == 'PierObject':       wanted = 'MASTER_Pier'
            elif cname == 'FoundationObject': wanted = 'MASTER_Foundation'
            if wanted:
                cs = cs_by_name.get(wanted.lower())
                if cs and getattr(cs, 'json_name', None):
                    final_json_file = cs.json_name[0]

    

    def tiny_hash(arr):
        data = (','.join(str(float(v)) for v in (arr or []))).encode('utf-8')
        import hashlib
        return hashlib.blake2b(data, digest_size=12).hexdigest()

    axis_sig = (tiny_hash(stations), tiny_hash(x_coords), tiny_hash(y_coords), tiny_hash(z_coords), 'm')

    return {
        "json_file": final_json_file or "",
        "stations_axis": stations,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "stations_to_plot": stations_to_plot,
        "AxisVariables": axis_variables,
        "name": getattr(obj, 'name', ''),
        "colors": getattr(obj, 'colors', None) or {
            "axis": "black",
            "first_station_points": "blue",
            "cross_section_loops": "red",
            "longitudinal_lines": "gray"
        },
        "AxisRotation": axis_rotation,
        "axis_signature": axis_sig,
    }


# def _clean_json_lines(meta_obj):
#     """Pretty-print object metadata as a list of lines for the right panel."""
#     try:
#         import json
#         txt = json.dumps(meta_obj, indent=2, ensure_ascii=False)
#     except Exception:
#         return [str(meta_obj)]
#     lines = []
#     for raw in txt.splitlines():
#         s = raw.strip()
#         if s.endswith(','):
#             s = s[:-1]
#         if s.startswith('"') and '":' in s:
#             k, rest = s.split('":', 1)
#             k = k.strip('"')
#             s = f"{k}:{rest}"
#         if s:
#             lines.append(s)
#     return lines or ["(no metadata)"]

# def get_plot_traces_matrix(
#     axis,
#     json_data,
#     stations_mm,      # (S,) in mm (already filtered)
#     ids,              # length N
#     P_mm,             # (S,N,3) global coords in mm (already flipped)
#     X_mm=None,        # (S,N) local Y in mm (for metadata)
#     Y_mm=None,        # (S,N) local Z in mm (for metadata)
#     obj_name="Object",
#     colors=None,
#     cls_obj=None,
#     show_labels=False,
#     *,
#     # NEW (all keyword-only, so old callers keep working)
#     loops_idx=None,                   # pass from build_point_matrices(...) to skip recompute
#     first_station: int = 0,           # which station to show as “first” points
#     show_axis: bool = True,
#     show_points: bool = True,
#     show_loops: bool = True,
#     show_longitudinal: bool = True,
#     station_stride_for_loops: int = 1,   # decimate loop stations
#     longitudinal_stride: int = 1,        # decimate longitudinal along stations
#     compact_meta: bool = True,           # arrays instead of dicts in customdata
# ):
#     """
#     Fast, vectorized plotting with optional decimation and compact metadata.
#     - uses precomputed loops_idx if provided (keeps geometry consistent)
#     - uses _clean_json_lines(...) to provide readable right-panel payloads
#     - converts ALL customdata to plain Python lists (no numpy arrays)
#     """
#     import numpy as _np
#     import plotly.graph_objects as go

#     default_colors = {
#         'axis': 'black',
#         'first_station_points': 'blue',
#         'cross_section_loops': 'red',
#         'longitudinal_lines': 'gray',
#     }
#     colors = {**default_colors, **(colors or {})}

#     traces = []

#     # ---------- normalize arrays once ----------
#     P_mm = _np.asarray(P_mm, dtype=float)
#     if P_mm.ndim != 3:
#         raise ValueError("P_mm must be (S,N,3)")
#     S, N, _ = P_mm.shape
#     X_mm = None if X_mm is None else _np.asarray(X_mm, dtype=float)
#     Y_mm = None if Y_mm is None else _np.asarray(Y_mm, dtype=float)
#     stations_mm = _np.asarray(stations_mm, dtype=float)
#     stations_m  = stations_mm / 1000.0
#     P_m = P_mm / 1000.0
#     X_m = None if X_mm is None else (X_mm / 1000.0)
#     Y_m = None if Y_mm is None else (Y_mm / 1000.0)

#     # ---------- Axis ----------
#     axis_x_m = (_np.asarray(axis.x_coords, float) / 1000.0).tolist()
#     axis_y_m = (_np.asarray(axis.y_coords, float) / 1000.0).tolist()
#     axis_z_m = (_np.asarray(axis.z_coords, float) / 1000.0).tolist()
#     if show_axis:
#         axis_stations_m = (_np.asarray(axis.stations, float) / 1000.0).tolist()  # <- LIST, not ndarray
#         traces.append(go.Scatter3d(
#             x=axis_x_m, y=axis_y_m, z=axis_z_m,
#             mode='lines+markers',
#             line=dict(color=colors['axis'], width=3),
#             marker=dict(size=3, color=colors['axis'], opacity=0.9),
#             name=f'{obj_name} Axis',
#             meta=obj_name,
#             customdata=axis_stations_m,
#             hovertemplate="<b>%{meta}</b><br>Axis @ %{customdata:.3f} m<extra></extra>",
#         ))

#     if S == 0 or N == 0:
#         return traces, axis_x_m, axis_y_m, axis_z_m

#     # ---------- First-station points ----------
#     fs = int(max(0, min(first_station, S-1)))
#     P0 = P_m[fs]                              # (N,3)
#     valid0 = ~_np.isnan(P0).any(axis=1)
#     x0 = []; y0 = []; z0 = []; ids0 = []
#     if show_points and valid0.any():
#         x0 = P0[valid0, 0].tolist()
#         y0 = P0[valid0, 1].tolist()
#         z0 = P0[valid0, 2].tolist()
#         ids0 = [ids[j] for j, ok in enumerate(valid0) if ok]
#         st0 = float(stations_m[fs])

#         if compact_meta:
#             # [id, station_m, localY_m, localZ_m, X, Y, Z]
#             ly0 = (X_m[fs, valid0] if X_m is not None else _np.full(len(ids0), _np.nan))
#             lz0 = (Y_m[fs, valid0] if Y_m is not None else _np.full(len(ids0), _np.nan))
#             cdat = _np.column_stack([
#                 _np.asarray(ids0, dtype=object),   # may be mixed types
#                 _np.full(len(ids0), st0, float),
#                 _np.asarray(ly0, float),
#                 _np.asarray(lz0, float),
#                 _np.asarray(x0, float),
#                 _np.asarray(y0, float),
#                 _np.asarray(z0, float),
#             ]).tolist()                             # <- ensure plain lists
#             traces.append(go.Scatter3d(
#                 x=x0, y=y0, z=z0,
#                 mode='markers+text' if show_labels else 'markers',
#                 marker=dict(size=4, opacity=0.9, color=colors['first_station_points']),
#                 text=ids0 if show_labels else None,
#                 textposition='top center',
#                 name=f'{obj_name} Points @ {st0:.3f} m',
#                 meta=obj_name,
#                 customdata=cdat,
#                 hovertemplate=(
#                     "<b>%{meta}</b><br>"
#                     "Point: %{customdata[0]}<br>"
#                     "Station: %{customdata[1]:.3f} m<br>"
#                     "Local Y: %{customdata[2]:.3f} m<br>"
#                     "Local Z: %{customdata[3]:.3f} m<br>"
#                     "X: %{customdata[4]:.3f} m<br>"
#                     "Y: %{customdata[5]:.3f} m<br>"
#                     "Z: %{customdata[6]:.3f} m<extra></extra>"
#                 ),
#             ))
#         else:
#             ly0 = (X_m[fs, valid0].tolist() if X_m is not None else [None]*len(ids0))
#             lz0 = (Y_m[fs, valid0].tolist() if Y_m is not None else [None]*len(ids0))
#             meta = [
#                 {"type":"point","obj":obj_name,"id":pid,"station_m":st0,
#                  "localY_m":ly,"localZ_m":lz,"globalX_m":gx,"globalY_m":gy,"globalZ_m":gz}
#                 for pid, ly, lz, gx, gy, gz in zip(ids0, ly0, lz0, x0, y0, z0)
#             ]
#             traces.append(go.Scatter3d(
#                 x=x0, y=y0, z=z0,
#                 mode='markers+text' if show_labels else 'markers',
#                 marker=dict(size=4, opacity=0.9, color=colors['first_station_points']),
#                 text=ids0 if show_labels else None,
#                 textposition='top center',
#                 name=f'{obj_name} Points @ {st0:.3f} m',
#                 customdata=meta,  # list of dicts -> JSON OK
#                 hovertemplate=(
#                     "<b>%{customdata.obj}</b><br>"
#                     "Point: %{customdata.id}<br>"
#                     "Station: %{customdata.station_m:.3f} m<br>"
#                     "Local Y: %{customdata.localY_m:.3f} m<br>"
#                     "Local Z: %{customdata.localZ_m:.3f} m<br>"
#                     "X: %{customdata.globalX_m:.3f} m<br>"
#                     "Y: %{customdata.globalY_m:.3f} m<br>"
#                     "Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
#                 ),
#             ))

#     # ---------- Cross-section loops ----------
#     all_loop_x, all_loop_y, all_loop_z, all_loop_meta = [], [], [], []
#     used_loops_idx = loops_idx
#     if show_loops:
#         if used_loops_idx is None:
#             # build once from JSON (fallback)
#             id_to_col = {pid: j for j, pid in enumerate(ids)}
#             used_loops_idx = []
#             for loop in (json_data or {}).get('Loops', []) or []:
#                 idxs = [id_to_col.get(p.get('Id')) for p in loop.get('Points', []) or []]
#                 idxs = [ix for ix in idxs if ix is not None]
#                 if idxs:
#                     used_loops_idx.append(_np.asarray(idxs, dtype=int))

#         if used_loops_idx:
#             s_range = range(0, S, max(1, int(station_stride_for_loops)))
#             for s in s_range:
#                 st_m = float(stations_m[s])
#                 for idxs in used_loops_idx:
#                     seg = P_m[s, idxs, :]           # (L,3)
#                     valid = ~_np.isnan(seg).any(axis=1)
#                     if not valid.any():
#                         continue

#                     # split contiguous runs
#                     k0 = 0
#                     Ltot = len(valid)
#                     while k0 < Ltot:
#                         while k0 < Ltot and not valid[k0]:
#                             k0 += 1
#                         if k0 >= Ltot:
#                             break
#                         k1 = k0
#                         while k1 < Ltot and valid[k1]:
#                             k1 += 1
#                         run_pts = seg[k0:k1, :]
#                         if run_pts.shape[0] >= 2:
#                             xs = run_pts[:, 0].tolist()
#                             ys = run_pts[:, 1].tolist()
#                             zs = run_pts[:, 2].tolist()

#                             # if whole loop valid, close by repeating first vertex
#                             is_closed = (k1 - k0) == Ltot
#                             if is_closed:
#                                 xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])

#                             all_loop_x.extend(xs + [None])
#                             all_loop_y.extend(ys + [None])
#                             all_loop_z.extend(zs + [None])

#                             if compact_meta:
#                                 gxs = _np.asarray(xs, float)
#                                 gys = _np.asarray(ys, float)
#                                 gzs = _np.asarray(zs, float)
#                                 L = gxs.shape[0]

#                                 if X_m is None or Y_m is None:
#                                     ly = _np.full(L, _np.nan)
#                                     lz = _np.full(L, _np.nan)
#                                 else:
#                                     ly_base = _np.asarray(X_m[s, idxs[k0:k1]], float)
#                                     lz_base = _np.asarray(Y_m[s, idxs[k0:k1]], float)
#                                     if is_closed and ly_base.size >= 1:
#                                         ly = _np.concatenate([ly_base, ly_base[:1]])
#                                         lz = _np.concatenate([lz_base, lz_base[:1]])
#                                     else:
#                                         ly = ly_base
#                                         lz = lz_base

#                                 # [station_m, localY, localZ, X, Y, Z] — lists
#                                 block = _np.column_stack([
#                                     _np.full(L, st_m),
#                                     ly, lz, gxs, gys, gzs
#                                 ]).tolist()
#                                 all_loop_meta.extend(block + [None])
#                             else:
#                                 for kk in range(len(xs)):
#                                     base = idxs[k0 + kk] if (k0 + kk) < len(idxs) else idxs[k0]
#                                     pid  = ids[base]
#                                     lyv  = float(X_m[s, base]) if X_m is not None else None
#                                     lzv  = float(Y_m[s, base]) if Y_m is not None else None
#                                     all_loop_meta.append({
#                                         "type": "loop", "obj": obj_name, "id": pid,
#                                         "station_m": st_m,
#                                         "localY_m": lyv, "localZ_m": lzv,
#                                         "globalX_m": xs[kk], "globalY_m": ys[kk], "globalZ_m": zs[kk],
#                                     })
#                                 all_loop_meta.append(None)
#                         k0 = k1 + 1

#             if all_loop_x:
#                 if compact_meta:
#                     traces.append(go.Scatter3d(
#                         x=all_loop_x, y=all_loop_y, z=all_loop_z,
#                         mode='lines',
#                         line=dict(color=colors['cross_section_loops'], width=2),
#                         name=f'{obj_name} Cross Sections',
#                         meta=obj_name,
#                         customdata=all_loop_meta,   # <- LIST (no ndarray)
#                         hovertemplate=(
#                             "<b>%{meta}</b><br>"
#                             "Station: %{customdata[0]:.3f} m<br>"
#                             "Local Y: %{customdata[1]:.3f} m<br>"
#                             "Local Z: %{customdata[2]:.3f} m<br>"
#                             "Global X: %{customdata[3]:.3f} m<br>"
#                             "Global Y: %{customdata[4]:.3f} m<br>"
#                             "Global Z: %{customdata[5]:.3f} m<extra></extra>"
#                         ),
#                     ))
#                 else:
#                     traces.append(go.Scatter3d(
#                         x=all_loop_x, y=all_loop_y, z=all_loop_z,
#                         mode='lines',
#                         line=dict(color=colors['cross_section_loops'], width=2),
#                         name=f'{obj_name} Cross Sections',
#                         customdata=all_loop_meta,   # list of dicts + None separators
#                         hovertemplate=(
#                             "<b>%{customdata.obj}</b><br>"
#                             "Point: %{customdata.id}<br>"
#                             "Station: %{customdata.station_m:.3f} m<br>"
#                             "Local Y: %{customdata.localY_m:.3f} m<br>"
#                             "Local Z: %{customdata.localZ_m:.3f} m<br>"
#                             "Global X: %{customdata.globalX_m:.3f} m<br>"
#                             "Global Y: %{customdata.globalY_m:.3f} m<br>"
#                             "Global Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
#                         ),
#                     ))

#     # ---------- Longitudinal lines (filtered to loop points; with cleaned panel lines) ----------
#     long_x, long_y, long_z, long_meta = [], [], [], []
#     if show_longitudinal:
#         # Prefer loop IDs from loops_idx (keeps geometry consistent)
#         loop_point_ids = set()
#         if loops_idx:
#             for idxs in loops_idx:
#                 for j in idxs:
#                     loop_point_ids.add(ids[j])
#         else:
#             for loop in (json_data or {}).get('Loops', []) or []:
#                 for p in loop.get('Points', []) or []:
#                     pid = p.get('Id')
#                     if pid is not None:
#                         loop_point_ids.add(pid)

#         obj_lines  = _clean_json_lines(getattr(cls_obj, "get_object_metada", lambda: {})())
#         loop_lines = _clean_json_lines(sorted(loop_point_ids))

#         stride = max(1, int(longitudinal_stride))
#         Sm = range(0, S, stride)

#         for j, pid in enumerate(ids):
#             if loop_point_ids and (pid not in loop_point_ids):
#                 continue
#             col = P_m[:, j, :]                 # (S,3)
#             valid = ~_np.isnan(col).any(axis=1)

#             last_ok = False
#             for k in Sm:
#                 ok = bool(valid[k])
#                 if ok:
#                     p = col[k]
#                     long_x.append(float(p[0])); long_y.append(float(p[1])); long_z.append(float(p[2]))
#                     # panel payload: coords first, then loop ids, then obj meta
#                     pay = [f"X: {p[0]:.3f} m", f"Y: {p[1]:.3f} m", f"Z: {p[2]:.3f} m", "—"] + loop_lines + ["—"] + obj_lines
#                     long_meta.append(pay)
#                 elif last_ok:
#                     long_x.append(None); long_y.append(None); long_z.append(None); long_meta.append(None)
#                 last_ok = ok

#             if last_ok:
#                 long_x.append(None); long_y.append(None); long_z.append(None); long_meta.append(None)

#         if long_x:
#             traces.append(go.Scatter3d(
#                 x=long_x, y=long_y, z=long_z,
#                 mode='lines',
#                 line=dict(color=colors['longitudinal_lines'], width=1),
#                 name=f'{obj_name} Longitudinal',
#                 meta=obj_name,
#                 customdata=long_meta,     # list[list[str]] + None
#                 hovertemplate=(
#                     "<b>%{meta}</b><br>"
#                     "X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>"
#                 ),
#             ))

#     # ---------- Return all coords (meters; may include None) ----------
#     all_x_m = axis_x_m + (x0 if show_points and valid0.any() else []) + all_loop_x + long_x
#     all_y_m = axis_y_m + (y0 if show_points and valid0.any() else []) + all_loop_y + long_y
#     all_z_m = axis_z_m + (z0 if show_points and valid0.any() else []) + all_loop_z + long_z
#     return traces, all_x_m, all_y_m, all_z_m
