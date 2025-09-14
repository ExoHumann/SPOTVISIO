from __future__ import annotations
## Step by step: 
## 1. Once you have venv activared: in cdm -> C:\venv\Scripts\activate.bat
## 2. get to folder location where main.py is located -> cd C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\SPOT_VISO
## 3. run in cmd code below:

# pyinstaller --onedir ^
#   --add-data "Axis.py;." ^
#   --add-data "AxisVariables.py;." ^
#   --add-data "AddClasses.py;." ^
#   --add-data "resources/plotly-latest.min.js;resources" ^
#   --hidden-import=numpy ^
#   --hidden-import=scipy ^
#   --hidden-import=scipy.interpolate ^
#   --hidden-import=plotly ^
#   --hidden-import=plotly.graph_objects ^
#   --hidden-import=jinja2 ^
#   --upx-dir="C:\upx" ^
#   main_dash.py

import math
import json
import re
import os
import sys
import webbrowser
import ast
from functools import lru_cache

import numpy as np
import plotly
import plotly.graph_objects as go
from jinja2 import Template

from models import CrossSection

#from AddClasses import *           # CrossSection, DeckObject, FoundationObject, PierObject, mapping, find_git_folder, load_from_json
#from AxisVariables import AxisVariable
#from Axis import Axis

#from models import *
#from models.axis import Axis
#from models.cross_section import CrossSection

verbose = False
import sys; print(sys.executable)
try:
    import orjson as _fastjson
except Exception as e:
    print(f"[orjson] Unavailable: {e}")   # shows the actual reason
    _fastjson = None

class Utility:

    # import plotly.io
    # @staticmethod
    # def save_as_website(fig, file_path, *, open_browser=True, include_plotlyjs='cdn'):
    #     """
    #     Fastest: generate a self-contained HTML (or CDN) directly via Plotly.
    #     """
    #     # cdn keeps the HTML small and loads fastest
    #     html_file = os.path.splitext(file_path)[0] + '_3d.html'
    #     plotly.io.write_html(
    #         fig, file=html_file, auto_open=open_browser,
    #         include_plotlyjs=include_plotlyjs, full_html=True
    #     )
    #     print(f"Saved 3D plot to {html_file}")

    from jinja2 import Template
    import plotly.io as pio
    @staticmethod
    def save_as_website(fig, file_path, *, open_browser=True):
                # choose where to write the HTML
                base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
                html_file = os.path.join(base_path, os.path.splitext(os.path.basename(file_path))[0] + '_3d.html')

                template_str = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>3D Plot</title>
                <script src="resources/plotly-latest.min.js"></script>
                <script src="_internal/resources/plotly-latest.min.js"></script>
                <style>
                html,body{margin:0;height:100%}
                body{display:flex;flex-direction:row;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}
                #plot-container{flex:2;min-width:320px;height:100vh;background:#f5f6f7}
                #plot{width:100%;height:100%}
                #json-panel{flex:1.4;min-width:240px;max-width:560px;padding:14px;background:#fff;border-left:1px solid #ddd;overflow-y:auto}
                h3{margin:0 0 10px 0;font-size:15px;font-weight:600}
                pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.35}
                .hint{color:#666;font-size:12px;margin-top:6px}
                @media (max-width: 900px){
                    body{flex-direction:column}
                    #plot-container{height:60vh}
                    #json-panel{height:40vh;border-left:none;border-top:1px solid #ddd}
                }
                </style>
                </head>
                <body>
                <div id="plot-container"><div id="plot"></div></div>
                <div id="json-panel">
                    <h3>Hover to view metadata</h3>
                    <pre id="json-output">No item selected.</pre>
                    <div class="hint">Click also sets the panel to that item.</div>
                </div>

                <script>
                function isObject(x){ return x && typeof x === 'object' && !Array.isArray(x); }
                function pretty(x){ try { return JSON.stringify(x, null, 2); } catch(e){ return String(x); } }

                function toPanelPayload(pt){
                if (!pt) return null;
                if (pt.customdata !== undefined) return pt.customdata;
                return { trace: (pt.fullData && pt.fullData.name) || '(unknown)', x: pt.x, y: pt.y, z: pt.z };
                }

                function bootPlot(){
                var plotDiv = document.getElementById('plot');

                var figure = {{ fig | tojson | safe }};
                
                
                Plotly.newPlot(plotDiv, figure.data, figure.layout, {responsive:true}).then(function(){
                    var panel = document.getElementById('json-output');
                    var last = null;

                    function updateFrom(evt){
                    if (!evt || !evt.points || !evt.points.length) return;
                    var payload = toPanelPayload(evt.points[0]);
                    if (payload === null || payload === undefined) return;
                    last = payload;
                    panel.textContent = isObject(payload) ? pretty(payload) : String(payload);
                    }
                    plotDiv.on('plotly_hover',  updateFrom);
                    plotDiv.on('plotly_click',  updateFrom);
                    plotDiv.on('plotly_unhover', function(){
                    // keep last selection; clear if you prefer:
                    // panel.textContent = "No item selected.";
                    });
                }).catch(function(err){
                    console.error(err);
                    document.getElementById('json-output').textContent = "Plotly error. See console.";
                });
                }

                // Try local -> internal -> CDN
                (function ensurePlotly(){
                if (window.Plotly) return bootPlot();

                function load(src, onload, onerror){
                    var s = document.createElement('script');
                    s.src = src; s.async = false;
                    s.onload = onload; s.onerror = onerror;
                    document.head.appendChild(s);
                }

                load('resources/plotly-latest.min.js', bootPlot, function(){
                    load('_internal/resources/plotly-latest.min.js', bootPlot, function(){
                    load('https://cdn.plot.ly/plotly-2.27.0.min.js', bootPlot, function(){
                        document.getElementById('json-output').textContent =
                        "Could not load Plotly (local or CDN).";
                    });
                    });
                });
                })();
                </script>

                </body>
                </html>
                        """
                try:
                    template = Template(template_str)

                    # OPTION A: pass a JSON-serializable dict
                    fig_dict = fig.to_plotly_json()           # <-- key change
                    final_html = template.render(fig=fig_dict)

                    # OPTION B (equally good): a JSON string and embed it “safe”
                    # fig_json = pio.to_json(fig, validate=False)  # string
                    # final_html = template.render(fig_json=fig_json)

                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(final_html)

                    print(f"Saved 3D plot with metadata panel to {html_file}")
                    if open_browser:
                        import webbrowser
                        webbrowser.open('file://' + os.path.abspath(html_file))
                except Exception as e:
                    print(f"Error saving {html_file}: {e}")            
        
import plotly.graph_objects as go
import numpy as np

def _clean_json_lines(meta_obj):
    """Pretty-print object metadata as a list of lines for the right panel."""
    try:
        import json
        txt = json.dumps(meta_obj, indent=2, ensure_ascii=False)
    except Exception:
        return [str(meta_obj)]
    lines = []
    for raw in txt.splitlines():
        s = raw.strip()
        if s.endswith(','):
            s = s[:-1]
        if s.startswith('"') and '":' in s:
            k, rest = s.split('":', 1)
            k = k.strip('"')
            s = f"{k}:{rest}"
        if s:
            lines.append(s)
    return lines or ["(no metadata)"]

def get_plot_traces_matrix(
    axis,
    json_data,
    stations_mm,      # (S,) in mm (already filtered)
    ids,              # length N
    P_mm,             # (S,N,3) global coords in mm (already flipped)
    X_mm=None,        # (S,N) local Y in mm (for metadata)
    Y_mm=None,        # (S,N) local Z in mm (for metadata)
    obj_name="Object",
    colors=None,
    cls_obj=None,
    show_labels=False,
    *,
    # NEW (all keyword-only, so old callers keep working)
    loops_idx=None,                   # pass from build_point_matrices(...) to skip recompute
    first_station: int = 0,           # which station to show as “first” points
    show_axis: bool = True,
    show_points: bool = True,
    show_loops: bool = True,
    show_longitudinal: bool = True,
    station_stride_for_loops: int = 1,   # decimate loop stations
    longitudinal_stride: int = 1,        # decimate longitudinal along stations
    compact_meta: bool = True,           # arrays instead of dicts in customdata
):
    """
    Fast, vectorized plotting with optional decimation and compact metadata.
    - uses precomputed loops_idx if provided (keeps geometry consistent)
    - uses _clean_json_lines(...) to provide readable right-panel payloads
    - converts ALL customdata to plain Python lists (no numpy arrays)
    """
    import numpy as _np
    import plotly.graph_objects as go

    default_colors = {
        'axis': 'black',
        'first_station_points': 'blue',
        'cross_section_loops': 'red',
        'longitudinal_lines': 'gray',
    }
    colors = {**default_colors, **(colors or {})}

    traces = []

    # ---------- normalize arrays once ----------
    P_mm = _np.asarray(P_mm, dtype=float)
    if P_mm.ndim != 3:
        raise ValueError("P_mm must be (S,N,3)")
    S, N, _ = P_mm.shape
    X_mm = None if X_mm is None else _np.asarray(X_mm, dtype=float)
    Y_mm = None if Y_mm is None else _np.asarray(Y_mm, dtype=float)
    stations_mm = _np.asarray(stations_mm, dtype=float)
    stations_m  = stations_mm / 1000.0
    P_m = P_mm / 1000.0
    X_m = None if X_mm is None else (X_mm / 1000.0)
    Y_m = None if Y_mm is None else (Y_mm / 1000.0)

    # ---------- Axis ----------
    axis_x_m = (_np.asarray(axis.x_coords, float) / 1000.0).tolist()
    axis_y_m = (_np.asarray(axis.y_coords, float) / 1000.0).tolist()
    axis_z_m = (_np.asarray(axis.z_coords, float) / 1000.0).tolist()
    if show_axis:
        axis_stations_m = (_np.asarray(axis.stations, float) / 1000.0).tolist()  # <- LIST, not ndarray
        traces.append(go.Scatter3d(
            x=axis_x_m, y=axis_y_m, z=axis_z_m,
            mode='lines+markers',
            line=dict(color=colors['axis'], width=3),
            marker=dict(size=3, color=colors['axis'], opacity=0.9),
            name=f'{obj_name} Axis',
            meta=obj_name,
            customdata=axis_stations_m,
            hovertemplate="<b>%{meta}</b><br>Axis @ %{customdata:.3f} m<extra></extra>",
        ))

    if S == 0 or N == 0:
        return traces, axis_x_m, axis_y_m, axis_z_m

    # ---------- First-station points ----------
    fs = int(max(0, min(first_station, S-1)))
    P0 = P_m[fs]                              # (N,3)
    valid0 = ~_np.isnan(P0).any(axis=1)
    x0 = []; y0 = []; z0 = []; ids0 = []
    if show_points and valid0.any():
        x0 = P0[valid0, 0].tolist()
        y0 = P0[valid0, 1].tolist()
        z0 = P0[valid0, 2].tolist()
        ids0 = [ids[j] for j, ok in enumerate(valid0) if ok]
        st0 = float(stations_m[fs])

        if compact_meta:
            # [id, station_m, localY_m, localZ_m, X, Y, Z]
            ly0 = (X_m[fs, valid0] if X_m is not None else _np.full(len(ids0), _np.nan))
            lz0 = (Y_m[fs, valid0] if Y_m is not None else _np.full(len(ids0), _np.nan))
            cdat = _np.column_stack([
                _np.asarray(ids0, dtype=object),   # may be mixed types
                _np.full(len(ids0), st0, float),
                _np.asarray(ly0, float),
                _np.asarray(lz0, float),
                _np.asarray(x0, float),
                _np.asarray(y0, float),
                _np.asarray(z0, float),
            ]).tolist()                             # <- ensure plain lists
            traces.append(go.Scatter3d(
                x=x0, y=y0, z=z0,
                mode='markers+text' if show_labels else 'markers',
                marker=dict(size=4, opacity=0.9, color=colors['first_station_points']),
                text=ids0 if show_labels else None,
                textposition='top center',
                name=f'{obj_name} Points @ {st0:.3f} m',
                meta=obj_name,
                customdata=cdat,
                hovertemplate=(
                    "<b>%{meta}</b><br>"
                    "Point: %{customdata[0]}<br>"
                    "Station: %{customdata[1]:.3f} m<br>"
                    "Local Y: %{customdata[2]:.3f} m<br>"
                    "Local Z: %{customdata[3]:.3f} m<br>"
                    "X: %{customdata[4]:.3f} m<br>"
                    "Y: %{customdata[5]:.3f} m<br>"
                    "Z: %{customdata[6]:.3f} m<extra></extra>"
                ),
            ))
        else:
            ly0 = (X_m[fs, valid0].tolist() if X_m is not None else [None]*len(ids0))
            lz0 = (Y_m[fs, valid0].tolist() if Y_m is not None else [None]*len(ids0))
            meta = [
                {"type":"point","obj":obj_name,"id":pid,"station_m":st0,
                 "localY_m":ly,"localZ_m":lz,"globalX_m":gx,"globalY_m":gy,"globalZ_m":gz}
                for pid, ly, lz, gx, gy, gz in zip(ids0, ly0, lz0, x0, y0, z0)
            ]
            traces.append(go.Scatter3d(
                x=x0, y=y0, z=z0,
                mode='markers+text' if show_labels else 'markers',
                marker=dict(size=4, opacity=0.9, color=colors['first_station_points']),
                text=ids0 if show_labels else None,
                textposition='top center',
                name=f'{obj_name} Points @ {st0:.3f} m',
                customdata=meta,  # list of dicts -> JSON OK
                hovertemplate=(
                    "<b>%{customdata.obj}</b><br>"
                    "Point: %{customdata.id}<br>"
                    "Station: %{customdata.station_m:.3f} m<br>"
                    "Local Y: %{customdata.localY_m:.3f} m<br>"
                    "Local Z: %{customdata.localZ_m:.3f} m<br>"
                    "X: %{customdata.globalX_m:.3f} m<br>"
                    "Y: %{customdata.globalY_m:.3f} m<br>"
                    "Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
                ),
            ))

    # ---------- Cross-section loops ----------
    all_loop_x, all_loop_y, all_loop_z, all_loop_meta = [], [], [], []
    used_loops_idx = loops_idx
    if show_loops:
        if used_loops_idx is None:
            # build once from JSON (fallback)
            id_to_col = {pid: j for j, pid in enumerate(ids)}
            used_loops_idx = []
            for loop in (json_data or {}).get('Loops', []) or []:
                idxs = [id_to_col.get(p.get('Id')) for p in loop.get('Points', []) or []]
                idxs = [ix for ix in idxs if ix is not None]
                if idxs:
                    used_loops_idx.append(_np.asarray(idxs, dtype=int))

        if used_loops_idx:
            s_range = range(0, S, max(1, int(station_stride_for_loops)))
            for s in s_range:
                st_m = float(stations_m[s])
                for idxs in used_loops_idx:
                    seg = P_m[s, idxs, :]           # (L,3)
                    valid = ~_np.isnan(seg).any(axis=1)
                    if not valid.any():
                        continue

                    # split contiguous runs
                    k0 = 0
                    Ltot = len(valid)
                    while k0 < Ltot:
                        while k0 < Ltot and not valid[k0]:
                            k0 += 1
                        if k0 >= Ltot:
                            break
                        k1 = k0
                        while k1 < Ltot and valid[k1]:
                            k1 += 1
                        run_pts = seg[k0:k1, :]
                        if run_pts.shape[0] >= 2:
                            xs = run_pts[:, 0].tolist()
                            ys = run_pts[:, 1].tolist()
                            zs = run_pts[:, 2].tolist()

                            # if whole loop valid, close by repeating first vertex
                            is_closed = (k1 - k0) == Ltot
                            if is_closed:
                                xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])

                            all_loop_x.extend(xs + [None])
                            all_loop_y.extend(ys + [None])
                            all_loop_z.extend(zs + [None])

                            if compact_meta:
                                gxs = _np.asarray(xs, float)
                                gys = _np.asarray(ys, float)
                                gzs = _np.asarray(zs, float)
                                L = gxs.shape[0]

                                if X_m is None or Y_m is None:
                                    ly = _np.full(L, _np.nan)
                                    lz = _np.full(L, _np.nan)
                                else:
                                    ly_base = _np.asarray(X_m[s, idxs[k0:k1]], float)
                                    lz_base = _np.asarray(Y_m[s, idxs[k0:k1]], float)
                                    if is_closed and ly_base.size >= 1:
                                        ly = _np.concatenate([ly_base, ly_base[:1]])
                                        lz = _np.concatenate([lz_base, lz_base[:1]])
                                    else:
                                        ly = ly_base
                                        lz = lz_base

                                # [station_m, localY, localZ, X, Y, Z] — lists
                                block = _np.column_stack([
                                    _np.full(L, st_m),
                                    ly, lz, gxs, gys, gzs
                                ]).tolist()
                                all_loop_meta.extend(block + [None])
                            else:
                                for kk in range(len(xs)):
                                    base = idxs[k0 + kk] if (k0 + kk) < len(idxs) else idxs[k0]
                                    pid  = ids[base]
                                    lyv  = float(X_m[s, base]) if X_m is not None else None
                                    lzv  = float(Y_m[s, base]) if Y_m is not None else None
                                    all_loop_meta.append({
                                        "type": "loop", "obj": obj_name, "id": pid,
                                        "station_m": st_m,
                                        "localY_m": lyv, "localZ_m": lzv,
                                        "globalX_m": xs[kk], "globalY_m": ys[kk], "globalZ_m": zs[kk],
                                    })
                                all_loop_meta.append(None)
                        k0 = k1 + 1

            if all_loop_x:
                if compact_meta:
                    traces.append(go.Scatter3d(
                        x=all_loop_x, y=all_loop_y, z=all_loop_z,
                        mode='lines',
                        line=dict(color=colors['cross_section_loops'], width=2),
                        name=f'{obj_name} Cross Sections',
                        meta=obj_name,
                        customdata=all_loop_meta,   # <- LIST (no ndarray)
                        hovertemplate=(
                            "<b>%{meta}</b><br>"
                            "Station: %{customdata[0]:.3f} m<br>"
                            "Local Y: %{customdata[1]:.3f} m<br>"
                            "Local Z: %{customdata[2]:.3f} m<br>"
                            "Global X: %{customdata[3]:.3f} m<br>"
                            "Global Y: %{customdata[4]:.3f} m<br>"
                            "Global Z: %{customdata[5]:.3f} m<extra></extra>"
                        ),
                    ))
                else:
                    traces.append(go.Scatter3d(
                        x=all_loop_x, y=all_loop_y, z=all_loop_z,
                        mode='lines',
                        line=dict(color=colors['cross_section_loops'], width=2),
                        name=f'{obj_name} Cross Sections',
                        customdata=all_loop_meta,   # list of dicts + None separators
                        hovertemplate=(
                            "<b>%{customdata.obj}</b><br>"
                            "Point: %{customdata.id}<br>"
                            "Station: %{customdata.station_m:.3f} m<br>"
                            "Local Y: %{customdata.localY_m:.3f} m<br>"
                            "Local Z: %{customdata.localZ_m:.3f} m<br>"
                            "Global X: %{customdata.globalX_m:.3f} m<br>"
                            "Global Y: %{customdata.globalY_m:.3f} m<br>"
                            "Global Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
                        ),
                    ))

    # ---------- Longitudinal lines (filtered to loop points; with cleaned panel lines) ----------
    long_x, long_y, long_z, long_meta = [], [], [], []
    if show_longitudinal:
        # Prefer loop IDs from loops_idx (keeps geometry consistent)
        loop_point_ids = set()
        if loops_idx:
            for idxs in loops_idx:
                for j in idxs:
                    loop_point_ids.add(ids[j])
        else:
            for loop in (json_data or {}).get('Loops', []) or []:
                for p in loop.get('Points', []) or []:
                    pid = p.get('Id')
                    if pid is not None:
                        loop_point_ids.add(pid)

        obj_lines  = _clean_json_lines(getattr(cls_obj, "get_object_metada", lambda: {})())
        loop_lines = _clean_json_lines(sorted(loop_point_ids))

        stride = max(1, int(longitudinal_stride))
        Sm = range(0, S, stride)

        for j, pid in enumerate(ids):
            if loop_point_ids and (pid not in loop_point_ids):
                continue
            col = P_m[:, j, :]                 # (S,3)
            valid = ~_np.isnan(col).any(axis=1)

            last_ok = False
            for k in Sm:
                ok = bool(valid[k])
                if ok:
                    p = col[k]
                    long_x.append(float(p[0])); long_y.append(float(p[1])); long_z.append(float(p[2]))
                    # panel payload: coords first, then loop ids, then obj meta
                    pay = [f"X: {p[0]:.3f} m", f"Y: {p[1]:.3f} m", f"Z: {p[2]:.3f} m", "—"] + loop_lines + ["—"] + obj_lines
                    long_meta.append(pay)
                elif last_ok:
                    long_x.append(None); long_y.append(None); long_z.append(None); long_meta.append(None)
                last_ok = ok

            if last_ok:
                long_x.append(None); long_y.append(None); long_z.append(None); long_meta.append(None)

        if long_x:
            traces.append(go.Scatter3d(
                x=long_x, y=long_y, z=long_z,
                mode='lines',
                line=dict(color=colors['longitudinal_lines'], width=1),
                name=f'{obj_name} Longitudinal',
                meta=obj_name,
                customdata=long_meta,     # list[list[str]] + None
                hovertemplate=(
                    "<b>%{meta}</b><br>"
                    "X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>"
                ),
            ))

    # ---------- Return all coords (meters; may include None) ----------
    all_x_m = axis_x_m + (x0 if show_points and valid0.any() else []) + all_loop_x + long_x
    all_y_m = axis_y_m + (y0 if show_points and valid0.any() else []) + all_loop_y + long_y
    all_z_m = axis_z_m + (z0 if show_points and valid0.any() else []) + all_loop_z + long_z
    return traces, all_x_m, all_y_m, all_z_m


# ──────────────────────────────────────────────────────────────────────────────
# PIER ORIENTATION: vertical extrusion (world Z) + horizontal Ŷ from deck
# ──────────────────────────────────────────────────────────────────────────────
import numpy as _np
from models import Axis, AxisVariable

_PIER_MIN_FALLBACK_DROP_MM = 12000.0
_EPS = 1e-9

def _clamp_station_m(axis, station_m: float) -> float:
    Smm = _np.asarray(axis.stations, float)
    if Smm.size == 0 or not _np.isfinite(Smm).any():
        return float(station_m)
    s_mm = float(station_m) * 1000.0
    s_eff = float(_np.clip(s_mm, _np.nanmin(Smm), _np.nanmax(Smm)))
    return s_eff / 1000.0

def _interp_axis_point_mm(ax, station_m: float) -> _np.ndarray:
    S = _np.asarray(ax.stations, float)
    X = _np.asarray(ax.x_coords, float)
    Y = _np.asarray(ax.y_coords, float)
    Z = _np.asarray(ax.z_coords, float)
    s_mm = station_m * 1000.0
    return _np.array([
        _np.interp(s_mm, S, X),
        _np.interp(s_mm, S, Y),
        _np.interp(s_mm, S, Z),
    ], float)

def _interp_curve_and_tangent(ax, s_mm: float):
    S = _np.asarray(ax.stations, float)
    X = _np.asarray(ax.x_coords, float)
    Y = _np.asarray(ax.y_coords, float)
    Z = _np.asarray(ax.z_coords, float)
    x = _np.interp(s_mm, S, X); y = _np.interp(s_mm, S, Y); z = _np.interp(s_mm, S, Z)
    ds = max(1.0, 0.001 * (S[-1] - S[0]))
    s0 = float(_np.clip(s_mm - ds, S[0], S[-1]))
    s1 = float(_np.clip(s_mm + ds, S[0], S[-1]))
    p0 = _np.array([_np.interp(s0, S, X), _np.interp(s0, S, Y), _np.interp(s0, S, Z)], float)
    p1 = _np.array([_np.interp(s1, S, X), _np.interp(s1, S, Y), _np.interp(s1, S, Z)], float)
    T = p1 - p0
    n = float(_np.linalg.norm(T))
    if n < _EPS: T[:] = (1.0, 0.0, 0.0)
    else: T /= n
    return _np.array([x, y, z], float), T

def _deck_frame(deck_axis, station_m: float):
    """Return right-handed frame at deck station: (Tdeck, Y_dir, Z_dir)."""
    Zg = _np.array([0.0, 0.0, 1.0], float)
    _, Tdeck = _interp_curve_and_tangent(deck_axis, station_m * 1000.0)

    # lateral ≈ left of Tdeck
    Y_dir = _np.cross(Tdeck, Zg); nY = float(_np.linalg.norm(Y_dir))
    if nY < _EPS:
        # T vertical → fallback
        Xg = _np.array([1.0, 0.0, 0.0], float)
        Y_dir = _np.cross(Tdeck, Xg); nY = float(_np.linalg.norm(Y_dir))
        if nY < _EPS:
            Yg = _np.array([0.0, 1.0, 0.0], float)
            Y_dir = _np.cross(Tdeck, Yg); nY = float(_np.linalg.norm(Y_dir))
    Y_dir /= max(nY, _EPS)

    Z_dir = _np.cross(Y_dir, Tdeck); Z_dir /= max(float(_np.linalg.norm(Z_dir)), _EPS)

    # keep Z roughly "up"
    if _np.dot(Z_dir, Zg) < 0.0:
        Z_dir = -Z_dir
        Y_dir = -Y_dir

    return Tdeck, Y_dir, Z_dir

def _find_deck_master(loader):
    for vr, vo in zip(getattr(loader, "vis_data", []), getattr(loader, "vis_objs", [])):
        clsname = getattr(getattr(vo, "__class__", None), "__name__", "")
        if clsname == "DeckObject" and getattr(vo, "type", "") == "Deck":
            return vr, vo
    return None, None

def _compute_deck_at_station(engine, deck_vis_row, deck_obj, station_m: float):
    section_json = deck_vis_row.get("json_data") if deck_vis_row else None
    axis = getattr(deck_obj, "axis_obj", None) if deck_obj else None
    axis_vars_objs = getattr(deck_obj, "axis_variables_obj", None) if deck_obj else None
    if section_json is None or axis is None:
        return None
    st_m_eff = _clamp_station_m(axis, float(station_m))
    results = AxisVariable.evaluate_at_stations_cached(axis_vars_objs, [st_m_eff])
    ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx = engine.compute(
        section_json=section_json, axis=axis, axis_var_results=results,
        stations_m=[st_m_eff], twist_deg=90, negate_x=True,
    )
    P_mm = _np.asarray(P_mm)
    if P_mm.ndim != 3 or P_mm.shape[0] == 0 or P_mm.shape[2] != 3 or len(ids) == 0:
        return None
    return {"ids": ids, "P_mm": P_mm, "axis": axis, "station_m_used": st_m_eff}

def _idx_or_default(ids: list[str], wanted: str | None, *, alt: str | None = None) -> int:
    try:
        if wanted and wanted in ids: return ids.index(wanted)
        if alt and alt in ids: return ids.index(alt)
    except Exception:
        pass
    return 0

def build_pier_matrices_for_plot(
    engine,
    loader,
    pier_vis_row, pier_obj,
    *,
    deck_eval_cache: dict,
    vertical_slices: int = 2,           # top + bottom is enough for an extrusion
    default_drop_mm: float = _PIER_MIN_FALLBACK_DROP_MM,
):
    """
    Rotate only the PIER AXIS to follow the deck; keep every 2D section slice
    in a plane ⟂ to that axis (no twist). No foundation dependency.
    """
    axis = getattr(pier_obj, "axis_obj", None)
    if axis is None:
        return None
    pier_section_json = pier_vis_row.get("json_data")
    if pier_section_json is None:
        return None

    st_m = _clamp_station_m(axis, float(getattr(pier_obj, "station_value", 0.0)))
    top_name = getattr(pier_obj, "top_cross_section_points_name", "") or ""
    bot_name = getattr(pier_obj, "bot_cross_section_points_name", "") or top_name

    # local offsets (mm) measured in the section plane
    top_y_off_mm = float(getattr(pier_obj, "top_yoffset", 0.0))
    top_z_off_mm = float(getattr(pier_obj, "top_zoffset", 0.0))
    bot_y_off_mm = float(getattr(pier_obj, "bot_yoffset", 0.0))
    bot_z_off_mm = float(getattr(pier_obj, "bot_zoffset", 0.0))
    pier_elev_mm = float(getattr(pier_obj, "bot_pier_elevation", 0.0) or 0.0)

    # ---- deck eval + frame at the pier station ----
    key = ("deck", st_m)
    if key not in deck_eval_cache:
        deck_vis_row, deck_obj = _find_deck_master(loader)
        deck_eval_cache[key] = (
            _compute_deck_at_station(engine, deck_vis_row, deck_obj, st_m)
            if deck_vis_row else None
        )
    deck_eval = deck_eval_cache[key]

    deck_axis = deck_eval["axis"] if (deck_eval and "axis" in deck_eval) else axis
    Tdeck, Y_dir, Z_dir = _deck_frame(deck_axis, st_m)

    # pier axis points “down” along −Z_dir
    Tpier = Z_dir

    # anchors at the deck section point (fallback: axis point)
    if deck_eval:
        ids_d = deck_eval["ids"]; Pd0 = _np.asarray(deck_eval["P_mm"])[0]
        j_top = _idx_or_default(ids_d, top_name, alt="RA")
        j_bot = _idx_or_default(ids_d, bot_name, alt="RA")
        top_anchor_mm = Pd0[j_top].astype(float)
        bot_ref_mm    = Pd0[j_bot].astype(float)
    else:
        top_anchor_mm = _interp_axis_point_mm(axis, st_m)
        bot_ref_mm    = top_anchor_mm.copy()

    # apply top offsets IN THE SECTION PLANE (⟂ Tpier)
    # choose an orthonormal in-plane basis:
    #   U := Y_dir projected into plane ⟂ Tpier (it already is)
    #   V := Tdeck (also ⟂ Tpier)
    U = Y_dir
    V = Tdeck

    top_anchor_mm = top_anchor_mm + top_y_off_mm * U + top_z_off_mm * V

    # pier length
    drop_mm = pier_elev_mm + bot_z_off_mm
    if drop_mm < 1000.0:
        drop_mm = default_drop_mm

    bottom_anchor_mm = top_anchor_mm + (-drop_mm) * Tpier + bot_y_off_mm * U

    # ---- evaluate the pier 2D shape ONCE (we only need local Y/Z arrays) ----
    var_rows = AxisVariable.evaluate_at_stations_cached(getattr(pier_obj, "axis_variables_obj", None), [st_m]) or [{}]
    ids, _st_mm_dummy, _P_dummy, X_mm, Y_mm, loops_idx = engine.compute(
        section_json=pier_section_json,
        axis=axis,                # only to get local (X_mm,Y_mm); placement is manual
        axis_var_results=var_rows,
        stations_m=[st_m],
        twist_deg=0,
        negate_x=False,
    )
    if len(ids) == 0:
        return None

    # engine convention: X_mm == local "Y", Y_mm == local "Z"
    y_mm = _np.asarray(X_mm[0], float)
    z_mm = _np.asarray(Y_mm[0], float)
    N = y_mm.shape[0]

    # ---- build S slices: anchors along the pier axis; sections in plane ⟂ Tpier ----
    S = max(2, int(vertical_slices))
    alphas = _np.linspace(0.0, 1.0, S)[:, None]  # (S,1)
    line_vec = (bottom_anchor_mm - top_anchor_mm)[None, :]  # (1,3)
    anchors = top_anchor_mm[None, :] + alphas * line_vec    # (S,3)

    # IMPORTANT: place section points ONLY in the plane (U,V) ⟂ Tpier
    offs = (y_mm[:, None] * U[None, :]) + (z_mm[:, None] * V[None, :])   # (N,3)
    P_mm = anchors[:, None, :] + offs[None, :, :]                        # (S,N,3)

    # repeat local coords (for hover/meta)
    X_out = _np.tile(y_mm[None, :], (S, 1))
    Y_out = _np.tile(z_mm[None, :], (S, 1))
    stations_mm = _np.linspace(0.0, float(drop_mm), S)

    # a short line axis so you can render the pier axis
    pier_axis = Axis(
        stations=[0.0, float(_np.linalg.norm(line_vec)) / 1000.0],
        x_coords=[float(top_anchor_mm[0])/1000.0, float(bottom_anchor_mm[0])/1000.0],
        y_coords=[float(top_anchor_mm[1])/1000.0, float(bottom_anchor_mm[1])/1000.0],
        z_coords=[float(top_anchor_mm[2])/1000.0, float(bottom_anchor_mm[2])/1000.0],
        units="m",
    )
    try:
        pier_axis.name = f"{getattr(pier_obj,'name','Pier')} Axis"
    except Exception:
        pass

    return pier_axis, ids, stations_mm, P_mm, X_out, Y_out, loops_idx



def debug_planarity(axis, stations_mm, P_mm):
    """
    axis: your Axis instance
    stations_mm: (S,) mm used in embed
    P_mm: (S, N, 3) from engine (global mm)
    Prints RMS dot(T, offset) in mm for a handful of stations.
    """
    import numpy as np

    # sample stations along the axis curve (interpolate curve + tangent)
    # replace these with your actual helpers if you have them
    def _interp_curve_and_tangent(ax, s_mm):
        # ax.x_coords, y_coords, z_coords are along ax.stations (mm)
        S = np.asarray(ax.stations, float)
        X = np.asarray(ax.x_coords, float)
        Y = np.asarray(ax.y_coords, float)
        Z = np.asarray(ax.z_coords, float)
        # linear interp for point
        x = np.interp(s_mm, S, X)
        y = np.interp(s_mm, S, Y)
        z = np.interp(s_mm, S, Z)
        # finite-diff tangent (very small ± for numeric derivative)
        ds = max(1.0, 0.001 * (S[-1] - S[0]))
        s0 = np.clip(s_mm - ds, S[0], S[-1])
        s1 = np.clip(s_mm + ds, S[0], S[-1])
        x0 = np.interp(s0, S, X); y0 = np.interp(s0, S, Y); z0 = np.interp(s0, S, Z)
        x1 = np.interp(s1, S, X); y1 = np.interp(s1, S, Y); z1 = np.interp(s1, S, Z)
        t = np.array([x1-x0, y1-y0, z1-z0], float)
        nrm = np.linalg.norm(t)
        if nrm == 0: t[:] = (1,0,0)
        else: t /= nrm
        return np.array([x,y,z], float), t

    S, N, _ = P_mm.shape
    pick = np.linspace(0, S-1, min(S, 8), dtype=int)  # up to 8 sample stations
    for i in pick:
        A, T = _interp_curve_and_tangent(axis, float(stations_mm[i]))
        off = P_mm[i] - A[None, :]          # (N,3)
        dots = off @ T                      # (N,)
        rms = float(np.sqrt(np.nanmean(np.square(dots))))
        print(f"station #{i} @ {stations_mm[i]:.1f} mm -> planarity RMS = {rms:.3f} mm")


def memoize_on_object(obj, key, builder):
    """
    Use BaseObject.memo when available. Otherwise attach a tiny cache on the object.
    """
    m = getattr(obj, "memo", None)
    if callable(m):
        return m(key, builder)

    cache = getattr(obj, "_memo_cache", None)
    if cache is None:
        cache = {}
        setattr(obj, "_memo_cache", cache)
    if key not in cache:
        cache[key] = builder()
    return cache[key]


# main.py

import os, sys, json
import plotly.graph_objects as go
import numpy as np

from models import PierObject
from models import DeckObject
from models import FoundationObject
from models import AxisVariable

from spot_loader import SpotLoader
from models import VisoContext
# get_plot_traces_matrix imported from wherever you keep it
#from models import get_plot_traces_matrix   # or your actual path




if __name__ == "__main__":
    import logging, os, json, numpy as np
    import plotly.graph_objects as go

    logging.basicConfig(level=logging.INFO)  # INFO or DEBUG while tuning

    MASTER_GIT = r"C:\Git\SPOT_VISO_krzys\SPOT_VISO\GIT"
    BRANCH     = "MAIN"

    loader = (SpotLoader(master_folder=MASTER_GIT, branch=BRANCH, verbose=True)
              .load_raw()
              .group_by_class()
              .build_all_with_context())   # make sure this calls your cross-section enrichment

    ctx = loader.ctx
    vis_objs_all = loader.vis_objs

    #for ncs, cs in ctx.crosssec_by_ncs.items():
        #npts, nloops = cs.geometry_counts()
        #print(f"[check] {getattr(cs,'name','?')} (ncs={ncs}) -> points={npts}, loops={nloops}, json={bool(cs.json_data)}")


    def compute_object_geometry(obj, ctx, stations_m=None, slices=None, twist_deg=0.0, negate_x=True):
        if isinstance(obj, PierObject):
            return obj.compute_geometry(ctx=ctx, stations_m=stations_m, slices=slices, twist_deg=twist_deg, negate_x=negate_x)
        if isinstance(obj, DeckObject):
            return obj.compute_geometry(ctx=ctx, stations_m=stations_m, twist_deg=twist_deg+90, negate_x=negate_x)
        if isinstance(obj, FoundationObject):
            return obj.compute_geometry(ctx=ctx, stations_m=stations_m, twist_deg=twist_deg, negate_x=negate_x)
        # fallback
        return {"ids": [], "stations_mm": np.array([], float), "points_mm": np.zeros((0, 0, 3)),
                "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": []}

    #print(f"Loaded {len(vis_data_all)} objects from {MASTER_GIT}\\{BRANCH}")

    fig = go.Figure()
    _bbox_min = np.array([np.inf, np.inf, np.inf], float)
    _bbox_max = -_bbox_min

    

    def _update_bbox_from_axis(ax):
        A = np.c_[np.asarray(ax.x_coords)/1000.0,
                np.asarray(ax.y_coords)/1000.0,
                np.asarray(ax.z_coords)/1000.0]
        nonlocal_min = np.nanmin(A, axis=0)
        nonlocal_max = np.nanmax(A, axis=0)
        global _bbox_min, _bbox_max
        _bbox_min = np.minimum(_bbox_min, nonlocal_min)
        _bbox_max = np.maximum(_bbox_max, nonlocal_max)

    def _update_bbox_from_points(P_mm):
        if P_mm.size == 0: return
        P = P_mm.reshape(-1, 3) / 1000.0
        good = np.isfinite(P).all(axis=1)
        if not good.any(): return
        P = P[good]
        global _bbox_min, _bbox_max
        _bbox_min = np.minimum(_bbox_min, np.min(P, axis=0))
        _bbox_max = np.maximum(_bbox_max, np.max(P, axis=0))


    def smart_stations(obj, vis=None, ctx=None):
        import numpy as np
        base = set()

        def _floats(v):
            if v is None: return []
            try:
                if isinstance(v, np.ndarray):
                    v = v.ravel().tolist()
                elif not isinstance(v, (list, tuple, set)):
                    v = [v]
            except Exception:
                v = [v]
            out = []
            for x in v:
                try: out.append(float(x))
                except: pass
            return out

        # 1) any precomputed stations (if a dict vis row was passed)
        if isinstance(vis, dict):
            base.update(_floats(vis.get("stations_to_plot")))

        # 2) object-side breakpoints (meters)
        for key in ("user_stations", "station_value", "internal_station_value"):
            base.update(_floats(getattr(obj, key, None)))

        # 3) variable knots (meters)
        for var in getattr(obj, "axis_variables_obj", []) or []:
            base.update(_floats(var.xs))

        # 4) axis extents (axis.stations are mm)
        ax = getattr(obj, "axis_obj", None)
        if ax is not None:
            stations_attr = getattr(ax, "stations", None)
            if stations_attr is not None:
                try:
                    S = np.asarray(stations_attr, dtype=float)
                    if S.size:
                        base.update([float(S.flat[0])/1000.0, float(S.flat[-1])/1000.0])
                except Exception:
                    try:
                        S = list(stations_attr)
                        if len(S) >= 2:
                            base.update([float(S[0])/1000.0, float(S[-1])/1000.0])
                    except:
                        pass

        # (optional) mainstations per axis, if your ctx exposes them
        if ctx is not None:
            ax_name = getattr(obj, "axis_name", None) or getattr(obj, "object_axis_name", None)
            try:
                # try a few likely shapes of your context
                if hasattr(ctx, "mainstations_by_axis") and ax_name in ctx.mainstations_by_axis:
                    for ms in ctx.mainstations_by_axis[ax_name]:
                        base.update(_floats(getattr(ms, "station_value", None)))
                elif hasattr(ctx, "mainstations"):
                    for ms in ctx.mainstations:
                        if getattr(ms, "axis_name", None) == ax_name:
                            base.update(_floats(getattr(ms, "station_value", None)))
            except Exception:
                pass

        # densify like before
        b = sorted(base)
        if len(b) < 2: 
            return [round(v, 8) for v in b]

        out = []
        for a, c in zip(b[:-1], b[1:]):
            d = c - a
            if   d <   5: n = 2
            elif d <  10: n = 3
            elif d <  50: n = 10
            elif d < 100: n = 20
            elif d < 200: n = 40
            elif d < 300: n = 30
            elif d < 400: n = 40
            elif d < 500: n = 50
            else:         n = 500
            step = d / n
            out.extend(a + j*step for j in range(n))
        out.append(b[-1])
        return [round(v, 8) for v in out]


    for vis_row in vis_objs_all:
        # unify
        if isinstance(vis_row, dict):
            obj            = vis_row.get("obj") or vis_row.get("object")
            name           = vis_row.get("name", getattr(obj, "name", obj.__class__.__name__))
            section_json   = vis_row.get("json_data")
            json_file      = vis_row.get("json_file")
            loops_idx      = vis_row.get("loops_idx")
            colors         = vis_row.get("colors", {})
            twist_deg      = float(vis_row.get("AxisRotation", 0.0))
            stations_to_plot = vis_row.get("stations_to_plot")
        else:
            obj            = vis_row
            name           = getattr(obj, "name", obj.__class__.__name__)
            section_json   = getattr(obj, "json_data", None)
            json_file      = getattr(obj, "json_file", None)
            loops_idx      = getattr(obj, "loops_idx", None)
            colors         = getattr(obj, "colors", {})
            twist_deg      = float(getattr(obj, "AxisRotation", getattr(obj, "axis_rotation", 0.0) or 0.0))
            stations_to_plot = getattr(obj, "stations_to_plot", None)

        if obj is None:
            continue

        axis = getattr(obj, "axis_obj", None)
        if axis is None:
            continue

        stations_m = smart_stations(obj, ctx=ctx)  # instead of smart_stations(obj, obj)

        geo = compute_object_geometry(
            obj, ctx=ctx, stations_m=stations_m,
            slices=getattr(obj, "slices", None),
            twist_deg=twist_deg, negate_x=True
        )


        ids         = geo["ids"]
        stations_mm = geo["stations_mm"]
        P_mm        = geo["points_mm"]
        X_mm        = geo["local_Y_mm"]
        Y_mm        = geo["local_Z_mm"]
        loops_idx   = geo.get("loops_idx") or loops_idx  # prefer precomputed if present

        # load section JSON only if needed
        if section_json is None and json_file and os.path.isfile(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                section_json = json.load(f)

        traces, xs, ys, zs = get_plot_traces_matrix(
            axis, section_json, stations_mm, ids, P_mm,
            X_mm=X_mm, Y_mm=Y_mm, cls_obj=obj, obj_name=name,
            colors=colors, loops_idx=loops_idx,
            station_stride_for_loops=1, longitudinal_stride=1, compact_meta=True,
        )
        fig.add_traces(traces)
        _update_bbox_from_points(P_mm)
        _update_bbox_from_axis(axis)


    # Fit scene
    if np.isfinite(_bbox_min).all() and np.isfinite(_bbox_max).all():
        cx, cy, cz = (_bbox_min + _bbox_max)/2.0
        rx, ry, rz = (_bbox_max - _bbox_min)
        r = float(max(rx, ry, rz)/2.0 or 1.0)
        ranges = dict(x=[cx-r, cx+r], y=[cy-r, cy+r], z=[cz-r, cz+r])
    else:
        ranges = dict(x=[-1,1], y=[-1,1], z=[-1,1])

    fig.update_layout(
        title='SPOT VISO — Object-owned geometry',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            xaxis=dict(range=ranges["x"]),
            yaxis=dict(range=ranges["y"]),
            zaxis=dict(range=ranges["z"]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        template='plotly_white',
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='closest',
    )

    Utility.save_as_website(fig, os.path.join(MASTER_GIT, 'combined_objects'))



# if __name__ == "__main__":
#     import os, sys, json
#     #from models.base import load_from_json
#     from models import Axis, AxisVariable

#     from section_engine import SectionGeometryEngine
#     _engine = SectionGeometryEngine()

#     from spot_loader import SpotLoader
#     MASTER_GIT = r"C:\Git\SPOT_VISO_krzys\SPOT_VISO\GIT"
#     #MASTER_GIT = r"C:\RCZ\krzysio\SPOT_KRZYSIO\GIT"
#     BRANCH     = "MAIN"   # or any other subfolder
#     COND_PIER_FOUNDATION = False  # your flag

    
#     loader = SpotLoader(MASTER_GIT, BRANCH, cond_pier_foundation=COND_PIER_FOUNDATION, verbose=True)
#     # vis_data_all, vis_objs_all = (
#     #     loader
#     #     .load_raw()
#     #     .group_by_class()
#     #     .build_vis_components(attach_section_json=True)   # attaches 'json_data' like before
#     #     .vis_data, loader.vis_objs
#     # )
#     # ctx = loader.get_ctx()

#     loader.load_raw().group_by_class().build_vis_components_no_context(attach_section_json=True, verbose=True)
#     vis_data_all = loader.vis_data
#     vis_objs_all = loader.vis_objs

    

#     if vis_objs_all and isinstance(vis_objs_all[0], dict):
#             raise RuntimeError("vis_objs_all should contain model instances, not dicts.")

#     print(f"Loaded {len(vis_data_all)} visualizable objects from {MASTER_GIT}\\{BRANCH}")

#     # Build the consolidated "objects to draw"
#     #vis_data_all, vis_objs_all = Utility.generate_input_json(folder_path, COND_PIER_FOUNDATION=False)

#     # Base path for HTML
#     base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

#     fig = go.Figure()
#     all_x = []; all_y = []; all_z = []

#     import numpy as _np
#     _bbox_min = _np.array([_np.inf, _np.inf, _np.inf], dtype=float)
#     _bbox_max = -_bbox_min

#     def _update_bbox_from_axis(ax):
#         global _bbox_min, _bbox_max
#         A = _np.c_[ _np.asarray(ax.x_coords)/1000.0,
#                     _np.asarray(ax.y_coords)/1000.0,
#                     _np.asarray(ax.z_coords)/1000.0 ]
#         _bbox_min = _np.minimum(_bbox_min, _np.nanmin(A, axis=0))
#         _bbox_max = _np.maximum(_bbox_max, _np.nanmax(A, axis=0))

#     def _update_bbox_from_points(P_mm):
#         """P_mm: (S,N,3) in mm"""
#         global _bbox_min, _bbox_max
#         if P_mm.size == 0: 
#             return
#         P = P_mm.reshape(-1, 3) / 1000.0
#         # drop NaN rows
#         good = _np.isfinite(P).all(axis=1)
#         if not good.any(): 
#             return
#         P = P[good]
#         _bbox_min = _np.minimum(_bbox_min, _np.min(P, axis=0))
#         _bbox_max = _np.maximum(_bbox_max, _np.max(P, axis=0))

#     # Example inside your rendering/build loop:
#     for obj in loader.vis_objs:
#         geo = compute_object_geometry(obj, ctx=viso_ctx, stations_m=None, slices=getattr(obj, "slices", None))

#         ids         = geo["ids"]
#         stations_mm = geo["stations_mm"]
#         P_mm        = geo["points_mm"]      # (S, N, 3)
#         Y_mm        = geo["local_Y_mm"]     # (S, N)  (historical “local Y” == X_mm)
#         Z_mm        = geo["local_Z_mm"]     # (S, N)

#         loops_idx   = geo["loops_idx"]
#         length_mm   = geo.get("length_mm", None)

#         # hand off to your plotting/renderer exactly as before
#         # e.g., vis.add_section_mesh(ids, P_mm, loops_idx, color=obj.color, name=obj.name)


#     # ---------- PASS 1: draw everything EXCEPT piers via the standard engine ----------
#     for vis_row, cls_obj in zip(loader.vis_data, loader.vis_objs):
#         # skip PierObjects in pass 1 (we'll do them in pass 2)
#         if getattr(cls_obj, "__class__", type("X",(object,),{})).__name__ == "PierObject":
#             continue

#         section_json = vis_row.get("json_data")
#         if section_json is None:
#             jf = vis_row.get("json_file")
#             if not jf:
#                 continue
#             with open(jf, "r", encoding="utf-8") as f:
#                 section_json = json.load(f)

#         axis = getattr(cls_obj, "axis_obj", None)
#         if axis is None:
#             continue

#         axis_vars_objs = getattr(cls_obj, "axis_variables_obj", None)
#         stations_to_plot_m = vis_row["stations_to_plot"]        # meters
#         twist_deg = float(vis_row.get("AxisRotation", 0.0))

#         results = AxisVariable.evaluate_at_stations_cached(axis_vars_objs, stations_to_plot_m)

#         ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx = _engine.compute(
#             section_json=section_json,
#             axis=axis,
#             axis_var_results=results,
#             stations_m=stations_to_plot_m,
#             twist_deg=90,          # <- your current setting
#             negate_x=True,
#         )

#         traces, xs, ys, zs = get_plot_traces_matrix(
#             axis, section_json, stations_mm, ids, P_mm,
#             X_mm=X_mm, Y_mm=Y_mm, cls_obj=cls_obj,
#             obj_name=vis_row["name"], colors=vis_row.get("colors", {}),
#             loops_idx=loops_idx, station_stride_for_loops=1, longitudinal_stride=1,
#             compact_meta=False,
#         )
#         fig.add_traces(traces)
#         _update_bbox_from_points(P_mm)
#         _update_bbox_from_axis(axis)

#     deck_eval_cache = {}
#     _seen = set()

#     for vis_row, cls_obj in zip(loader.vis_data, loader.vis_objs):
#         if getattr(cls_obj, "__class__", type("X",(object,),{})).__name__ != "PierObject":
#             continue
#         nm = getattr(cls_obj, "name", None)
#         if nm in _seen:
#             continue
#         _seen.add(nm)

#         built = build_pier_matrices_for_plot(
#             _engine, loader, vis_row, cls_obj,
#             deck_eval_cache=deck_eval_cache,
#             vertical_slices=6,          # denser extrusion if you like
#             default_drop_mm=12000.0,
#         )
#         if not built:
#             continue

#         pier_axis, ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx = built

#         traces, *_ = get_plot_traces_matrix(
#             pier_axis, vis_row.get("json_data"),
#             stations_mm, ids, P_mm,
#             X_mm=X_mm, Y_mm=Y_mm, cls_obj=cls_obj,
#             obj_name=vis_row["name"], colors=vis_row.get("colors", {}),
#             loops_idx=loops_idx,
#             show_axis=True,             # show the pier axis line
#             station_stride_for_loops=1, longitudinal_stride=1,
#             compact_meta=False,
#         )
#         fig.add_traces(traces)

#         _update_bbox_from_points(P_mm)

#     # 7) Global equal ranges from bbox
#     if _np.isfinite(_bbox_min).all() and _np.isfinite(_bbox_max).all():
#         cx, cy, cz = (_bbox_min + _bbox_max)/2.0
#         rx, ry, rz = (_bbox_max - _bbox_min)
#         r = float(max(rx, ry, rz)/2.0 or 1.0)
#         x_range = [cx - r, cx + r]
#         y_range = [cy - r, cy + r]
#         z_range = [cz - r, cz + r]
#     else:
#         x_range = y_range = z_range = [-1, 1]

#     fig.update_layout(
#         title='3D Scatter Plot (vectorized, with metadata & magnetic hover)',
#         scene=dict(
#             xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
#             xaxis=dict(range=x_range), yaxis=dict(range=y_range), zaxis=dict(range=z_range),
#             aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
#             camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
#         ),
#         showlegend=True, template='plotly_white',
#         margin=dict(l=0, r=0, t=50, b=0),
#         hovermode='closest',
#         hoverdistance=-1, spikedistance=-1,
#         uirevision="keep",
#     )
#     fig.update_scenes(xaxis=dict(showspikes=True), yaxis=dict(showspikes=True), zaxis=dict(showspikes=True))
#     Utility.save_as_website(fig, os.path.join(base_path, 'combined_objects'))

