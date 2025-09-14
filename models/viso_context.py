# models/viso_context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Iterable
from collections import defaultdict
from .mapping import mapping as MAP

if TYPE_CHECKING:
    from .axis import Axis
    from .cross_section import CrossSection
    from .main_station import MainStation
    from .pier_object import PierObject
    from .deck_object import DeckObject
    from .foundation_object import FoundationObject
    # Optional classes
    from .bearing_articulation import BearingArticulation
    from .secondary_object import SecondaryObject
    #from .materials import Materials
    #from .global_variable import GlobalVariable
    from .axis_variable import AxisVariable

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

@dataclass(slots=True)
class VisoContext:
    # Core
    axes_by_name: Dict[str, "Axis"] = field(default_factory=dict)
    crosssec_by_ncs: Dict[int, "CrossSection"] = field(default_factory=dict)
    crosssec_by_name: Dict[str, "CrossSection"] = field(default_factory=dict)
    mainstations_by_name: Dict[str, "MainStation"] = field(default_factory=dict)

    # Generic registries
    objects_by_class: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    objects_by_axis: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    objects_by_name: Dict[str, Any] = field(default_factory=dict)
    objects_by_id: Dict[Any, Any] = field(default_factory=dict)

    # Type-specific registries (optional but handy)
    piers_by_name: Dict[str, "PierObject"] = field(default_factory=dict)
    decks_by_name: Dict[str, "DeckObject"] = field(default_factory=dict)
    fnds_by_name: Dict[str, "FoundationObject"] = field(default_factory=dict)
    bearings_by_name: Dict[str, "BearingArticulation"] = field(default_factory=dict)
    secondary_by_name: Dict[str, "SecondaryObject"] = field(default_factory=dict)

    # Optional maps kept to avoid attribute errors even if classes are absent
    materials_by_no: Dict[int, Any] = field(default_factory=dict)
    materials_by_name: Dict[str, Any] = field(default_factory=dict)
    globals_by_name: Dict[str, Any] = field(default_factory=dict)

    # Debug
    verbose: bool = False

    # --------------- internals ---------------
    def _dbg(self, *parts: object) -> None:
        if self.verbose:
            print(*("[VisoContext]", *parts))

    # -------------------------------------------------------------------------
    # Build from JSON collections (axes + cross-sections + mainstations)
    # -------------------------------------------------------------------------
    @classmethod
    def from_json(
        cls,
        axis_data: List[dict],
        cross_sections: List["CrossSection"],
        mainstations: Optional[List["MainStation"]] = None,
        *,
        mapping_cfg=None,
        verbose: bool = False,
    ) -> "VisoContext":
        from .axis import Axis  # avoid cycles
        m = mapping_cfg or MAP
        amap = (m.get("Axis", None) or m.get(Axis, {})) or {}
        name_k = amap.get("name", "Name")
        sta_k  = amap.get("stations", "StaionValue")
        x_k    = amap.get("x_coords", "CurvCoorX")
        y_k    = amap.get("y_coords", "CurvCoorY")
        z_k    = amap.get("z_coords", "CurvCoorZ")
        cls_k  = amap.get("class", "Class")

        axes: Dict[str, Axis] = {}
        bad_axis_rows = 0
        for row in (axis_data or []):
            if str(row.get(cls_k, "")).strip().lower() != "axis":
                bad_axis_rows += 1
                continue
            nm = str(row.get(name_k, "")).strip()
            if not nm:
                bad_axis_rows += 1
                continue
            axes[_norm(nm)] = Axis(
                stations=[float(s) for s in row.get(sta_k, [])],
                x_coords=[float(v) for v in row.get(x_k, [])],
                y_coords=[float(v) for v in row.get(y_k, [])],
                z_coords=[float(v) for v in row.get(z_k, [])],
            )
        if verbose:
            print("[VisoContext]", "Axes built:", len(axes), "| ignored rows:", bad_axis_rows, "| names:", list(axes.keys())[:5], ("..." if len(axes) > 5 else ""))

        cs_by_ncs  = {}
        cs_by_name = {}
        for cs in (cross_sections or []):
            try:
                cs_by_ncs[int(getattr(cs, "ncs"))] = cs
            except Exception:
                pass
            nm = _norm(getattr(cs, "name", ""))
            if nm:
                cs_by_name[nm] = cs
        if verbose:
            print("[VisoContext]", "CrossSections: by_ncs=", len(cs_by_ncs), "by_name=", len(cs_by_name))

        ms_by_name = {_norm(getattr(ms, "name", "")): ms
                      for ms in (mainstations or []) if getattr(ms, "name", "").strip()}
        if verbose:
            print("[VisoContext]", "MainStations:", len(ms_by_name))

        return cls(
            axes_by_name=axes,
            crosssec_by_ncs=cs_by_ncs,
            crosssec_by_name=cs_by_name,
            mainstations_by_name=ms_by_name,
            verbose=verbose,
        )

    # -------------------------------------------------------------------------
    # Wiring & registration
    # -------------------------------------------------------------------------
    def build_viso_row(obj, ctx, mapping_cfg):
        # 1) Axis by name
        ax_name = getattr(obj, "axis_name", None) or getattr(obj, "object_axis_name", None)
        if ax_name:
            ax = ctx.get_axis(ax_name)
            if ax is not None and hasattr(obj, "axis_obj"):
                obj.axis_obj = ax

        # 2) Cross-sections by NCS
        if hasattr(obj, "_resolve_cross_sections_from_ncs"):
            obj._cross_sections = obj._resolve_cross_sections_from_ncs(ctx)

        # 3) Axis variables -> objects
        if hasattr(obj, "axis_variables") and hasattr(obj, "set_axis_variables"):
            from models.axis_variable import AxisVariable
            axis_var_map = mapping_cfg.get(AxisVariable, {}) if mapping_cfg else {}
            obj.set_axis_variables(axis_var_map)

        # 4) Return a vis row if you still render via the adapter
        return obj.get_input_for_visualisation(
            cross_section_objects=getattr(obj, "_cross_sections", None),
            axis_rotation=getattr(obj, "axis_rotation", 0.0),
            colors=getattr(obj, "colors", None),
        )


    def wire_objects(self, objs: Iterable[Any], *, axis_var_map: dict) -> None:
        """Attach axis objects & promote raw axis-variables to AxisVariable objects."""
        for o in (objs or []):
            # Axis link (accept a few common field names already mapped by your loaders)
            ax_name = getattr(o, "axis_name", None) or getattr(o, "object_axis_name", None)
            if not ax_name:
                ax_name = getattr(o, "axis_at_name", None) or getattr(o, "axis", None)
            if ax_name:
                ax = self.get_axis(ax_name)
                if ax is not None and hasattr(o, "axis_obj"):
                    o.axis_obj = ax
                    self._dbg(f"Wired axis '{ax_name}' -> {o.__class__.__name__}('{getattr(o,'name',None)}')")
                elif ax is None:
                    self._dbg(f"Axis '{ax_name}' not found for {o.__class__.__name__}('{getattr(o,'name',None)}')")

            # Axis variables: build only if the loader didn't already do it
            if not getattr(o, "axis_variables_obj", None):
                if hasattr(o, "set_axis_variables"):
                    try:
                        o.set_axis_variables()  # reads o.axis_variables and populates axis_variables_obj
                    except Exception as e:
                        if self._verbose:
                            print(f"[VisoContext] axis variables parse failed for {getattr(o,'name','?')}: {e}")
                elif hasattr(o, "axis_variables"):
                    # last-resort direct creation
                    try:
                        from axis_variable import create_axis_variables
                        o.axis_variables_obj = create_axis_variables(o.axis_variables or [])
                    except Exception:
                        o.axis_variables_obj = []


    def register_objects(self, objs: Iterable[Any]) -> None:
        """Index objects generically + into type-specific maps when available."""
        for o in (objs or []):
            clsname = o.__class__.__name__
            self.objects_by_class[clsname].append(o)

            nm = getattr(o, "name", None)
            if nm:
                n = _norm(nm)
                prev = self.objects_by_name.get(n)
                self.objects_by_name[n] = o
                if prev is not None and prev is not o:
                    self._dbg(f"Name collision on '{nm}' for class {clsname}: replaced previous instance")
                # type-specific maps
                if clsname == "DeckObject":        self.decks_by_name[n] = o
                elif clsname == "PierObject":      self.piers_by_name[n] = o
                elif clsname == "FoundationObject":self.fnds_by_name[n] = o
                elif clsname == "BearingArticulation": self.bearings_by_name[n] = o
                elif clsname == "SecondaryObject": self.secondary_by_name[n] = o
                elif clsname == "Materials":       self.materials_by_name[n] = o
                elif clsname == "GlobalVariable":  self.globals_by_name[n] = o

            oid = getattr(o, "no", None) or getattr(o, "id", None)
            if oid is not None:
                prev_id = self.objects_by_id.get(oid)
                self.objects_by_id[oid] = o
                if prev_id is not None and prev_id is not o:
                    self._dbg(f"ID collision on '{oid}' for class {clsname}: replaced previous instance")

            ax = getattr(o, "axis_name", None) or getattr(o, "object_axis_name", None)
            if ax:
                self.objects_by_axis[_norm(ax)].append(o)

            # Materials often have MaterialNo
            if clsname == "Materials":
                try:
                    mno = int(getattr(o, "material_no"))
                    self.materials_by_no[mno] = o
                except Exception:
                    pass

    def add_objects(self, *objs_lists: list[Any], axis_var_map: dict) -> None:
        for lst in objs_lists:
            self._dbg("Adding objects list of len", len(lst or []))
            self.wire_objects(lst, axis_var_map=axis_var_map)
            self.register_objects(lst)
        self._dbg("Totals after add:",
                  "classes=", {k: len(v) for k, v in self.objects_by_class.items()},
                  "axes=", {k: len(v) for k, v in self.objects_by_axis.items()})

    # -------------------------------------------------------------------------
    # Lookups
    # -------------------------------------------------------------------------
    def has_axis(self, name: str) -> bool:
        return _norm(name) in self.axes_by_name


    def get_axis(self, name: str) -> Optional["Axis"]:
        return self.axes_by_name.get(_norm(name))

    def get_cross_section(self, ncs: int) -> Optional["CrossSection"]:
        try:
            return self.crosssec_by_ncs.get(int(ncs))
        except Exception:
            return None

    def get_cross_section_by_name(self, name: str) -> Optional["CrossSection"]:
        return self.crosssec_by_name.get(_norm(name))

    def get_mainstation(self, name: str) -> Optional["MainStation"]:
        return self.mainstations_by_name.get(_norm(name))

    #def get_material(self, ident: int | str) -> Optional["Materials"]:
    #    if isinstance(ident, int):
    #        return self.materials_by_no.get(ident)
    #    return self.materials_by_name.get(_norm(str(ident)))

    #def get_global(self, name: str) -> Optional["GlobalVariable"]:
    #    return self.globals_by_name.get(_norm(name))

    def get_bearing(self, name: str) -> Optional["BearingArticulation"]:
        return self.bearings_by_name.get(_norm(name))

    def get_secondary(self, name: str) -> Optional["SecondaryObject"]:
        return self.secondary_by_name.get(_norm(name))

    def all_objects(self, clsname: Optional[str] = None) -> List[Any]:
        if not clsname:
            out: List[Any] = []
            for arr in self.objects_by_class.values():
                out.extend(arr)
            return out
        return list(self.objects_by_class.get(clsname, []))

    def objects_for_axis(self, axis_name: str, clsname: Optional[str] = None) -> List[Any]:
        items = list(self.objects_by_axis.get(_norm(axis_name), []))
        return [o for o in items if (clsname is None or o.__class__.__name__ == clsname)]

    def get_object_by_name(self, name: str) -> Optional[Any]:
        return self.objects_by_name.get(_norm(name))

    def get_object_by_id(self, oid: Any) -> Optional[Any]:
        return self.objects_by_id.get(oid)

    # Convenience: geometry owned by objects
    def compute_object_geometry(
        self,
        obj: "PierObject | DeckObject | FoundationObject",
        *,
        stations_m: List[float] | None = None,
        slices: int | None = None,
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> dict:
        fn = getattr(obj, "compute_geometry", None)
        if callable(fn):
            self._dbg(f"compute_object_geometry -> {obj.__class__.__name__}('{getattr(obj,'name',None)}')")
            return fn(ctx=self, stations_m=stations_m, slices=slices, twist_deg=twist_deg, negate_x=negate_x)
        self._dbg("compute_object_geometry: object has no compute_geometry:", obj.__class__.__name__)
        return {"ids": [], "stations_mm": [], "points_mm": [], "local_Y_mm": [], "local_Z_mm": [], "loops_idx": []}
