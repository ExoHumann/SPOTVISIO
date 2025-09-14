# models/mapping.py
# python_field -> JSON key(s) mapping for SpotLoader._load_typed()

from typing import Any, Iterable, List

def _flt(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d

def _flts(seq) -> List[float]:
    try:
        return [float(v) for v in (seq or [])]
    except Exception:
        return []

def _ints(seq) -> List[int]:
    try:
        return [int(v) for v in (seq or [])]
    except Exception:
        return []

def _maybe_int(x, d=0):
    try:
        return int(x)
    except Exception:
        return d

def _list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]

def _first(row: dict, *keys: Iterable[str], default=None):
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return default

mapping = {
    # ---------------- Core typed families ----------------

    
    "CrossSection": {
        "no":                "No",
        "class_name":        "Class",
        "type":              "Type",
        "description":       "Description",
        "name":              "Name",
        "inactive":          "InActive",
        "ncs":               (lambda r: _maybe_int(_first(r, "NCS", "Ncs", "ncs"), 0)),
        "material1":         "Material1",
        "material2":         "Material2",
        "material_reinf":    "Material_Reinf",
        "json_name":         "JSON_name",
        "sofi_code":         "SofiCode",
        "points":            "Points",
        "variables":         "Variables",
        "cross_section_types": "CrossSection@Type",
        # (optional but useful)
        "axis_variables":      "AxisVariables"

    },

    "MainStation": {
        "no":                 "No",
        "class_name":         "Class",
        "type":               "Type",
        "description":        "Description",
        "name":               "Name",
        "inactive":           "InActive",
        "axis_name":          "Axis@Name",
        "placement_id":       "PlacementId",
        "ref_placement_id":   "Ref-PlacementId",
        "ref_station_offset": (lambda r: _flt(_first(r, "Ref-StationOffset"))),
        "station_value":      (lambda r: _flt(_first(r, "StationValue"))),
        "station_type":       "StationType",
        # match your dataclass field names exactly:
        "station_rotation_x": "StationRotationX",
        "station_rotation_z": "StationRotationZ",
        "sofi_code":          (lambda r: _first(r, "SOFi_Code", "SofiCode", "SOFiCode", default="")),
        

    },

    "DeckObject": {
        "no":                        "No",
        "class_name":                "Class",
        "type":                      "Type",
        "description":               "Description",
        "name":                      "Name",
        "inactive":                  "InActive",
        "axis_name":                 "Axis@Name",

        "cross_section_types":       (lambda r: _list(_first(r, "CrossSection@Type"))),
        "cross_section_names":       (lambda r: _list(_first(r, "CrossSection@Name"))),
        "cross_section_ncs":         (lambda r: _ints(_first(r, "CrossSection@Ncs", "CrossSection@NCS") or [])),
        "cross_section_points_name": (lambda r: _list(_first(r, "CrossSection_Points@Name"))),

        "placement_id":              (lambda r: _list(_first(r, "PlacementId"))),
        "placement_description":     (lambda r: _list(_first(r, "PlacementDescription"))),
        "ref_placement_id":          (lambda r: _list(_first(r, "Ref-PlacementId"))),
        "ref_station_offset":        (lambda r: _flts(_first(r, "Ref-StationOffset") or [])),
        "station_value":             (lambda r: _flts(_first(r, "StationValue") or [])),

        "grp":                       (lambda r: _list(_first(r, "Grp"))),
        "grp_offset":                (lambda r: _flts(_first(r, "GrpOffset") or [])),

        "axis_variables":            "AxisVariables",
    },

    "PierObject": {
        "no":                           "No",
        "class_name":                   "Class",
        "type":                         "Type",
        "description":                  "Description",
        "name":                         "Name",
        "inactive":                     "InActive",
        "object_axis_name":             "ObjectAxisName",
        "cross_section_type":           "CrossSection@Type",
        "cross_section_name":           "CrossSection@Name",
        "axis_name":                    "Axis@Name",

        "ref_placement_id":             "Ref-PlacementId",
        "ref_station_offset":           (lambda r: _flt(_first(r, "Ref-StationOffset"))),
        "station_value":                (lambda r: _flt(_first(r, "StationValue"))),

        "top_cross_section_points_name":(lambda r: _first(
                                                r,
                                                "Top-CrossSection_Point@Name",
                                                "Top-CrossSection_Points@Name",
                                                "TopCrossSection@PointsName",
                                                default=""
                                            )),
        "top_yoffset":                  (lambda r: _flt(_first(r, "Top-Yoffset"))),
        "top_zoffset":                  (lambda r: _flt(_first(r, "Top-Zoffset"))),
        "top_cross_section_ncs":        (lambda r: _maybe_int(_first(r, "Top-CrossSection@Ncs", "Top-CrossSection@NCS"), 0)),

        "bot_cross_section_points_name":(lambda r: _first(
                                                r,
                                                "Bot-CrossSection_Point@Name",
                                                "Bot-CrossSection_Points@Name",
                                                "BotCrossSection@PointsName",
                                                default=""
                                            )),
        "bot_yoffset":                  (lambda r: _flt(_first(r, "Bot-Yoffset"))),
        "bot_zoffset":                  (lambda r: _flt(_first(r, "Bot-Zoffset"))),
        "bot_cross_section_ncs":        (lambda r: _maybe_int(_first(r, "Bot-CrossSection@Ncs", "Bot-CrossSection@NCS"), 0)),
        "bot_pier_elevation":           (lambda r: _flt(_first(r, "Bot-PierElevation"))),

        "rotation_angle":               (lambda r: _flt(_first(r, "RotationAngle"))),
        "grp":                          (lambda r: _maybe_int(_first(r, "Grp"), 0)),
        "fixation":                     "Fixation",

        "internal_placement_id":        (lambda r: _list(_first(r, "Internal-PlacementId"))),
        "internal_ref_placement_id":    (lambda r: _list(_first(r, "Internal_Ref-PlacementId"))),
        "internal_ref_station_offset":  (lambda r: _flts(_first(r, "Internal_Ref-StationOffset") or [])),
        "internal_station_value":       (lambda r: _flts(_first(r, "Internal@StationValue", "Internal-StationValue") or [])),
        "internal_cross_section_ncs":   (lambda r: _ints(_first(r, "Internal-CrossSection@Ncs", "Internal-CrossSection@NCS") or [])),
        "grp_offset":                   (lambda r: _flts(_first(r, "GrpOffset") or [])),

        "axis_variables":               "AxisVariables",
    },

    "FoundationObject": {
        "no":                           "No",
        "class_name":                   "Class",
        "type":                         "Type",
        "description":                  "Description",
        "name":                         "Name",
        "inactive":                     "InActive",
        "object_axis_name":             "ObjectAxisName",
        "cross_section_type":           "CrossSection@Type",
        "cross_section_name":           "CrossSection@Name",
        "ref_placement_id":             "Ref-PlacementId",
        "ref_station_offset":           (lambda r: _flt(_first(r, "Ref-StationOffset"))),
        "station_value":                (lambda r: _flt(_first(r, "StationValue"))),
        "cross_section_points_name":    "CrossSection_Points@Name",
        "foundation_ref_point_y_offset":(lambda r: _flt(_first(r, "FoundationRefPoint-YOffset"))),
        "foundation_ref_point_x_offset":(lambda r: _flt(_first(r, "FoundationRefPoint-XOffset"))),
        "foundation_level":             (lambda r: _flt(_first(r, "FoundationLevel"))),
        "rotation_angle":               (lambda r: _flt(_first(r, "RotationAngle"))),
        "axis_name":                    "Axis@Name",
        "pier_object_name":             (lambda r: _list(_first(r, "PierObject@Name"))),

        "point1":                       "Point1",
        "point2":                       "Point2",
        "point3":                       "Point3",
        "point4":                       "Point4",
        "thickness":                    "Thickness",
        "grp":                          "Grp",

        "cross_section_ncs2":           (lambda r: _maybe_int(_first(r, "CrossSection@Ncs2"), 0)),
        "top_z_offset":                 (lambda r: _flt(_first(r, "Top_ZOffset"))),
        "bot_z_offset":                 (lambda r: _flt(_first(r, "Bot_ZOffset"))),
        "top_x_offset":                 (lambda r: _flt(_first(r, "Top_Xoffset"))),
        "top_y_offset":                 (lambda r: _flt(_first(r, "Top_Yoffset"))),

        "pile_dir_angle":               (lambda r: _flt(_first(r, "PileDirAngle"))),
        "pile_slope":                   (lambda r: _flt(_first(r, "PileSlope"))),

        "kx":                           (lambda r: _flt(_first(r, "Kx"))),
        "ky":                           (lambda r: _flt(_first(r, "Ky"))),
        "kz":                           (lambda r: _flt(_first(r, "Kz"))),
        "rx":                           (lambda r: _flt(_first(r, "Rx"))),
        "ry":                           (lambda r: _flt(_first(r, "Ry"))),
        "rz":                           (lambda r: _flt(_first(r, "Rz"))),

        "fixation":                     "Fixation",

        "eval_pier_object_name":        "Eval-PierObject@Name",
        "eval_station_value":           (lambda r: _flt(_first(r, "Eval-StationValue"))),
        "eval_bot_cross_section_points_name": "Eval_Bot-CrossSection_Points@Name",
        "eval_bot_y_offset":            (lambda r: _flt(_first(r, "Eval_Bot-Yoffset"))),
        "eval_bot_z_offset":            (lambda r: _flt(_first(r, "Eval_Bot-Zoffset"))),
        "eval_bot_pier_elevation":      (lambda r: _flt(_first(r, "Eval_Bot-PierElevation"))),

        "internal_placement_id":        (lambda r: _list(_first(r, "Internal-PlacementId"))),
        "internal_ref_placement_id":    (lambda r: _list(_first(r, "Internal_Ref-PlacementId"))),
        "internal_ref_station_offset":  (lambda r: _flts(_first(r, "Internal_Ref-StationOffset") or [])),
        "internal_station_value":       (lambda r: _flts(_first(r, "Internal-StationValue") or [])),
        "internal_cross_section_ncs":   (lambda r: _ints(_first(r, "Internal-CrossSection@Ncs") or [])),
        "grp_offset":                   (lambda r: _flts(_first(r, "GrpOffset") or [])),

        "axis_variables":               "AxisVariables",
    },

    "BearingArticulation": {
        "no":                           "No",
        "class_name":                   "Class",
        "type":                         "Type",
        "description":                  "Description",
        "name":                         "Name",
        "inactive":                     "InActive",

        "axis_name":                    "Axis@Name",
        "ref_placement_id":             "Ref-PlacementId",
        "ref_station_offset":           (lambda r: _flt(_first(r, "Ref-StationOffset"))),
        "station_value":                (lambda r: _flt(_first(r, "StationValue"))),
        "pier_object_name":             "PierObject@Name",

        "top_cross_section_points_name":(lambda r: _first(r, "TopCrossSection@PointsName", "Top-CrossSection_Points@Name")),
        "top_yoffset":                  (lambda r: _flt(_first(r, "Top-YOffset", "Top-Yoffset"))),
        "top_xoffset":                  (lambda r: _flt(_first(r, "Top-XOffset", "Top-Xoffset"))),

        "bot_cross_section_points_name":(lambda r: _first(r, "BotCrossSection@PointsName", "Bot-CrossSection_Points@Name")),
        "bot_yoffset":                  (lambda r: _flt(_first(r, "Bot-YOffset", "Bot-Yoffset"))),
        "bot_xoffset":                  (lambda r: _flt(_first(r, "Bot-XOffset", "Bot-Xoffset"))),

        "kx":                           (lambda r: _flt(_first(r, "Kx"))),
        "ky":                           (lambda r: _flt(_first(r, "Ky"))),
        "kz":                           (lambda r: _flt(_first(r, "Kz"))),
        "rx":                           (lambda r: _flt(_first(r, "Rx"))),
        "ry":                           (lambda r: _flt(_first(r, "Ry"))),
        "rz":                           (lambda r: _flt(_first(r, "Rz"))),

        "grp_offset":                   (lambda r: _flt(_first(r, "GRP-Offset", "GrpOffset", "GRPOffset"))),
        "grp":                          (lambda r: _first(r, "GRP", "Grp", default="")),

        "bearing_dimensions": "BearingDimentions", 
        
        "rotation_x":                   (lambda r: _flt(_first(r, "RotationX"))),
        "rotation_z":                   (lambda r: _flt(_first(r, "RotationZ"))),
        "fixation":                     "Fixation",

        "eval_pier_object_name":        "Eval-PierObject@Name",
        "eval_station_value":           (lambda r: _flt(_first(r, "Eval-StationValue"))),
        "eval_top_cross_section_points_name": "Eval_Top-CrossSection_Points@Name",
        "eval_top_yoffset":             (lambda r: _flt(_first(r, "Eval-Top-YOffset", "Eval-Top-Yoffset"))),
        "eval_top_zoffset":             (lambda r: _flt(_first(r, "Eval-Top-ZOffset", "Eval-Top-Zoffset"))),
        "sofi_code":                    (lambda r: _first(r, "SOFi_Code", "SofiCode", "SOFiCode", default="")),
    },

    "SecondaryObject": {
        "no":                      "No",
        "class_name":              "Class",
        "type":                    "Type",
        "description":             "Description",
        "name":                    "Name",
        "inactive":                "InActive",

        "beg_axis_name":           "Beg-Axis@Name",
        "end_axis_name":           "End-Axis@Name",

        "beg_placement_id":        "Beg-PlacementId",
        "beg_placement_description":"Beg-PlacementDescription",
        "beg_ref_placement_id":    (lambda r: _first(r, "Beg_Ref-PlacementId", "Beg-Ref-PlacementId", default="")),
        "beg_ref_station_offset":  (lambda r: _flt(_first(r, "Beg_Ref-StationOffset", "Beg-Ref-StationOffset"))),
        "beg_station_value":       (lambda r: _flt(_first(r, "Beg-StationValue"))),
        "beg_cross_section_points_name": "Beg-CrossSection@PointsName",
        "beg_ncs":                 (lambda r: _maybe_int(_first(r, "Beg-NCS", "Beg-Ncs"), 0)),

        "end_placement_id":        "End-PlacementId",
        "end_placement_description":"End-PlacementDescription",
        "end_ref_placement_id":    (lambda r: _first(r, "End_Ref-PlacementId", "End-Ref-PlacementId", default="")),
        "end_ref_station_offset":  (lambda r: _flt(_first(r, "End_Ref-StationOffset", "End-Ref-StationOffset"))),
        "end_station_value":       (lambda r: _flt(_first(r, "End-StationValue"))),
        "end_cross_section_points_name": "End-CrossSection@PointsName",
        "end_ncs":                 (lambda r: _maybe_int(_first(r, "End-NCS", "End-Ncs"), 0)),

        "grp_offset":              (lambda r: _flt(_first(r, "GRP-Offset", "GrpOffset", "GRPOffset"))),
        "grp":                     (lambda r: _first(r, "GRP", "Grp", default="")),

    },

    "Materials": {
        "name":            "Name",
        "structure":       "Structure",
        "material_no":     (lambda r: _maybe_int(_first(r, "MaterialNo"), 0)),
        "material_id":     "MaterialId",
        "fy":              (lambda r: _flt(_first(r, "Fy"))),
        "fu":              (lambda r: _flt(_first(r, "Fu"))),
        "elastic_modulus": (lambda r: _flt(_first(r, "ElasticModulus"))),
        "shear_modulus":   (lambda r: _flt(_first(r, "ShearModulus"))),
        "gam":             (lambda r: _flt(_first(r, "Gam"))),
        "tmax":            (lambda r: _flt(_first(r, "Tmax"))),
        "alfa":            (lambda r: _flt(_first(r, "Alfa"))),
        "custom_code":     "SOFiSTiKCustomCode",
        "description":     "Description",
    },

    "GlobalVariable": {
        "name":        "Name",
        "type":        "Type",
        "description": "Description",
        "value":       "Value",
        "unit":        "Unit",
        "inactive":    "InActive"
    },

    # used by VisoContext.wire_objects -> o.set_axis_variables(...)
    "AxisVariable": {
        "name":        "VariableName",
        "stations":    "StationValue",
        "values":      "VariableValues",
        "int_types":   "VariableIntTypes",
        "description": "VariableDescription",
    },

    # explicit keys for axes
    "Axis": {
        "name":      "Name",
        "stations":  "StaionValue",
        "x_coords":  "CurvCoorX",
        "y_coords":  "CurvCoorY",
        "z_coords":  "CurvCoorZ",
        "class":     "Class",
    },
}
