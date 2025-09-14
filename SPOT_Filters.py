"""
SPOT_Filters.py
===============

Functions for accessing SPOT JSON Objects from a master folder.

Workflow:
---------
1. Provide a master folder path.
2. Get a list of immediate subfolders (branches).
3. Default selection = "MAIN" if present.
4. Retrieve all JSON files in the selected subfolder
   where the filename starts with "_".
"""

import os
import json
import fnmatch

# ==============================
# Global branding for SPOT tools
# ==============================
GH_CATEGORY = "SPOT"
GH_SUBCATEGORY = "Load & Filter"

def get_subfolders(master_folder):
    """Return a list of subfolder names inside the master folder."""
    if not os.path.isdir(master_folder):
        raise ValueError(f"Invalid folder: {master_folder}")
    return [d for d in os.listdir(master_folder)
            if os.path.isdir(os.path.join(master_folder, d))]

def get_json_files(folder_path):
    """Return list of JSON file paths in a folder starting with '_'."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder: {folder_path}")
    return [os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.startswith("_") and f.lower().endswith(".json")]

def load_json_objects(json_files):
    """
    Load JSON objects from a list of file paths.
    Each file is expected to contain a list of JSON objects.
    Returns a single flattened list of all JSON objects.
    """
    json_objects = []
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    json_objects.extend(data)  # flatten list
                else:
                    print(f"[Warning] {file} does not contain a list, skipping.")
        except Exception as e:
            print(f"[Error] Failed to load {file}: {e}")
    return json_objects

class SpotJsonObject:
    """
    Lightweight wrapper around a JSON dict with:
      - dot & dict access
      - pretty repr
      - simple filtering helpers
      - optional auto-enrichment from CDDMapping rows

    Enrichment rule:
      - CDD rows are those with d["Class"] == "CDDMapping"
      - Lookup key: CDD["Name"]
      - Match target: target_obj["Class"]
      - On match: add AssetClass / AssetSubcategory / AssetCategory
                  (only if missing or empty) and remove 'Class'
    """

    # -------- class-level state for auto-enrichment --------
    _auto_enrich = False
    _cdd_lookup = {}   # { Name -> {AssetClass, AssetSubcategory, AssetCategory} }

    # ---- CDD mapping setup ------------------------------------------------------
    @classmethod
    def set_cdd_mapping_from_raw(cls, rows):
        """
        Build class-level CDD lookup from a list of raw dicts or SpotJsonObject.
        CDD rows are those with Class == 'CDDMapping'. Key = 'Name'.
        """
        lookup = {}
        for o in rows or []:
            d = o.to_dict() if hasattr(o, "to_dict") else (o if isinstance(o, dict) else None)
            if not isinstance(d, dict):
                continue
            if d.get("Class") == "CDDMapping":
                name = d.get("Name")
                if not name:
                    continue
                lookup[str(name)] = {
                    "AssetClass":       d.get("AssetClass", ""),
                    "AssetSubcategory": d.get("AssetSubcategory", ""),
                    "AssetCategory":    d.get("AssetCategory", "")
                }
        cls._cdd_lookup = lookup

    @classmethod
    def enable_auto_enrich(cls, flag=True):
        """Enable/disable auto enrichment at instantiation time."""
        cls._auto_enrich = bool(flag)

    @classmethod
    def clear_cdd_mapping(cls):
        """Clear the stored CDD lookup."""
        cls._cdd_lookup = {}

    # ---- core object ------------------------------------------------------------
    def __init__(self, data, source_file=None):
        if not isinstance(data, dict):
            raise TypeError("SpotJsonObject only supports JSON objects (dicts).")
        super().__setattr__('_data', dict(data))  # copy for safety
        super().__setattr__('source_file', source_file)

        # Auto-enrich now (skip CDD rows themselves)
        if self.__class__._auto_enrich and self._data.get("Class") != "CDDMapping":
            self._apply_cdd_from_lookup()

    def __repr__(self):
        obj_class = self._data.get("Class") or self._data.get("AssetClass", "NoClass")
        name = self._data.get("Name", "Unnamed")
        return f"SPOT-Obj: {obj_class} | {name}"

    # Dot access
    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"No attribute '{key}' in SpotJsonObject")

    def __setattr__(self, key, value):
        if key in ('_data', 'source_file'):
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    # Dict-style access
    def __getitem__(self, key): return self._data.get(key)
    def __setitem__(self, key, value): self._data[key] = value

    # Convenience
    def keys(self): return list(self._data.keys())
    def to_dict(self): return dict(self._data)
    def to_json(self): return json.dumps(self._data, indent=2)

    # Filtering helpers
    def matches(self, **kwargs):
        """Return True if all key=value pairs match exactly."""
        return all(self._data.get(k) == v for k, v in kwargs.items())

    def has_key(self, key): return key in self._data

    # ---- internal enrichment ----------------------------------------------------
    def _apply_cdd_from_lookup(self):
        """
        Apply class-level CDD lookup to this instance.
        Adds fields if missing/empty, and removes 'Class' if a match was applied.
        """
        lk = self.__class__._cdd_lookup
        key = self._data.get("Class")
        if not key:
            return False
        payload = lk.get(str(key))
        if not payload:
            return False

        changed = False
        for fld in ("AssetClass", "AssetSubcategory", "AssetCategory"):
            val = payload.get(fld, "")
            if val not in ("", None) and (fld not in self._data or self._data[fld] in ("", None)):
                self._data[fld] = val
                changed = True

        # Drop 'Class' only when enrichment actually applied
        if changed and "Class" in self._data:
            del self._data["Class"]

        return changed

    # ---- optional one-shot bulk enrichment (if you ever need it) ----------------
    @classmethod
    def enrich_all_with_cddmapping(cls, objects):
        """
        One-shot enrichment pass over a list of objects (SpotJsonObject or dict).
        Uses the same rule and removes 'Class' on match.
        """
        if not cls._cdd_lookup:
            # If no lookup prepared, try to build from the same list
            cls.set_cdd_mapping_from_raw(objects)

        enriched_objs = 0
        for o in objects or []:
            sj = o if isinstance(o, cls) else cls(o)
            if sj._data.get("Class") == "CDDMapping":
                continue
            if sj._apply_cdd_from_lookup():
                enriched_objs += 1
        return enriched_objs
    
class FilterObject:
    """
    Generic, safe filtering and sorting utilities for lists of
    dict-like or .to_dict()-capable objects.
    """

    def __init__(self, objects, key1=None, key2=None, key3=None):
        self.objects = list(objects) if objects else []
        self.key1 = key1
        self.key2 = key2
        self.key3 = key3

    # ---------------------------------------------------
    # Safe value extraction
    # ---------------------------------------------------
    def _get_val(self, obj, key):
        if key is None:
            return None
        if hasattr(obj, "to_dict"):
            return obj.to_dict().get(key)
        elif isinstance(obj, dict):
            return obj.get(key)
        else:
            return getattr(obj, key, None)

    # ---------------------------------------------------
    # Keyword-based filtering with wildcard support
    # ---------------------------------------------------
    def filter(self, invert=False, **kwargs):
        """
        Return only objects matching all key=value conditions.
        Supports wildcards (*, ?) in the filter values.

        If invert=True, returns all objects that do NOT match.
        """
        results = []
        for o in self.objects:
            data = o.to_dict() if hasattr(o, "to_dict") else o
            match = True
            for k, v in kwargs.items():
                val_str = str(data.get(k, "")) if data.get(k) is not None else ""
                v_str = str(v)
                if "*" in v_str or "?" in v_str:
                    # Wildcard match
                    if not fnmatch.fnmatch(val_str, v_str):
                        match = False
                        break
                else:
                    # Exact match
                    if val_str != v_str:
                        match = False
                        break
            if (match and not invert) or (not match and invert):
                results.append(o)
        return results

    # ---------------------------------------------------
    # Faceted value list
    # ---------------------------------------------------
    def get_facets(self, selected1=None, selected2=None):
        objs = self.objects
        k1_vals = sorted({
            self._get_val(o, self.key1)
            for o in objs
            if self._get_val(o, self.key1) not in (None, "")
        })

        stage1 = [o for o in objs if selected1 is None or self._get_val(o, self.key1) == selected1]
        k2_vals = sorted({
            self._get_val(o, self.key2)
            for o in stage1
            if self.key2 and self._get_val(o, self.key2) not in (None, "")
        })

        stage2 = [o for o in stage1 if selected2 is None or self._get_val(o, self.key2) == selected2]
        k3_vals = sorted({
            self._get_val(o, self.key3)
            for o in stage2
            if self.key3 and self._get_val(o, self.key3) not in (None, "")
        })

        return {"key1": k1_vals, "key2": k2_vals, "key3": k3_vals}

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------
    def exclude_classinfo(self):
        return [o for o in self.objects if self._get_val(o, "Class") != "ClassInfo"]
        
    def exclude_CDDMapping(self):
        return [o for o in self.objects if self._get_val(o, "Class") != "CDDMapping"]

    def exclude_missing(self, key):
        return [o for o in self.objects if self._get_val(o, key) is not None]

    def custom(self, func):
        results = []
        for o in self.objects:
            try:
                if func(o):
                    results.append(o)
            except Exception as e:
                print(f"[FilterError] Skipping object {o}: {e}")
        return results

    # ---------------------------------------------------
    # Smart sorting
    # ---------------------------------------------------
    def sort_smart(self, objects=None, key=None, descending=False):
        if objects is None:
            objects = self.objects
        if key is None:
            return objects

        def try_number(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip()
            if s == "":
                return None
            try:
                return float(s.replace("_", "").replace(",", ""))
            except ValueError:
                return None

        vals = [self._get_val(o, key) for o in objects]
        nums = [try_number(v) for v in vals if v not in (None, "")]
        all_numeric = len([n for n in nums if n is not None]) == len([v for v in vals if v not in (None, "")])

        if all_numeric:
            kfun = lambda o: (self._get_val(o, key) is None, try_number(self._get_val(o, key)) or 0.0)
        else:
            kfun = lambda o: (self._get_val(o, key) in (None, ""), str(self._get_val(o, key) or "").lower())

        return sorted(objects, key=kfun, reverse=descending)

if __name__ == "__main__":
    # Example usage
    master_folder = "C:\RCZ\krzysio\SPOT_KRZYSIO\GIT"
    subfolders = get_subfolders(master_folder)
    print("Subfolders:", subfolders)

    selected_folder = os.path.join(master_folder, "MAIN") if "MAIN" in subfolders else subfolders[0]
    json_files = get_json_files(selected_folder)
    print("JSON Files:", json_files)

    raw_objects = load_json_objects(json_files)    

    # raw_objects: list of dicts (from your JSON loader)
    SpotJsonObject.set_cdd_mapping_from_raw(raw_objects)
    SpotJsonObject.enable_auto_enrich(True)

    # Now wrap — each instance enriches itself (and drops 'Class' on match)
    JsonObjects = [SpotJsonObject(obj) for obj in raw_objects]

    #_log(f"CDD enrichment applied to {enriched} object(s)")

    # ================================
    # 6. Initial filter (drop ClassInfo)
    # ================================
    filt = FilterObject(JsonObjects)
    a = filt.exclude_classinfo()
    SPOT_Obj = FilterObject(a).exclude_CDDMapping()

    print("Output: {} SPOT objects, {} Objects filtered out (CDDMapping and ClassInfo objects)".format(
        len(SPOT_Obj), len(JsonObjects) - len(SPOT_Obj)
    ))


   