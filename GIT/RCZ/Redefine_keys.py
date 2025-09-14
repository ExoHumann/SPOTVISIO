import json

def clean_json_keys(data, prefix_to_keep="DK"):
    cleaned = {}
    for key, value in data.items():
        if "/--/" in key:
            prefix, suffix = key.split("/--/", 1)
            if prefix == prefix_to_keep:
                new_key = key  # Keep the full key
            else:
                new_key = suffix  # Remove prefix and /--/
        else:
            new_key = key
        cleaned[new_key] = value
    return cleaned

def process_file(input_file, output_file, prefix_to_keep="DK"):
    with open(input_file, 'r', encoding='utf-8') as f:
        json_objects = json.load(f)  # Load list of dicts

    processed = []
    for obj in json_objects:
        if isinstance(obj, dict):
            cleaned_obj = clean_json_keys(obj, prefix_to_keep=prefix_to_keep)
            processed.append(cleaned_obj)
        else:
            print("Skipping non-dict entry:", obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2)

process_file(
    r"C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\MAIN\_PierObject_JSON_exmp.json",
    r"C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\GIT\MAIN\_PierObject_JSON_exmp_cleaned.json",
    prefix_to_keep="}|"
)
