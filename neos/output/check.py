import os
import json
import re

def load_json_fi_values(json_file_path):
    """
    Loads the fi values from the depot_fi_values.json file.
    """
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def normalize_instance_name(name):
    """
    Normalizes instance names to ensure consistency between JSON data and text files.
    """
    return name.replace('.dat', '')

def parse_text_file_fi_values(file_path):
    """
    Parses fi values from a text file like coord100-10-1_rc_norm.txt.
    """
    fi_values = {}
    # Extract the instance name from the file name
    instance_name = os.path.basename(file_path).replace('_rc_norm.txt', '')
    instance_name = normalize_instance_name(instance_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Flag to identify when we're reading the normalization factors
    reading_fi = False
    for line in lines:
        line = line.strip()
        if "Normalization factor for route cost (rc_norm):" in line:
            reading_fi = True
            continue
        if reading_fi:
            # Expecting lines like "Depot 0: 49.0"
            match = re.match(r'Depot\s+(\d+):\s+([\d\.]+)', line)
            if match:
                depot_idx = match.group(1)
                fi_value = match.group(2)
                fi_values[depot_idx] = float(fi_value)
            else:
                # If the line doesn't match the expected format, stop reading fi values
                break
    return instance_name, fi_values

def compare_fi_values(json_fi_values, text_fi_values):
    """
    Compares the fi values from the JSON data and text files.
    """
    discrepancies = []
    for instance_name in text_fi_values:
        print(f"Comparing instance: {instance_name}")
        json_instance_data = json_fi_values.get(instance_name)
        if not json_instance_data:
            discrepancies.append(f"Instance {instance_name} not found in JSON data.")
            continue
        for depot_idx in text_fi_values[instance_name]:
            json_depot_data = json_instance_data.get(depot_idx)
            if not json_depot_data:
                discrepancies.append(f"Depot {depot_idx} in instance {instance_name} not found in JSON data.")
                continue
            json_fi_value = float(json_depot_data['fi'])
            text_fi_value = text_fi_values[instance_name][depot_idx]
            tolerance = 1e-6
            if abs(json_fi_value - text_fi_value) > tolerance:
                discrepancies.append(f"Mismatch in instance {instance_name}, Depot {depot_idx}: JSON fi={json_fi_value}, Text fi={text_fi_value}")
    return discrepancies

def main():
    # Define paths
    output_dir = '/Users/waquarkaleem/NEOS-LRP-Codes-2/neos/output'
    json_file_path = os.path.join(output_dir, 'depot_fi_values.json')
    
    # Load fi values from JSON file
    json_fi_values = load_json_fi_values(json_file_path)
    
    # Normalize instance names in JSON data
    normalized_json_fi_values = {}
    for instance_name, data in json_fi_values.items():
        normalized_name = normalize_instance_name(instance_name)
        normalized_json_fi_values[normalized_name] = data
    
    # Dictionary to store fi values from text files
    text_fi_values = {}
    
    # Process each text file
    for file_name in os.listdir(output_dir):
        if file_name.endswith('_rc_norm.txt'):
            file_path = os.path.join(output_dir, file_name)
            instance_name, fi_values = parse_text_file_fi_values(file_path)
            text_fi_values[instance_name] = fi_values
    
    # Compare fi values
    discrepancies = compare_fi_values(normalized_json_fi_values, text_fi_values)
    
    # Report results
    if not discrepancies:
        print("All fi values match between the JSON file and text files.")
    else:
        print("Discrepancies found:")
        for discrepancy in discrepancies:
            print(discrepancy)
    
if __name__ == '__main__':
    main()
