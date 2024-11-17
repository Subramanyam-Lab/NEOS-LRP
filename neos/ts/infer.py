import numpy as np
import onnxruntime as ort
import json
import re
import os
import openpyxl
import time
from datetime import datetime

# Import the solve_instance function
from solver_cvrp_vrpsolver_modified_objective_instancepath import solve_instance

def process_instance_for_inference(instance_file_path, fi_data):
    filename = os.path.basename(instance_file_path)
    match = re.match(r'cvrp_instance_(coord\d+-\d+-\d+)_depot_(\d+)_customers_\d+\.txt', filename)
    if match:
        instance_name = match.group(1)
        depot_id_str = match.group(2)
    else:
        raise ValueError(f"Cannot extract instance name and depot ID from filename '{filename}'")
    
    # Get fi from fi_data
    if instance_name not in fi_data:
        raise ValueError(f"Instance name '{instance_name}' not found in fi_data")
    depot_info = fi_data[instance_name]

    # Check if depot_id_str is in depot_info
    if depot_id_str not in depot_info:
        raise ValueError(f"Depot ID '{depot_id_str}' not found in fi_data for instance '{instance_name}'")
    fi = depot_info[depot_id_str]['fi']
    depot_coordinates_fi = depot_info[depot_id_str]['coordinates']

    # Now read the VRP instance from the file
    with open(instance_file_path, 'r') as file:
        instance_str = file.read()

    # Parse the instance string to extract necessary data
    problem_lines = instance_str.strip().split("\n")

    # Parse capacity
    capacity_line = next((line for line in problem_lines if line.startswith("CAPACITY :")), None)
    if not capacity_line:
        raise ValueError(f"'CAPACITY :' line not found in file {instance_file_path}")
    capacity = float(capacity_line.split(":")[1].strip())

    # Parse NODE_COORD_SECTION
    try:
        node_coord_start = problem_lines.index("NODE_COORD_SECTION") + 1
        demand_start = problem_lines.index("DEMAND_SECTION")
    except ValueError as e:
        raise ValueError(f"Section missing in file {instance_file_path}: {e}")

    customers = []
    first_line = True
    for line in problem_lines[node_coord_start:demand_start]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # Skip invalid lines
        idx, x, y = parts
        x, y = float(x), float(y)
        idx = int(idx)

        if first_line:
            depot = {'x': x, 'y': y, 'idx': idx}
            first_line = False
        else:
            customers.append({'x': x, 'y': y, 'idx': idx})

    # Verify that depot coordinates match the ones in fi_data
    x_depot_fi, y_depot_fi = depot_coordinates_fi
    if x_depot_fi != depot['x'] or y_depot_fi != depot['y']:
        raise ValueError(f"Depot coordinates in fi_data ({x_depot_fi}, {y_depot_fi}) do not match depot coordinates in instance ({depot['x']}, {depot['y']}) for instance '{instance_name}' depot '{depot_id_str}'")

    # Parse demands
    try:
        demand_start_idx = problem_lines.index("DEMAND_SECTION") + 1
        depot_section_idx = problem_lines.index("DEPOT_SECTION")
    except ValueError as e:
        raise ValueError(f"Section missing in file {instance_file_path}: {e}")

    for line in problem_lines[demand_start_idx:depot_section_idx]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue  # Skip invalid lines
        index = int(parts[0])
        demand = float(parts[1])
        if index == depot['idx']:  # Skip the depot
            continue
        customer_index = index - 2  # Adjust if necessary
        if 0 <= customer_index < len(customers):
            customers[customer_index]['demand'] = demand / capacity
        else:
            raise IndexError(f"Customer index {customer_index} out of range for instance '{instance_name}' in file {instance_file_path}")

    # Normalize coordinates using fi
    for customer in customers:
        customer['x'] = (customer['x'] - depot['x']) / fi
        customer['y'] = (customer['y'] - depot['y']) / fi

    # Prepare data for inference
    input_data = np.array([[customer['x'], customer['y'], customer['demand']] for customer in customers], dtype=np.float32)

    return input_data, fi

def run_single_onnx_inference(ort_session, input_data):
    input_data = input_data.astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    predictions = ort_session.run(None, {input_name: input_data.reshape(1, -1)})
    return predictions

# Load the phi and rho models once
phi_onnx_file_path = "/Users/waquarkaleem/NEOS-LRP-Codes-2/pre_trained_model/btd_updated/model_phi_dnn_100000.onnx" 
rho_onnx_file_path = "/Users/waquarkaleem/NEOS-LRP-Codes-2/pre_trained_model/btd_updated/model_rho_dnn_100000.onnx"

phi_session = ort.InferenceSession(phi_onnx_file_path)
rho_session = ort.InferenceSession(rho_onnx_file_path)

# Load fi_data.json
fi_data_path = '/Users/waquarkaleem/NEOS-LRP-Codes-2/neos/output/depot_fi_values.json'
with open(fi_data_path, 'r') as fi_file:
    fi_data = json.load(fi_file)

# Specify the path to your DIL instances folder
dil_instances_folder = '/Users/waquarkaleem/NEOS-LRP-Codes-2/neos/dil_instances/DIL_algo_btd_finorm_100000'

current_directory = os.getcwd()
excel_file_name = 'output_100000.xlsx'
excel_file_path = os.path.join(current_directory, excel_file_name)

if os.path.exists(excel_file_path):
    workbook = openpyxl.load_workbook(excel_file_path)
    worksheet = workbook.active
else:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    headers = ['Solution Folder', 'Instance', 'NN Cost', 'Exact Solver Cost']
    worksheet.append(headers)

# Initialize grand totals
grand_total_adjusted_cost = 0
grand_total_solver_cost = 0

# Loop over solution folders
for solution_folder_name in os.listdir(dil_instances_folder):
    solution_folder_path = os.path.join(dil_instances_folder, solution_folder_name)
    if os.path.isdir(solution_folder_path) and (solution_folder_name.startswith('feasible_solution_') or solution_folder_name == 'final_solution'):
        print(f"Processing solution folder: {solution_folder_name}")
        
        total_adjusted_cost = 0
        total_solver_cost = 0
        
        # Loop over all .txt files in the folder
        for filename in os.listdir(solution_folder_path):
            if filename.endswith(".txt"):
                vrp_instance_file_path = os.path.join(solution_folder_path, filename)
                try:
                    input_data, fi = process_instance_for_inference(vrp_instance_file_path, fi_data)

                    if len(input_data) == 0:
                        # No customers assigned to this depot
                        adjusted_cost = 0.0
                        solver_cost = 0.0
                        print(f"Instance: {filename} has zero customers. NN Cost set to {adjusted_cost}.")
                    else:
                        transformed_points = []

                        for customer_point in input_data:
                            prediction_phi = run_single_onnx_inference(phi_session, customer_point)
                            prediction_phi = prediction_phi[0]
                            transformed_points.append(prediction_phi)

                        # Sum transformed points
                        summed_transformed_points = np.sum(transformed_points, axis=0)
                        summed_transformed_points = summed_transformed_points.astype(np.float32) 

                        # Run rho model
                        prediction_rho = run_single_onnx_inference(rho_session, summed_transformed_points)
                        prediction_rho = prediction_rho[0]

                        # Get the cost and adjust
                        cost = float(prediction_rho[0])
                        adjusted_cost = float(fi * cost)

                        print(f"Instance: {filename}, NN Cost: {adjusted_cost}")

                        # Now, solve the instance using the exact solver
                        solver_cost, num_routes, message, routes, solver_time = solve_instance(vrp_instance_file_path)
                        
                        print(f"Instance: {filename}, Solver Cost: {solver_cost}")

                    # Add to totals
                    total_adjusted_cost += adjusted_cost
                    total_solver_cost += solver_cost

                    # Write to Excel
                    worksheet.append([
                        solution_folder_name,
                        filename,
                        adjusted_cost,
                        solver_cost])

                except Exception as e:
                    print(f"Error processing instance {filename}: {e}")
                    continue

        print(f"Total NN Cost for {solution_folder_name}: {total_adjusted_cost}")
        print(f"Total Solver Cost for {solution_folder_name}: {total_solver_cost}")

        # Add folder totals to grand totals
        grand_total_adjusted_cost += total_adjusted_cost
        grand_total_solver_cost += total_solver_cost

        # Write folder totals to Excel
        worksheet.append([
            solution_folder_name + ' Total',
            '',
            total_adjusted_cost,
            total_solver_cost])

# Save the Excel file
workbook.save(excel_file_path)
print(f"Results saved to {excel_file_path}")