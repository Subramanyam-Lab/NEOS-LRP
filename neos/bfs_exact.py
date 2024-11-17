import re
import os
import openpyxl
import time
from datetime import datetime

# Import the solve_instance function
from neos.solver_cvrp_vrpsolver_modified_objective_instancepath_ import solve_instance

def has_customers(instance_file_path):
    """Check if a VRP instance has any customers assigned."""
    with open(instance_file_path, 'r') as file:
        lines = file.readlines()
    try:
        demand_section_index = lines.index("DEMAND_SECTION\n")
        depot_section_index = lines.index("DEPOT_SECTION\n")
    except ValueError as e:
        raise ValueError(f"Section missing in file {instance_file_path}: {e}")
    demand_lines = lines[demand_section_index + 1:depot_section_index]
    # Remove depot demand (assumed to be the first line)
    customer_demands = demand_lines[1:]
    return len(customer_demands) > 0

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
    headers = ['Solution Folder', 'Instance', 'Exact Solver Cost']
    worksheet.append(headers)

# Initialize grand total
grand_total_solver_cost = 0

# Loop over solution folders
for solution_folder_name in os.listdir(dil_instances_folder):
    solution_folder_path = os.path.join(dil_instances_folder, solution_folder_name)
    if os.path.isdir(solution_folder_path) and (solution_folder_name.startswith('feasible_solution_') or solution_folder_name == 'final_solution'):
        print(f"Processing solution folder: {solution_folder_name}")
        
        total_solver_cost = 0
        
        # Loop over all .txt files in the folder
        for filename in os.listdir(solution_folder_path):
            if filename.endswith(".txt"):
                vrp_instance_file_path = os.path.join(solution_folder_path, filename)
                try:
                    if not has_customers(vrp_instance_file_path):
                        # No customers assigned to this depot
                        solver_cost = 0.0
                        print(f"Instance: {filename} has zero customers.")
                    else:
                        # Now, solve the instance using the exact solver
                        solver_cost, num_routes, message, routes, solver_time = solve_instance(vrp_instance_file_path)
                        print(f"Instance: {filename}, Solver Cost: {solver_cost}")

                    # Add to total cost
                    total_solver_cost += solver_cost

                    # Write to Excel
                    worksheet.append([
                        solution_folder_name,
                        filename,
                        solver_cost])

                except Exception as e:
                    print(f"Error processing instance {filename}: {e}")
                    continue

        print(f"Total Solver Cost for {solution_folder_name}: {total_solver_cost}")

        grand_total_solver_cost += total_solver_cost

        worksheet.append([
            solution_folder_name + ' Total',
            '',
            total_solver_cost])

# Save the Excel file
workbook.save(excel_file_path)
print(f"Results saved to {excel_file_path}")
