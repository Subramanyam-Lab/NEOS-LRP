import torch.nn as nn
import torch
import onnx
import numpy as np
from onnx2torch import convert
from dataparse import *
from network import *
from flp_org import *
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from gurobi_ml import *
import gurobi_ml.torch as gt           
from datetime import datetime
import logging
import os
import sys
import openpyxl
import pandas as pd

# Add the read_min_max_values function
def read_min_max_values(file_path):
    min_max_values = {}
    with open(file_path, 'r') as f:
        for line in f:
            if 'Overall Min X:' in line:
                min_max_values['min_x'] = float(line.strip().split(': ')[1])
            elif 'Overall Max X:' in line:
                min_max_values['max_x'] = float(line.strip().split(': ')[1])
            elif 'Overall Min Y:' in line:
                min_max_values['min_y'] = float(line.strip().split(': ')[1])
            elif 'Overall Max Y:' in line:
                min_max_values['max_y'] = float(line.strip().split(': ')[1])
            elif 'Overall Min Demand:' in line:
                min_max_values['min_demand'] = float(line.strip().split(': ')[1])
            elif 'Overall Max Demand:' in line:
                min_max_values['max_demand'] = float(line.strip().split(': ')[1])
            elif 'Minimum Cost:' in line:
                min_max_values['min_cost'] = float(line.strip().split(': ')[1])
            elif 'Maximum Cost:' in line:
                min_max_values['max_cost'] = float(line.strip().split(': ')[1])
    return min_max_values

# Modify norm_data to use min-max normalization
def norm_data_min_max(depot_cord, customer_cord, vehicle_capacity, customer_demand, rc_cal_index, min_max_values, depot_index=None):
    # Perform min-max normalization on customer coordinates and demands
    min_x = min_max_values['min_x']
    max_x = min_max_values['max_x']
    min_y = min_max_values['min_y']
    max_y = min_max_values['max_y']
    min_demand = min_max_values['min_demand']
    max_demand = min_max_values['max_demand']

    # Avoid division by zero
    denom_x = max_x - min_x if max_x != min_x else 1
    denom_y = max_y - min_y if max_y != min_y else 1
    denom_demand = max_demand - min_demand if max_demand != min_demand else 1

    # Normalize depot coordinates
    normalized_depot_cord = []
    for depot in depot_cord:
        x_norm = (depot[0] - min_x) / denom_x
        y_norm = (depot[1] - min_y) / denom_y
        normalized_depot_cord.append([x_norm, y_norm])

    # Normalize customer coordinates
    normalized_customer_cord = []
    for customer in customer_cord:
        x_norm = (customer[0] - min_x) / denom_x
        y_norm = (customer[1] - min_y) / denom_y
        normalized_customer_cord.append([x_norm, y_norm])

    # Normalize customer demands
    normalized_customer_demand = []
    for demand in customer_demand:
        demand_norm = (demand - min_demand) / denom_demand
        normalized_customer_demand.append(demand_norm)

    # Create facility_dict
    facility_dict = {}
    if depot_index is not None:
        # Processing a single depot
        customer_data = []
        for i in range(len(customer_cord)):
            customer_data.append({
                'x': normalized_customer_cord[i][0],
                'y': normalized_customer_cord[i][1],
                'demand': normalized_customer_demand[i]
            })
        facility_dict[depot_index] = pd.DataFrame(customer_data)
    else:
        # Processing multiple depots
        for idx, depot_idx in enumerate(range(len(depot_cord))):
            customer_data = []
            for i in range(len(customer_cord)):
                customer_data.append({
                    'x': normalized_customer_cord[i][0],
                    'y': normalized_customer_cord[i][1],
                    'demand': normalized_customer_demand[i]
                })
            facility_dict[depot_idx] = pd.DataFrame(customer_data)
    big_m = 1e6

    return facility_dict, big_m

class createLRP():
    def __init__(self, ans):
        self.customer_no = ans[0]
        self.depotno = ans[1]
        self.depot_cord = ans[2]
        self.customer_cord = ans[3]
        self.vehicle_capacity = ans[4]
        self.depot_capacity = ans[5]
        self.customer_demand = ans[6]
        self.facilitycost = ans[7]
        self.init_route_cost = ans[8]
        self.rc_cal_index = ans[9]

    def dataprocess(self, data_input_file):
        # Input data file location
        phi_loc = '/Users/waquarkaleem/NEOS-LRP-Codes-2/pre_trained_model/btd_nocostnorm/model_phi_dnn_77401.onnx'
        rho_loc = '/Users/waquarkaleem/NEOS-LRP-Codes-2/pre_trained_model/btd_nocostnorm/model_rho_dnn_77401.onnx'
        logging.info(f"The phi file name: {phi_loc}\n")
        logging.info(f"The rho file name: {rho_loc}\n")

        # Read min-max values
        min_max_file_path = '/Users/waquarkaleem/NEOS-LRP-Codes-2/neos/overall_min_max.txt'
        min_max_values = read_min_max_values(min_max_file_path)

        # self.customer_no, self.customer_demand, self.depotno, self.depot_cord, self.customer_cord, \
        # self.vehicle_capacity, self.depot_capacity, self.facilitycost, self.init_route_cost, \
        # self.rc_cal_index = createdata(data_input_file)

        # Normalize data using min-max normalization
        facility_dict, big_m = norm_data_min_max(
            self.depot_cord, self.customer_cord, self.vehicle_capacity, self.customer_demand,
            self.rc_cal_index, min_max_values
        )

        # Initial facility customer assignments
        initial_flp_assignment = flp(
            self.customer_no, self.depotno, self.depot_cord, self.customer_cord, self.depot_capacity,
            self.customer_demand, self.facilitycost, self.init_route_cost, self.rc_cal_index
        )
        logging.info(f"Initial FLP Assignments {initial_flp_assignment}")

        return facility_dict, big_m, initial_flp_assignment, phi_loc, rho_loc, min_max_values

    def warmstart(self, flp_assignment, init_route_cost, customer_cord, customer_demand,
                  rc_cal_index, phi_net, rho_net, min_max_values):
        ws_dt_st = datetime.now()
        y_open = list(flp_assignment[1].keys())

        x_start = [[0] * self.customer_no for _ in range(self.depotno)]

        for j in y_open:
            for i in flp_assignment[1][j]:
                x_start[j][i] = 1

        y_start = [0] * self.depotno
        for k in y_open:
            y_start[k] = 1

        # Normalize customers only to the facilities they are assigned
        phi_start = {}
        for i in flp_assignment[2]:
            d_cord = [flp_assignment[2][i][0]]
            c_cord = self.customer_cord
            c_dem = self.customer_demand
            v_cap = [self.vehicle_capacity[i]]
            phi_start[i] = norm_data_min_max(
                d_cord, c_cord, v_cap, c_dem, self.rc_cal_index, min_max_values, depot_index=i
            )

        fac_dict_initial = {}
        for k in phi_start:
            fac_dict_initial[k] = phi_start[k][0][k].loc[flp_assignment[1][k]]

        phi_outputs = {}
        for j in y_open:
            phi_outputs[j] = extract_onnx(fac_dict_initial[j].values, phi_net)

        sz = phi_outputs[y_open[0]].size()
        ls = sz[1]

        ws_phi_outputs = dict()
        for i in flp_assignment[1]:
            ws_phi_outputs[i] = {}
            for idx, cust_idx in enumerate(flp_assignment[1][i]):
                ws_phi_outputs[i][cust_idx] = phi_outputs[i][idx]

        z_start = {}
        for j in range(self.depotno):
            z_start[j] = [0] * ls
        for j in y_open:
            for l in range(ls):
                for i in flp_assignment[1][j]:
                    if x_start[j][i] == 1:
                        z_start[j][l] += ws_phi_outputs[j][i][l]

        # Initial routes cost
        route_cost_start = [0] * self.depotno

        for j in y_open:
            rho_output = extract_onnx(z_start[j], rho_net)
            route_cost_start[j] = rho_output[0].item()

        logging.info(f"Initial individual Route cost {route_cost_start}")
        initial_flp_cost = sum(self.facilitycost[j] * y_start[j] for j in range(self.depotno))
        logging.info(f"Initial Facility Objective value is {initial_flp_cost}")

        # Denormalize route costs
        # min_cost = min_max_values['min_cost']
        # max_cost = min_max_values['max_cost']
        # denom_cost = max_cost - min_cost if max_cost != min_cost else 1

        # Denormalize the route costs
        # route_cost_start_denorm = [route_cost_start[j] * denom_cost + min_cost for j in range(self.depotno)]

        # initial_vrp_cost_variable = sum(route_cost_start_denorm[j] * 100 for j in y_open)

        initial_vrp_cost_variable = sum(route_cost_start[j] * 100 for j in y_open)
        initial_vrp_cost = initial_vrp_cost_variable

        logging.info(f"Initial VRP Objective value is {initial_vrp_cost}")
        initial_obj = initial_flp_cost + initial_vrp_cost
        logging.info(f"Initial Total Objective value is {initial_obj}")
        ws_dt_ed = datetime.now()
        ws_exec = (ws_dt_ed - ws_dt_st).total_seconds()
        return initial_obj, x_start, y_start, route_cost_start, z_start, ws_exec

    def model(self, loc, log_filename):
        facility_dict, big_m, initial_flp_assignment, phi_loc, rho_loc, min_max_values = self.dataprocess(loc)

        # Initial Feasible Solution for Gurobi model
        initial_objective_value, xst, yst, routecost_st, z_st, ws_time = self.warmstart(
            initial_flp_assignment, self.init_route_cost, self.customer_cord, self.customer_demand,
            self.rc_cal_index, phi_loc, rho_loc, min_max_values
        )
        print("Initial Feasible solution:", initial_objective_value)

        # Passing data through phi network
        phi_final_outputs = {}
        for j in range(self.depotno):
            phi_final_outputs[j] = extract_onnx(facility_dict[j].values, phi_loc)

        sz = phi_final_outputs[0].size()
        latent_space = sz[1]

        # LRP Model
        m = gp.Model('facility_location')

        # Decision variables
        cartesian_prod = list(product(range(self.depotno), range(self.customer_no)))

        y = m.addVars(self.depotno, vtype=GRB.BINARY, lb=0, ub=1, name='Facility')
        for j in range(self.depotno):
            y[j].Start = yst[j]

        x = m.addVars(cartesian_prod, vtype=GRB.BINARY, lb=0, ub=1, name='Assign')
        for j in range(self.depotno):
            for i in range(self.customer_no):
                x[j, i].Start = xst[j][i]

        z = m.addVars(self.depotno, latent_space, vtype=GRB.CONTINUOUS, lb=0, name="z-plus")
        for j in range(self.depotno):
            for l in range(latent_space):
                z[j, l].Start = z_st[j][l]

        route_cost = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='route cost')
        for j in range(self.depotno):
            route_cost[j].Start = routecost_st[j]

        u = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name="dummy route cost")

        for j in range(self.depotno):
            for l in range(latent_space):
                m.addConstr(
                    z[j, l] == gp.quicksum(x[j, i] * phi_final_outputs[j][i, l] for i in range(self.customer_no)),
                    name=f'Z-plus[{j}][{l}]'
                )

        # Constraints
        m.addConstrs(
            (gp.quicksum(x[(j, i)] for j in range(self.depotno)) == 1 for i in range(self.customer_no)),
            name='Demand'
        )

        m.addConstrs(
            (
                gp.quicksum(x[j, i] * self.customer_demand[i] for i in range(self.customer_no)) <= self.depot_capacity[j] * y[j]
                for j in range(self.depotno)
            ),
            name="facility_capacity_constraint"
        )

        print("Start time for MIP part:", datetime.now())

        # Neural Network Constraints
        onnx_model = onnx.load(rho_loc)
        pytorch_rho_mdl = convert(onnx_model).double()
        layers = []
        for name, layer in pytorch_rho_mdl.named_children():
            layers.append(layer)
        sequential_model = nn.Sequential(*layers)

        z_values_per_depot = {}
        route_per_depot = {}

        for j in range(self.depotno):
            z_values_per_depot[j] = [z[j, l] for l in range(latent_space)]
            route_per_depot[j] = [route_cost[j]]

        for j in range(self.depotno):
            t_const = gt.add_sequential_constr(m, sequential_model, z_values_per_depot[j], route_per_depot[j])
            t_const.print_stats()

        # Denormalize route_cost[j]
        # min_cost = min_max_values['min_cost']
        # max_cost = min_max_values['max_cost']
        # denom_cost = max_cost - min_cost if max_cost != min_cost else 1

        # route_cost_denorm = m.addVars(self.depotno, vtype=GRB.CONTINUOUS, lb=0, name='route_cost_denorm')

        # for j in range(self.depotno):
            # m.addConstr(route_cost_denorm[j] == route_cost[j] * denom_cost + min_cost, name=f'DenormRouteCost[{j}]')

        # Indicator Constraint to stop cost calculation for closed depot
        for j in range(self.depotno):
            m.addConstr((y[j] == 0) >> (u[j] == 0))
            m.addConstr((y[j] == 1) >> (u[j] == route_cost[j]))

        # Objective
        facility_obj = gp.quicksum(self.facilitycost[j] * y[j] for j in range(self.depotno))
        route_obj = gp.quicksum(u[j] * 100 for j in range(self.depotno))

        m.setObjective(facility_obj + route_obj, GRB.MINIMIZE)

        sys.stdout = open(log_filename, "a")

        # Solution terminate at 1% gap
        m.setParam('MIPGap', 0.01)

        # Optimize model
        St_time1 = datetime.now()
        m.optimize()
        Ed_time = datetime.now()
        print("Objective value is ", m.objVal, '\n')

        lrp_obj = m.objVal
        logging.info(f"Objective value is {lrp_obj}")

        print('Facility objective value:', facility_obj.getValue())
        f_obj = facility_obj.getValue()
        logging.info(f'Facility objective value: {f_obj}')

        print('Route Objective value:', route_obj.getValue())
        r_obj = route_obj.getValue()
        logging.info(f'Route Objective value: {r_obj}')

        execution_time = (Ed_time - St_time1).total_seconds()
        print("Lrp model Execution time:", execution_time)
        logging.info(f"Lrp model Execution time: {execution_time}")

        # Execution time per depot
        cou = 0
        y_val = []
        for j in range(self.depotno):
            y_val.append(y[j].X)
            if y[j].X != 0:
                cou += 1
            print(cou)

        x_val = []
        for j in range(self.depotno):
            ls1 = []
            for i in range(self.customer_no):
                ls1.append(x[j, i].X)
            x_val.append(ls1)

        etpd = execution_time / cou if cou != 0 else 0

        sys.stdout.close()
        sys.stdout = sys.__stdout__

        # Close the log file
        logging.shutdown()

        return y_val, x_val, f_obj, r_obj, ws_time, execution_time

    def writeexcel(self, data_input_file, f_obj, r_obj, lrp_obj, etpd):
        existing_excel_file = "F:\\MLEV\\lrp_output.xlsx"
        workbook = openpyxl.load_workbook(existing_excel_file)

        # Select the worksheet where you want to append the new row
        worksheet = workbook.active  # Or specify a particular worksheet by name if needed

        # Create a new row
        new_row = [data_input_file, f_obj, r_obj, lrp_obj, etpd]  # Replace with your data

        # Append the new row to the worksheet
        worksheet.append(new_row)

        # Save the modified Excel file
        workbook.save(existing_excel_file)