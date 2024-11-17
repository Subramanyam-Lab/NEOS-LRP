""" This module allows to solve CVRPLIB instances of
Capacitated Vehicle Routing Problem """
import math
import getopt
import sys
import os
import argparse
import time
from VRPSolverEasy.src import solver

class DataCvrp:
    """Contains all data for CVRP problem"""
    def __init__(
            self,
            vehicle_capacity: int,
            nb_customers: int,
            cust_demands=None,
            cust_coordinates=None,
            depot_coordinates=None):
        self.vehicle_capacity = vehicle_capacity
        self.nb_customers = nb_customers
        self.cust_demands = cust_demands
        self.cust_coordinates = cust_coordinates
        self.depot_coordinates = depot_coordinates

def compute_euclidean_distance(x_i, y_i, x_j, y_j):
    """Compute the euclidean distance between 2 points from graph"""
    distance = 100 * math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
    return int(distance)  # Always truncate to integer

def read_instance(name: str):
    """ Read an instance in the folder data from a given name """
    with open(os.path.normpath(name), "r", encoding="UTF-8") as file:
        return [str(element) for element in file.read().split()]

def read_cvrp_instances(instance_full_path):
    """Read literature instances from CVRPLIB by giving the name of instance
       and returns dictionary containing all elements of model"""
    instance_iter = iter(read_instance(instance_full_path))

    id_point = 0
    dimension_input = -1
    capacity_input = -1

    while True:
        element = next(instance_iter)
        if element == "DIMENSION":
            next(instance_iter)  # pass ":"
            dimension_input = int(next(instance_iter))
        elif element == "CAPACITY":
            next(instance_iter)  # pass ":"
            capacity_input = int(next(instance_iter))
        elif element == "EDGE_WEIGHT_TYPE":
            next(instance_iter)  # pass ":"
            element = next(instance_iter)
            if element != "EUC_2D":
                raise Exception("EDGE_WEIGHT_TYPE : " + element + " is not supported (only EUC_2D)")
        elif element == "NODE_COORD_SECTION":
            break

    vehicle_capacity = capacity_input
    cust_coordinates = []
    depot_coordinates = []

    for current_id in range(dimension_input):
        point_id = int(next(instance_iter))
        if point_id != current_id + 1:
            raise Exception("Unexpected index")
        x_coord = float(next(instance_iter))
        y_coord = float(next(instance_iter))
        
        if id_point == 0:
            depot_coordinates = [x_coord, y_coord]
        else:
            cust_coordinates.append([x_coord, y_coord])
        id_point += 1

    element = next(instance_iter)
    if element != "DEMAND_SECTION":
        raise Exception("Expected line DEMAND_SECTION")

    # Get the demands
    cust_demands = []
    for current_id in range(dimension_input):
        point_id = int(next(instance_iter))
        if point_id != current_id + 1:
            raise Exception("Unexpected index")
        demand = int(next(instance_iter))
        if current_id > 0:
            cust_demands.append(demand)

    element = next(instance_iter)
    if element != "DEPOT_SECTION":
        raise Exception("Expected line DEPOT_SECTION")
    next(instance_iter)  # pass id depot

    end_depot_section = int(next(instance_iter))
    if end_depot_section != -1:
        raise Exception("Expected only one depot.")

    return DataCvrp(vehicle_capacity,
                    dimension_input - 1,
                    cust_demands,
                    cust_coordinates,
                    depot_coordinates)

def solve_demo(instance_name, 
               F=1000, 
               time_resolution=5000, 
               solver_name_input="CLP", 
               solver_path=""):
    """return a solution from modelisation"""
    # read parameters given in command line
    type_instance = "CVRPTW/"
    if len(sys.argv) > 1:
        opts = getopt.getopt(instance_name, "i:t:s:p:")
        for opt, arg in opts[0]:
            if opt in ["-i"]:
                instance_name = arg
            if opt in ["-t"]:
                time_resolution = float(arg)
            if opt in ["-s"]:
                solver_name_input = arg
            if opt in ["-p"]:
                solver_path = arg

    # read instance
    data = read_cvrp_instances(instance_name)

    # modelisation of problem
    model = solver.Model()

    # add vehicle type
    model.add_vehicle_type(id=1,
                           start_point_id=0,
                           end_point_id=0,
                           max_number=data.nb_customers,
                           capacity=data.vehicle_capacity,
                           fixed_cost = 1000,
                           var_cost_dist=1
                           )
    # add depot
    model.add_depot(id=0)

    # add all customers
    for i in range(data.nb_customers):
        model.add_customer(id=i + 1,
                           demand=data.cust_demands[i]
                           )

    # Compute the links between depot and other points
    for i, cust_i in enumerate(data.cust_coordinates):
        dist = compute_euclidean_distance(cust_i[0], cust_i[1], 
                                          data.depot_coordinates[0], data.depot_coordinates[1])
        modified_dist = dist
        model.add_link(start_point_id=0, end_point_id=i + 1, distance=modified_dist)

    # Compute the links between points
    for i, cust_i in enumerate(data.cust_coordinates):
        for j in range(i + 1, len(data.cust_coordinates)):
            dist = compute_euclidean_distance(cust_i[0], cust_i[1],
                                              data.cust_coordinates[j][0], data.cust_coordinates[j][1])
            model.add_link(start_point_id=i + 1, end_point_id=j + 1, distance=dist)

    # set parameters
    model.set_parameters(time_limit=time_resolution,
                         solver_name=solver_name_input)

    if (solver_name_input == "CPLEX" and solver_path != ""):
        model.parameters.cplex_path = solver_path

    # solve model
    model.solve()

    if model.solution.is_defined:
        cost = model.solution.value
        m = len(model.solution.routes)
        message = model.message
        routes = []
        for route in model.solution.routes:
            routes.append({
                "Ids": list(map(str, route.point_ids))    
            })

    return cost, m, message, routes

def solve_instance(instance_path):
    start_time = time.time()
    cost, m, message, routes = solve_demo(instance_path)
    cost = round(cost)
    solve_time = time.time() - start_time

    unique_uid = os.path.basename(instance_path)

    print(f'Solution for {unique_uid} computed.')
    
    return cost, m, message, routes, solve_time

def parse_arguments():
    parser = argparse.ArgumentParser(description='CVRP Solver')
    parser.add_argument('instance_path', help='Path to the CVRP instance file')
    args = parser.parse_args()
    return args.instance_path

if __name__ == '__main__':
    instance_path = parse_arguments()
    solve_instance(instance_path)