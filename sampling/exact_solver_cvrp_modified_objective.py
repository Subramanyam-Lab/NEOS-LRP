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
    """Contains all data for CVRP problem
    """

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

def compute_euclidean_distance(x_i, y_i, x_j, y_j, number_digit=3):
    """Compute the euclidean distance between 2 points from graph"""
    return round(math.sqrt((x_i - x_j)**2 +
                           (y_i - y_j)**2), number_digit)

def read_instance(name: str):
    """ Read an instance in the folder data from a given name """

    with open(
            os.path.normpath(name),
            "r", encoding="UTF-8") as file:
        return [str(element) for element in file.read().split()]


# def solve_demo(instance_name,
#                time_resolution=5000,
#                solver_name_input="CLP",
#                solver_path=""):

def solve_demo(instance_name, 
               F=10, 
               time_resolution=5000, 
               solver_name_input="CLP", 
               solver_path=""):

    """return a solution from modelisation"""

    # read parameters given in command line
    type_instance = "CVRPTW/"
    if len(sys.argv) > 1:
        #print(instance_name)
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
    # for i,cust_i in enumerate(data.cust_coordinates):
    #     dist = compute_euclidean_distance(cust_i[0],
    #                                       cust_i[1],
    #                                       data.depot_coordinates[0],
    #                                       data.depot_coordinates[1],
    #                                       0)
    #     model.add_link(start_point_id=0,
    #                    end_point_id=i + 1,
    #                    distance=dist
    #                    )


    # Compute the links between depot and other points
    for i, cust_i in enumerate(data.cust_coordinates):
        dist = compute_euclidean_distance(cust_i[0], cust_i[1], data.depot_coordinates[0], data.depot_coordinates[1], 0)
        modified_dist = dist + F/2
        model.add_link(start_point_id=0, end_point_id=i + 1, distance=modified_dist)


    # Compute the links between points
    for i,cust_i in enumerate(data.cust_coordinates):
        for j in range(i + 1, len(data.cust_coordinates)):
            dist = compute_euclidean_distance(cust_i[0],
                                              cust_i[1],
                                              data.cust_coordinates[j][0],
                                              data.cust_coordinates[j][1],
                                              0)
            model.add_link(start_point_id=i + 1,
                           end_point_id=j + 1,
                           distance=dist
                           )


    # set parameters
    model.set_parameters(time_limit=time_resolution,
                         solver_name=solver_name_input)

    ''' If you have cplex 22.1 installed on your laptop windows you have to specify
        solver path'''
    if (solver_name_input == "CPLEX" and solver_path != ""):
        model.parameters.cplex_path = solver_path

    #model.export(instance_name)

    # solve model
    model.solve()

    # export the result
    #model.solution.export(instance_name.split(".")[0] + "_result")


    if model.solution.is_defined :
      cost = model.solution.value
      m= len(model.solution.routes)
      message= model.message
      routes = []
      for route in model.solution.routes:
          routes.append({
              "Ids" :list(map(str, route.point_ids))    
                    })

    return cost, m, message, routes


def read_cvrp_instances(instance_full_path):
    """Read literature instances from CVRPLIB by giving the name of instance
       and returns dictionary containing all elements of model"""

    instance_iter = iter(
        read_instance(instance_full_path))

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
                raise Exception("EDGE_WEIGHT_TYPE : " + element + """
                is not supported (only EUC_2D)""")
        elif element == "NODE_COORD_SECTION":
            break

    vehicle_capacity = capacity_input

    # get demands and coordinates
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
                    depot_coordinates
                    )


def solve_files_in_directory(input_dir):
    
    for instance_file in os.listdir(input_dir):
        instance_path = os.path.join(input_dir, instance_file)

        with open(instance_path, 'r') as file:
            lines = file.readlines()

            # Check if the last line is "#EOF" and skip if it yes then we skip
            if lines[-1].strip() == '#EOF':
                print(f'Skipping {instance_file}, already solved.')
                continue

            assert all(not line.strip().startswith('#') for line in lines), f"Unexpected '#' character found in {instance_file}"

        start_time = time.time()
        cost, m, message, routes = solve_demo(instance_path)
        solve_time = time.time() - start_time

        unique_uid = instance_file

        with open(instance_path, 'a') as file: 
            file.write(f'\n#name {unique_uid}\n')
            file.write(f'#cost {cost}\n')
            file.write(f'#num_routes {m}\n')
            file.write(f'#message {message}\n')
            file.write(f'#solve_time {solve_time}\n')
            file.write(f'#actual_routes {routes}\n')
            file.write('#EOF\n') 

        print(f'Solution for {instance_file} appended to {instance_path}')


def parse_arguments():
    parser = argparse.ArgumentParser(description='CVRP Solver')
    parser.add_argument('input_dir', help='Please give path to the input directory')
    args = parser.parse_args()
    return args.input_dir


if __name__ == '__main__':

    input_dir_path = parse_arguments()
    solve_files_in_directory(input_dir_path)