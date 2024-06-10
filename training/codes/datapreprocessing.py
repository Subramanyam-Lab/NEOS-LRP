import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class InstanceNew:
    def __init__(self, instance_str, max_len):

        problem_str, metadata_str = instance_str.split("EOF\n", 1)
        problem_lines = problem_str.strip().split("\n")
        metadata_lines = metadata_str.strip().split("\n")
        name_line = next(line for line in instance_str.split('\n') if line.startswith('NAME :'))
        self.name = name_line.split(':')[1].strip()

        self.capacity = int(next(line.split(":")[1].strip() for line in problem_lines if line.startswith("CAPACITY :")))
        
        # Parsing NODE_COORD_SECTION
        node_coord_start = next(i for i, line in enumerate(problem_lines) if line.startswith("NODE_COORD_SECTION")) + 1
        demand_start = next(i for i, line in enumerate(problem_lines) if line.startswith("DEMAND_SECTION"))

        self.customers = []
        self.depots={}
        first_line = True
        for line in problem_lines[node_coord_start:demand_start]:
            x, y = map(int, line.strip().split()[1:])
            if first_line:
                self.depot = {'x': x, 'y': y}
                first_line = False
            else:
                self.customers.append({'x': x, 'y': y})

        for line in problem_lines[demand_start + 1:]:
            if line.startswith("DEPOT_SECTION"):
                break
            parts = line.strip().split()
            index, demand = map(int, parts)
            if index == 1:  # Skip the depot
                continue
            self.customers[index - 2]['demand'] = demand / self.capacity  

        # Parsing metadata
        self.metadata = {}
        for line in metadata_lines:
            if line.startswith("#"):
                parts = line[1:].strip().split(" ", 1)
                if len(parts) == 2:
                    key, value = parts
                    self.metadata[key] = value

        self.solve_time = float(self.metadata.get('solve_time', 0.0))
        # print(self.solve_time)

        xmin = min([customer['x'] for customer in self.customers] + [self.depot['x']])
        xmax = max([customer['x'] for customer in self.customers] + [self.depot['x']])
        ymin = min([customer['y'] for customer in self.customers] + [self.depot['y']])
        ymax = max([customer['y'] for customer in self.customers] + [self.depot['y']])

        Fi = max(xmax - xmin, ymax - ymin)

        if Fi == 0:
            print("Fi is zero for instance:", self.name)
            print("Instance Details:")
            print("Depot:", self.depot)
            print("Customers:", self.customers)
            print("Capacity:", self.capacity)
            print("X range: xmin =", xmin, ", xmax =", xmax)
            print("Y range: ymin =", ymin, ", ymax =", ymax)
        depot_x = self.depot['x']
        depot_y = self.depot['y']

        self.cost = float(self.metadata.get('cost', 0))
        tolerance = 1e-6
        if Fi < tolerance:
            for customer in self.customers:
                customer['x'] = customer['x'] - self.depot['x']
                customer['y'] = customer['y'] - self.depot['y']
        else:
            for customer in self.customers:
                customer['x'] = (customer['x'] - self.depot['x']) / Fi
                customer['y'] = (customer['y'] - self.depot['y']) / Fi
            
            self.cost /= Fi

        # Padding
        pad_len = max_len - len(self.customers)
        if pad_len > 0:
            special_pad_values = {'x': -1.0e+04, 'y': -1.0e+04, 'demand': -1.0e+04}
            zero_pad_values = {'x': 0, 'y': 0, 'demand': 0}
            
            self.customers.append(special_pad_values)
            for _ in range(pad_len - 1):
                self.customers.append(zero_pad_values)

        self.num_routes = int(self.metadata.get('num_routes', 0))
        message = self.metadata.get('message', '')

        if message != "The solution found is optimal.":
            raise ValueError(f"Invalid solution message: {message}")

        self.depot['x'] = 0.0
        self.depot['y'] = 0.0
        self.depot['demand'] = 0.0 

        training_customers = self.customers

        self.training_data = {
            'x': [customer['x'] for customer in training_customers],
            'y': [customer['y'] for customer in training_customers],
            'demand': [customer.get('demand', 0.0) for customer in training_customers], 
            'cost': self.cost}

def preprocess_data(file_path, num_instances, seed=42):

    random.seed(seed)
    training_data = []

    with open(file_path, 'r') as file:
        content = file.read()
        instances = content.split("#EOF")[:-1]

    selected_instances = random.sample(instances, min(num_instances, len(instances)))

    max_len = 0    
    for instance_str in selected_instances:
        try:
            instance = InstanceNew(instance_str, max_len)
            max_len = max(max_len, len(instance.customers))
            
        except ValueError as e:
            print(f"Discarding instance during max_len computation: {e}")
            print(f"The problematic instance_str is: {instance_str[:100]}...")  

    for instance_str in selected_instances:
        try:
            instance = InstanceNew(instance_str, max_len)

            training_data.append(instance.training_data)

        except ValueError as e:
            print(f"Discarding instance during training data collection: {e}")
            print(f"The problematic instance_str is: {instance_str[:100]}...")  
            continue

    training_data = pd.DataFrame(training_data)  
    print(f"Data preprocessing completed. Total instances selected: {len(selected_instances)}")

    training_data_np = np.vstack(training_data.apply(lambda x: np.array(list(x), dtype=object), axis=1))
    X = training_data_np[:, :-1]

    Y = training_data_np[:, -1:]

    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = np.array([np.column_stack(instance) for instance in X_train])
    X_val = np.array([np.column_stack(instance) for instance in X_val])
    X_test = np.array([np.column_stack(instance) for instance in X_test])

    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)   

    return X_train, X_val, X_test, y_train, y_val, y_test
