import gurobipy as gp
from gurobipy import GRB
import math

def create_data(file_loc):
    data1 = []
    for line in open(file_loc, 'r'):
        item = line.rstrip().split()
        if len(item) != 0:
            data1.append(item)

    # Number of Customers
    no_cust = int(data1[0][0])

    # Number of Depots
    no_depot = int(data1[1][0])

    # Depot coordinates
    depot_cord = []
    ct = 2
    ed_depot = ct + no_depot
    for i in range(ct, ed_depot):
        depot_cord.append(tuple(map(float, data1[i])))

    # Customer coordinates
    cust_cord = []
    ed_cust = ed_depot + no_cust
    for j in range(ed_depot, ed_cust):
        cust_cord.append(tuple(map(float, data1[j])))

    # Vehicle Capacity
    vehicle_cap_value = int(data1[ed_cust][0])
    vehicle_cap = [vehicle_cap_value] * no_depot  # Assuming same capacity for all vehicles

    # Depot capacities
    depot_cap = []
    start = ed_cust + 1
    end = start + no_depot
    for k in range(start, end):
        depot_cap.append(int(data1[k][0]))

    # Customer Demands
    dem_end = end + no_cust
    cust_dem = []
    for l in range(end, dem_end):
        cust_dem.append(int(data1[l][0]))

    # Opening Cost of Depots
    open_dep_cost = []
    cost_end = dem_end + no_depot
    for x in range(dem_end, cost_end):
        open_dep_cost.append(int(data1[x][0]))

    route_cost = int(data1[cost_end][0])

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost]

no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost = create_data('/Users/waquarkaleem/NEOS-LRP-Codes-2/prodhon_dataset/coord20-5-1.dat')

I = range(no_depot)  # Depots
J = range(no_cust)  # Customers

V = list(I) + [no_depot + j for j in J]  # All nodes
customer_nodes = [no_depot + j for j in J]  # Customer nodes with adjusted indices

coords = {}
for i in I:
    coords[i] = depot_cord[i]
for idx, j in enumerate(customer_nodes):
    coords[j] = cust_cord[idx]

def distance(i, j):
    xi, yi = coords[i]
    xj, yj = coords[j]
    dist = 100 * math.hypot(xi - xj, yi - yj)  
    if (i in I and j in customer_nodes) or (i in customer_nodes and j in I):
        dist += 500
    return dist

E = []
for i in V:
    for j in V:
        if i != j:
            E.append((i, j))

ell = {(i, j): distance(i, j) for i, j in E}

model = gp.Model('LRP')


max_vehicles_per_depot = len(J) # say

vehicle_list = []
vehicle_to_depot = {}
vehicle_index = 0
for i_dep in I:
    for k in range(max_vehicles_per_depot):
        vehicle_list.append(vehicle_index)
        vehicle_to_depot[vehicle_index] = i_dep
        vehicle_index += 1

# Variables
y = model.addVars(I, vtype=GRB.BINARY, name='y')  # Depot open variables
x = model.addVars(I, J, vtype=GRB.BINARY, name='x')  # Customer assignment variables
v = model.addVars(vehicle_list, vtype=GRB.BINARY, name='v')  # Vehicle usage variables
z_k = model.addVars(E, vehicle_list, vtype=GRB.BINARY, name='z_k')  # Edge usage per vehicle
u = model.addVars(J, vehicle_list, vtype=GRB.BINARY, name='u')  # Vehicle visits customer

# edge usage variable z
z = model.addVars(E, vtype=GRB.BINARY, name='z')  # Overall edge usage variables

# edge usage constraints
model.addConstrs(
    (z[i, j] == gp.quicksum(z_k[i, j, vehicle] for vehicle in vehicle_list) for i, j in E),
    name='EdgeUsage'
)

# Objective
model.setObjective(
    gp.quicksum(open_dep_cost[i] * y[i] for i in I) +
    gp.quicksum(ell[i, j] * z[i, j] for i, j in E),
    GRB.MINIMIZE
)

# Constraints

# Each customer is assigned to one depot
model.addConstrs((gp.quicksum(x[i, j] for i in I) == 1 for j in J), name='Assignment')

# Customers assigned only to open depots
model.addConstrs((x[i, j] <= y[i] for i in I for j in J), name='OpenDepotAssignment')

# Depot capacity constraints
model.addConstrs(
    (gp.quicksum(cust_dem[j] * x[i, j] for j in J) <= depot_cap[i] * y[i] for i in I),
    name='DepotCapacity'
)

# Vehicle usage constraints
for vehicle in vehicle_list:
    i_dep = vehicle_to_depot[vehicle]
    model.addConstr(
        v[vehicle] <= y[i_dep],
        name=f'VehicleUsage_{vehicle}'
    )

# Ensure that vehicles start and end at their depots
for vehicle in vehicle_list:
    i_dep = vehicle_to_depot[vehicle]
    model.addConstr(
        gp.quicksum(z_k[i_dep, j, vehicle] for j in V if j != i_dep and (i_dep, j) in E) == v[vehicle],
        name=f'StartAtDepot_{vehicle}'
    )
    model.addConstr(
        gp.quicksum(z_k[j, i_dep, vehicle] for j in V if j != i_dep and (j, i_dep) in E) == v[vehicle],
        name=f'EndAtDepot_{vehicle}'
    )

# Flow conservation constraints per vehicle
for vehicle in vehicle_list:
    for v_node in V:
        model.addConstr(
            gp.quicksum(z_k[i, v_node, vehicle] for i in V if i != v_node and (i, v_node) in E) ==
            gp.quicksum(z_k[v_node, j, vehicle] for j in V if j != v_node and (v_node, j) in E),
            name=f'FlowConservation_{v_node}_{vehicle}'
        )

# Linking u and z_k variables
for j in J:
    customer_node = no_depot + j
    for vehicle in vehicle_list:
        model.addConstr(
            u[j, vehicle] == gp.quicksum(z_k[i, customer_node, vehicle] for i in V if (i, customer_node) in E),
            name=f'VisitCustomer_{j}_{vehicle}'
        )

# Each customer is visited exactly once by the vehicles
model.addConstrs((gp.quicksum(u[j, vehicle] for vehicle in vehicle_list) == 1 for j in J), name='CustomerVisited')

# Vehicle capacity constraints per vehicle
for vehicle in vehicle_list:
    i_dep = vehicle_to_depot[vehicle]
    model.addConstr(
        gp.quicksum(cust_dem[j] * u[j, vehicle] for j in J) <= vehicle_cap[i_dep] * v[vehicle],
        name=f'VehCap_{vehicle}'
    )

# Linking customer assignments with vehicle routes
for i in I:
    for j in J:
        customer_node = no_depot + j
        model.addConstr(
            x[i, j] >= gp.quicksum(z_k[i, customer_node, vehicle] for vehicle in vehicle_list if vehicle_to_depot[vehicle] == i and (i, customer_node) in E),
            name=f'AssignLink_{i}_{j}'
        )

# Ensure that arcs are used only if the vehicle is active
for vehicle in vehicle_list:
    model.addConstrs(
        (z_k[i, j, vehicle] <= v[vehicle] for i, j in E),
        name=f'ArcUse_{vehicle}'
    )

# Subtour elimination constraints (added as lazy constraints)
def subtour_elimination(model, where):
    if where == GRB.Callback.MIPSOL:
        vals_zk = model.cbGetSolution(model._z_k)
        vals_v = model.cbGetSolution(model._v)
        for vehicle in vehicle_list:
            if vals_v.get(vehicle, 0) < 0.5:
                continue  # Vehicle is not used
            edges_k = gp.tuplelist((i, j) for i, j in E if vals_zk.get((i, j, vehicle), 0) > 0.5)
            tours = get_subtours(edges_k, V)
            for tour in tours:
                if len(tour) <= 2:
                    continue  # No subtour
                model.cbLazy(
                    gp.quicksum(model._z_k[i, j, vehicle] for i in tour for j in tour if i != j and (i, j) in E) <= len(tour) - 1
                )

def get_subtours(edges, nodes):
    from collections import defaultdict

    unvisited = set(nodes)
    tours = []
    edges_dict = defaultdict(list)
    for i, j in edges:
        edges_dict[i].append(j)

    while unvisited:
        current = unvisited.pop()
        tour = [current]
        stack = [current]
        while stack:
            node = stack.pop()
            for neighbor in edges_dict[node]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
                    tour.append(neighbor)
        if len(tour) > 1:
            tours.append(tour)
    return tours

# Set the callback function and variables used in the callback
model._z_k = z_k
model._v = v
model.Params.LazyConstraints = 1
model.optimize(subtour_elimination)

# Extract and print the solution
def extract_routes(z_k_vars, vehicle_list, vehicle_to_depot, V):
    routes = []
    for vehicle in vehicle_list:
        if v[vehicle].X < 0.5:
            continue  # Vehicle not used
        i_dep = vehicle_to_depot[vehicle]
        edges_k = [(i, j) for (i, j) in E if z_k_vars[i, j, vehicle].X > 0.5]
        if not edges_k:
            continue
        # Build the route starting from depot i_dep
        route = [i_dep]
        current_node = i_dep
        visited = set()
        while True:
            visited.add(current_node)
            next_nodes = [j for i, j in edges_k if i == current_node and j not in visited]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            route.append(next_node)
            current_node = next_node
            if current_node == i_dep:
                break
        routes.append(route)
    return routes

if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
    print('Objective Value:', model.objVal)
    print('\nOpened Depots:')
    for i in I:
        if y[i].X > 0.5:
            print(f'  Depot {i} at location {coords[i]}')

    print('\nCustomer Assignments:')
    for j in J:
        for i in I:
            if x[i, j].X > 0.5:
                print(f'  Customer {j} assigned to Depot {i}')

    print('\nVehicle Routes:')
    routes = extract_routes(z_k, vehicle_list, vehicle_to_depot, V)
    for idx, route in enumerate(routes):
        node_sequence = []
        for node in route:
            if node in I:
                node_sequence.append(f'Depot {node}')
            else:
                customer_idx = node - no_depot
                node_sequence.append(f'Customer {customer_idx}')
        print(f'  Route {idx + 1}: ' + ' -> '.join(node_sequence))
else:
    print('No feasible solution found.')
