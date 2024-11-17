import gurobipy as gp
from gurobipy import GRB
import math
from collections import defaultdict

def create_data(file_loc):
    data = []
    try:
        with open(file_loc, 'r') as file:
            for line in file:
                items = line.strip().split()
                if items:
                    data.append(items)
    except FileNotFoundError:
        print(f"Error: File not found at location {file_loc}")
        return None

    # Number of Customers
    no_cust = int(data[0][0])

    # Number of Depots
    no_depot = int(data[1][0])

    # Depot coordinates
    depot_cord = []
    depot_start = 2
    depot_end = depot_start + no_depot
    for i in range(depot_start, depot_end):
        depot_cord.append(tuple(map(float, data[i])))

    # Customer coordinates
    cust_cord = []
    cust_start = depot_end
    cust_end = cust_start + no_cust
    for i in range(cust_start, cust_end):
        cust_cord.append(tuple(map(float, data[i])))

    # Vehicle Capacity
    vehicle_cap = []
    vehicle_cap_value = float(data[cust_end][0])
    vehicle_cap = [vehicle_cap_value] * no_depot  # Assuming same capacity for all depots

    # Depot capacities
    depot_cap = []
    depot_cap_start = cust_end + 1
    depot_cap_end = depot_cap_start + no_depot
    for i in range(depot_cap_start, depot_cap_end):
        depot_cap.append(float(data[i][0]))

    # Customer Demands
    cust_dem = []
    cust_dem_start = depot_cap_end
    cust_dem_end = cust_dem_start + no_cust
    for i in range(cust_dem_start, cust_dem_end):
        cust_dem.append(float(data[i][0]))

    # Opening Cost of Depots
    open_dep_cost = []
    open_dep_cost_start = cust_dem_end
    open_dep_cost_end = open_dep_cost_start + no_depot
    for i in range(open_dep_cost_start, open_dep_cost_end):
        open_dep_cost.append(float(data[i][0]))

    # Fixed cost per route
    route_cost = float(data[open_dep_cost_end][0])

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost]

def calculate_euclidean_distance(coord1, coord2):
    return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

def build_cost_matrix(V, node_cord):
    c = {}
    for i in V:
        for j in V:
            if i != j:
                c[i, j] = calculate_euclidean_distance(node_cord[i], node_cord[j])
            else:
                c[i, j] = 0  # No cost for staying at the same node
    return c

def extract_routes(model, x_vars, y_vars, depot_cord, I, J, V, no_depot, no_cust):
    opened_depots = [i for i in I if y_vars[i].X > 0.5]
    routes = []
    assigned_customers = set()

    x_sol = model.getAttr('X', x_vars)

    for depot in opened_depots:
        for j in J:
            customer_node = j + no_depot
            if x_sol[depot, customer_node] > 0.5 and j not in assigned_customers:
                route = [depot]
                current = customer_node
                route.append(current)
                assigned_customers.add(j)
                while True:
                    next_node = None
                    for k in V:
                        if x_sol[current, k] > 0.5:
                            next_node = k
                            break
                    if next_node is None:
                        break
                    if next_node in I:
                        route.append(next_node)
                        break
                    customer_idx = next_node - no_depot
                    if customer_idx in assigned_customers:
                        # Prevent cycles
                        break
                    route.append(customer_idx)
                    assigned_customers.add(customer_idx)
                    current = next_node
                routes.append(route)

    return opened_depots, routes

def main():
    # Replace the file path with your actual data file location
    data = create_data('/Users/waquarkaleem/NEOS-LRP-Codes-2/prodhon_dataset/coord20-5-1.dat')
    if data is None:
        return

    no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost = data

    # Define sets
    I = range(no_depot)                     # Depots indexed from 0 to no_depot-1
    J = range(no_cust)                      # Customers indexed from 0 to no_cust-1
    V = range(no_depot + no_cust)           # All nodes: depots + customers

    # Create node coordinates list
    node_cord = depot_cord + cust_cord

    # Calculate cost matrix c_{ij} as Euclidean distances
    c = build_cost_matrix(V, node_cord)

    # Initialize the Gurobi model
    model = gp.Model("CLRP")

    # Set model to handle lazy constraints
    model.setParam('LazyConstraints', 1)

    # Decision Variables
    # x_{ij}: whether a vehicle travels from node i to node j
    x = model.addVars(V, V, vtype=GRB.BINARY, name="x")

    # w_{ij}: whether customer j is served by depot i
    w = model.addVars(I, J, vtype=GRB.BINARY, name="w")

    # y_i: whether depot i is opened
    y = model.addVars(I, vtype=GRB.BINARY, name="y")

    # z_{ij}: whether a route from depot i to customer j is used
    z = model.addVars(I, J, vtype=GRB.BINARY, name="z")

    # u_i: number of vehicles used at depot i
    u = model.addVars(I, vtype=GRB.INTEGER, name="u")

    # Objective Function
    # Minimize total routing cost + fixed vehicle cost + depot opening cost
    model.setObjective(
        gp.quicksum(x[i, j] * c[i, j] for i in V for j in V) +
        gp.quicksum(z[i, j] * route_cost for i in I for j in J) +
        gp.quicksum(y[i] * open_dep_cost[i] for i in I),
        GRB.MINIMIZE
    )

    # Constraint (1): Each customer has exactly one incoming vehicle
    model.addConstrs(
        (gp.quicksum(x[i, j + no_depot] for i in V) == 1 for j in J),
        name="OneIncoming"
    )

    # Constraint (2): Each customer has exactly one outgoing vehicle
    model.addConstrs(
        (gp.quicksum(x[j + no_depot, k] for k in V) == 1 for j in J),
        name="OneOutgoing"
    )

    # Constraint (3): Each customer is assigned to exactly one depot
    model.addConstrs(
        (gp.quicksum(w[i, j] for i in I) == 1 for j in J),
        name="OneDepotAssignment"
    )

    # Constraint (4): Customers can only be served by open depots
    model.addConstrs(
        (w[i, j] <= y[i] for i in I for j in J),
        name="DepotOpened"
    )

    # Constraint (5): Depot capacity
    model.addConstrs(
        (gp.quicksum(w[i, j] * cust_dem[j] for j in J) <= depot_cap[i] for i in I),
        name="DepotCapacity"
    )

    # Constraint (6): Link z to x variables (route activation)
    model.addConstrs(
        (z[i, j] >= x[i, j + no_depot] for i in I for j in J),
        name="RouteActivation"
    )

    # Constraint (7): Prevent self-loops at depots
    model.addConstrs(
        (x[i, i] == 0 for i in I),
        name="NoDepotSelfLoop"
    )

    # Constraint (8): Ensure that if a vehicle travels between two customers,
    # both are assigned to the same depot
    model.addConstrs(
        (x[j + no_depot, k + no_depot] <= gp.quicksum(w[i, j] for i in I) for j in J for k in J if j != k),
        name="SameDepotAssignment1"
    )
    model.addConstrs(
        (x[j + no_depot, k + no_depot] <= gp.quicksum(w[i, k] for i in I) for j in J for k in J if j != k),
        name="SameDepotAssignment2"
    )

    # Constraint (9): Route starts at the depot serving the customer
    model.addConstrs(
        (x[i, j + no_depot] <= w[i, j] for i in I for j in J),
        name="StartAtDepot"
    )

    # Constraint (10): Route ends at the depot serving the customer
    model.addConstrs(
        (x[j + no_depot, i] <= w[i, j] for i in I for j in J),
        name="EndAtDepot"
    )

    # Constraint (11): Link number of vehicles to depot assignments
    model.addConstrs(
        (u[i] >= gp.quicksum(w[i, j] * cust_dem[j] for j in J) / vehicle_cap[i] for i in I),
        name="VehicleCount"
    )

    # Prevent arcs between different depots
    model.addConstrs(
        (x[i, j] == 0 for i in I for j in I if i != j),
        name="NoArcBetweenDepots"
    )

    # Assign variables to the model for callback access
    model._x_vars = x
    model._y_vars = y
    model._w_vars = w
    model._z_vars = z
    model._u_vars = u
    model._V = V
    model._I = I
    model._J = J
    model._no_depot = no_depot
    model._no_cust = no_cust
    model._cust_dem = cust_dem
    model._vehicle_cap = vehicle_cap

    # Optimize the model with a callback for subtour elimination
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # Retrieve the solution
            x_vals = model.cbGetSolution(model._x_vars)

            # Build adjacency list based on x_vals
            adj = defaultdict(list)
            for i in model._V:
                for j in model._V:
                    if i != j and x_vals[i, j] > 0.5:
                        adj[i].append(j)

            # Function to find all subtours
            def find_subtours(adj_list, nodes):
                visited = set()
                subtours = []

                for node in nodes:
                    if node not in visited:
                        current_subtour = []
                        stack = [node]
                        while stack:
                            current = stack.pop()
                            if current not in visited:
                                visited.add(current)
                                current_subtour.append(current)
                                stack.extend(adj_list[current])
                        if len(current_subtour) > 1:
                            subtours.append(current_subtour)
                return subtours

            # Only consider customer nodes for subtours
            customer_nodes = set(range(model._no_depot, model._no_depot + model._no_cust))
            subtours = find_subtours(adj, customer_nodes)

            # Iterate through all detected subtours
            for s in subtours:
                # Check if the subtour is a valid cycle (no depots connected)
                connects_to_depot = False
                for n in s:
                    for depot in model._I:
                        if x_vals[depot, n] > 0.5 or x_vals[n, depot] > 0.5:
                            connects_to_depot = True
                            break
                    if connects_to_depot:
                        break
                if not connects_to_depot and len(s) > 1:
                    # Add the subtour elimination constraint
                    model.cbLazy(
                        gp.quicksum(model._x_vars[i, j] for i in s for j in s if i != j) <= len(s) - 1
                    )
                    # Optional: Uncomment the next line for debugging
                    # print(f"Subtour detected and eliminated: {s}")

    # Optimize with the callback
    model.optimize(subtourelim)

    # Check if a feasible solution was found
    if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("\nOptimal Solution Found:")

        # Retrieve and print depot openings with detailed information
        opened_depots = [i for i in I if y[i].X > 0.5]
        print("Opened Depots:")
        for depot in opened_depots:
            print(f"Depot {depot}:")
            print(f"  Coordinates: {depot_cord[depot]}")
            print(f"  Opening Cost: {open_dep_cost[depot]}")

        # Extract routes using the improved function
        opened_depots, routes = extract_routes(model, x, y, depot_cord, I, J, V, no_depot, no_cust)

        # Print all routes
        if routes:
            print("\nRoutes:")
            for idx, route in enumerate(routes):
                # Map node indices to meaningful labels
                route_labels = []
                for node in route:
                    if node in I:
                        route_labels.append(f"Depot {node}")
                    else:
                        route_labels.append(f"Customer {node}")
                route_str = " -> ".join(route_labels)
                print(f"Route {idx + 1}: {route_str}")
        else:
            print("\nNo routes were extracted. Please verify the model and extraction logic.")

        # Print total cost
        print(f"\nTotal Cost: {model.ObjVal}")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()
