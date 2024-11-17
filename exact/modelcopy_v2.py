import gurobipy as gp
from gurobipy import GRB
import math
import matplotlib.pyplot as plt

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
    vehicle_cap_value = float(data[cust_end][0])

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

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap_value, depot_cap, cust_dem, open_dep_cost, route_cost]

def main():
    data_file = '/Users/waquarkaleem/NEOS-LRP-Codes-2/prodhon_dataset/coord20-5-1b.dat'
    data = create_data(data_file)
    if data is None:
        return 

    [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap_value, depot_cap, cust_dem, open_dep_cost, route_cost] = data

    N = no_depot + no_cust  # Total number of nodes
    I = range(no_depot)  # Depot indices
    J = range(no_depot, N)  # Customer indices
    V = range(N)  # All node indices

    node_coords = depot_cord + cust_cord  

    c = {}
    for i in V:
        xi, yi = node_coords[i]
        for j in V:
            if i != j:
                xj, yj = node_coords[j]
                distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                c[(i,j)] = int(100 * distance)  
            else:
                c[(i,j)] = 0  

    model = gp.Model("CLRP")

    x = model.addVars(V, V, vtype=GRB.BINARY, name='x')
    w = model.addVars(J, I, vtype=GRB.BINARY, name='w')
    y = model.addVars(I, vtype=GRB.BINARY, name='y')
    u = model.addVars(J, lb=0, ub=vehicle_cap_value, vtype=GRB.CONTINUOUS, name='u') # load variables for customers only

    obj = gp.quicksum(x[i,j] * c[i,j] for i in V for j in V if i != j) + \
          gp.quicksum(x[i,j] * route_cost for i in I for j in J) + \
          gp.quicksum(y[i] * open_dep_cost[i] for i in I)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Constraint (2): Each customer is arrived at exactly once
    for j in J:
        model.addConstr(gp.quicksum(x[i,j] for i in V if i != j) == 1, name=f"c2_{j}")

    # Constraint (3): Each customer departs exactly once
    for j in J:
        model.addConstr(gp.quicksum(x[j,i] for i in V if i != j) == 1, name=f"c3_{j}")

    # Constraint (4): Each customer is assigned to exactly one depot
    for j in J:
        model.addConstr(gp.quicksum(w[j,i] for i in I) == 1, name=f"c4_{j}")

    # Constraint (5): Customers can only be served by open depots
    for i in I:
        for j in J:
            model.addConstr(w[j,i] <= y[i], name=f"c5_{i}_{j}")

    # Constraint (6): Depot capacity constraints
    for i in I:
        model.addConstr(
            gp.quicksum(w[j,i] * cust_dem[j - no_depot] for j in J) <= depot_cap[i],
            name=f"c6_{i}"
        )

    # Constraints (7) and (8)
    for i in I:
        for j in J:
            for k in J:
                if j != k:
                    model.addConstr(x[j,k] <= 1 - w[j,i] + w[k,i], name=f"c7_{i}_{j}_{k}")
                    model.addConstr(x[j,k] <= 1 + w[j,i] - w[k,i], name=f"c8_{i}_{j}_{k}")

    # Constraint (9): Route starts at the depot serving the customers
    for i in I:
        for j in J:
            model.addConstr(x[i,j] <= w[j,i], name=f"c9_{i}_{j}")

    # Constraint (10): Route ends at the depot serving the customers
    for i in I:
        for j in J:
            model.addConstr(x[j,i] <= w[j,i], name=f"c10_{i}_{j}")

    # Accumulated load constraints
    # Load bounds at customers
    for j in J:
        model.addConstr(u[j] >= cust_dem[j - no_depot], name=f"u_lb_{j}")
        model.addConstr(u[j] <= vehicle_cap_value, name=f"u_ub_{j}")

    # Load accumulation constraints between customers
    for i in J:
        for j in J:
            if i != j:
                model.addConstr(
                    u[i] - u[j] + vehicle_cap_value * x[i,j] <= vehicle_cap_value - cust_dem[j - no_depot],
                    name=f"mtz_{i}_{j}"
                )

    # Load constraints for arcs from depots to customers
    for i in I:
        for j in J:
            model.addConstr(
                u[j] >= cust_dem[j - no_depot] * x[i,j],
                name=f"load_from_depot_{i}_{j}"
            )

    model.setParam('MIPGap', 0.01)  

    try:
        model.optimize()
    except KeyboardInterrupt:
        print("Solver interrupted by user.")

    if model.SolCount > 0:
        print("Feasible solution found.")
        if model.status == GRB.OPTIMAL:
            print("Optimal objective value:", model.objVal)
        else:
            print("Best objective value found:", model.objVal)
        x_vals = model.getAttr('x', x)
        w_vals = model.getAttr('x', w)
        y_vals = model.getAttr('x', y)
        u_vals = model.getAttr('x', u)

        depot_opening_cost = 0
        routing_cost = 0
        fixed_route_cost = 0

        for i in I:
            if y_vals[i] > 0.5:
                depot_opening_cost += open_dep_cost[i]
        print("Total Depot Opening Cost:", depot_opening_cost)

        for i in V:
            for j in V:
                if i != j and x_vals[i,j] > 0.5:
                    routing_cost += c[i,j]
        print("Total Routing Cost:", routing_cost)

        for i in I:
            num_vehicles = 0
            for j in J:
                if x_vals[i,j] > 0.5:
                    num_vehicles += 1
            fixed_route_cost += num_vehicles * route_cost
        print("Total Fixed Route Cost:", fixed_route_cost)

        total_calculated_cost = depot_opening_cost + routing_cost + fixed_route_cost
        print("Total Calculated Cost:", total_calculated_cost)

        if abs(total_calculated_cost - model.objVal) <= 1e-6:
            print("Total calculated cost matches the model's objective value.")
        else:
            print("Warning: Total calculated cost does not match the model's objective value.")

        print("\nOpened depots:")
        for i in I:
            if y_vals[i] > 0.5:
                print(f"Depot {i}")
        print("\nRoutes:")
        routes = []  
        route_demands = []  
        for i in I:
            if y_vals[i] > 0.5:
                for j in J:
                    if x_vals[i,j] > 0.5:
                        route = [i, j]
                        demand = cust_dem[j - no_depot]
                        next_node = j
                        while True:
                            for k in V:
                                if k != next_node and x_vals[next_node,k] > 0.5:
                                    route.append(k)
                                    next_node = k
                                    if k in J:
                                        demand += cust_dem[k - no_depot]
                                    break
                            else:
                                break 
                            if next_node in I:
                                break
                        print(" -> ".join(str(node) for node in route))
                        routes.append(route) 
                        route_demands.append((route, demand))

        # 1. Each customer is assigned to exactly one depot
        for j in J:
            assigned_depots = [i for i in I if w_vals[j,i] > 0.5]
            assert len(assigned_depots) == 1, f"Customer {j} is assigned to {len(assigned_depots)} depots."

        # 2. Each customer is visited exactly once (arrived and departed)
        for j in J:
            arrivals = [i for i in V if x_vals[i,j] > 0.5]
            departures = [k for k in V if x_vals[j,k] > 0.5]
            assert len(arrivals) == 1, f"Customer {j} has {len(arrivals)} arrivals."
            assert len(departures) == 1, f"Customer {j} has {len(departures)} departures."

        # 3. Depot capacities are not exceeded
        for i in I:
            assigned_customers = [j for j in J if w_vals[j,i] > 0.5]
            total_demand = sum(cust_dem[j - no_depot] for j in assigned_customers)
            assert total_demand <= depot_cap[i] + 1e-6, f"Depot {i} capacity exceeded: {total_demand} > {depot_cap[i]}."

        # 4. Vehicle capacity constraints
        for (route, demand) in route_demands:
            assert demand <= vehicle_cap_value + 1e-6, f"Vehicle capacity exceeded on route {route}: {demand} > {vehicle_cap_value}."

        # 5. Routes start and end at the correct depots
        for route in routes:
            depot_start = route[0]
            depot_end = route[-1]
            assert depot_start in I, f"Route does not start at a depot: starts at {depot_start}."
            assert depot_end in I, f"Route does not end at a depot: ends at {depot_end}."

        # 6. Total demand matches sum of customer demands
        total_assigned_demand = sum(cust_dem[j - no_depot] for j in J)
        total_served_demand = sum(cust_dem[j - no_depot] for j in J if any(w_vals[j,i] > 0.5 for i in I))
        assert abs(total_assigned_demand - total_served_demand) <= 1e-6, "Total assigned demand does not match total customer demands."

        plt.figure(figsize=(10,8))

        customer_x = [node_coords[j][0] for j in J]
        customer_y = [node_coords[j][1] for j in J]
        plt.scatter(customer_x, customer_y, c='blue', label='Customers')

        depot_x = [node_coords[i][0] for i in I]
        depot_y = [node_coords[i][1] for i in I]
        opened_depot_x = [node_coords[i][0] for i in I if y_vals[i] > 0.5]
        opened_depot_y = [node_coords[i][1] for i in I if y_vals[i] > 0.5]
        closed_depot_x = [node_coords[i][0] for i in I if y_vals[i] <= 0.5]
        closed_depot_y = [node_coords[i][1] for i in I if y_vals[i] <= 0.5]
        plt.scatter(opened_depot_x, opened_depot_y, c='red', marker='s', s=100, label='Opened Depots')
        plt.scatter(closed_depot_x, closed_depot_y, c='gray', marker='s', s=100, label='Closed Depots')

        for route in routes:
            route_x = [node_coords[node][0] for node in route]
            route_y = [node_coords[node][1] for node in route]
            plt.plot(route_x, route_y, c='green', linestyle='-', linewidth=1)

        plt.title('CLRP Solution Plot')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.savefig("clrp.png", dpi=320)
        plt.close()

    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()
