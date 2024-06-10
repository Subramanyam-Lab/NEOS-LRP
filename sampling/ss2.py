import math
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def plot_customers_and_depots(all_customers, depots, chosen_depot, selected_customers):
    customer_x, customer_y = zip(*[(int(cust[0]), int(cust[1])) for cust in all_customers])

    depot_x, depot_y = zip(*[(int(dep[0]), int(dep[1])) for dep in depots])

    chosen_depot_x, chosen_depot_y = int(chosen_depot[0]), int(chosen_depot[1])

    selected_customers_x, selected_customers_y = zip(*[(int(cust[0]), int(cust[1])) for cust in selected_customers])

    all_x = customer_x + depot_x + (chosen_depot_x,)
    all_y = customer_y + depot_y + (chosen_depot_y,)

    plt.xlim(min(all_x), max(all_x))
    plt.ylim(min(all_y), max(all_y))

    plt.scatter(customer_x, customer_y, color='blue', label='Customers')

    plt.scatter(depot_x, depot_y, color='green', marker='s', label='Depots')

    plt.scatter(chosen_depot_x, chosen_depot_y, color='red', marker='s', label='Chosen Depot', s=100)

    plt.scatter(selected_customers_x, selected_customers_y, color='orange', label='Selected Customers', s=50)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Customers and Depots')
    plt.legend()

    plt.savefig("sampledata_6.png", dpi=350)


def create_data(file_loc):
    data1=[]
    for line in open(file_loc, 'r'):
        item = line.rstrip().split()
        if len(item)!=0:
            data1.append(item)

    
    # Number of Customers
    no_cust= int(data1[0][0]) 

    # Number of Depots
    no_depot= int(data1[1][0])

    #depot coordinates
    depot_cord=[]
    ct=2
    ed_depot=ct+no_depot
    for i in range(ct,ed_depot):
        depot_cord.append(tuple(data1[i]))

    #Customer coordinates
    cust_cord=[]
    ed_cust= ed_depot+no_cust
    for j in range(ed_depot,ed_cust):
        cust_cord.append(tuple(data1[j]))

    # Vehicle Capacity
    vehicle_cap= int(data1[ed_cust][0])
    vehicle_cap=[vehicle_cap]*no_depot

    #Depot capacities
    depot_cap=[]
    start=ed_cust+1
    end=start+no_depot
    for k in range(start,end):
        depot_cap.append(int(data1[k][0]))

    # Customer Capacities
    dem_end=end+no_cust
    cust_dem=[]
    for l in range(end,dem_end):
        cust_dem.append(int(data1[l][0]))

    # Opening Cost of Depots
    open_dep_cost=[]
    cost_end=dem_end+no_depot
    for x in range (dem_end,cost_end):
        open_dep_cost.append(int(data1[x][0]))

    route_cost=int(data1[cost_end][0])

    return [no_cust,no_depot,depot_cord,cust_cord,vehicle_cap,depot_cap,cust_dem,open_dep_cost,route_cost]

def read_lrp_instance(file_loc):
    return create_data(file_loc)

def updated_write_to_csv_cvrplib_format(sampled_data, filename, vehicle_capacity, num_customers_to_sample, seed):
    just_filename = os.path.basename(filename)
    rows = []
    rows.append({'Column1': f'NAME : {just_filename}'})

    rows.append({'Column1': 'COMMENT : None'})
    rows.append({'Column1': 'TYPE : CVRP'})
    rows.append({'Column1': f'DIMENSION : {len(sampled_data["customers"]) + 1}'})

    rows.append({'Column1': 'EDGE_WEIGHT_TYPE : EUC_2D'})
    rows.append({'Column1': f'CAPACITY : {vehicle_capacity}'})
    
    rows.append({'Column1': 'NODE_COORD_SECTION'})

    depot = sampled_data['depots'][0]
    rows.append({'Column1': f"1 {depot[0]} {depot[1]}"})

    
    for i, point in enumerate(sampled_data['customers'], start=2):
        rows.append({'Column1': f"{i} {point[0]} {point[1]}"})

    rows.append({'Column1': 'DEMAND_SECTION'})
    
    rows.append({'Column1': "1 0"})
    
    for i, point in enumerate(sampled_data['customers'], start=2):
        rows.append({'Column1': f"{i} {point[2]}"})


    rows.append({'Column1': 'DEPOT_SECTION'})
    rows.append({'Column1': "1"})  # Depot ID
    rows.append({'Column1': '-1'})  # EOF for DEPOT_SECTION
    
    rows.append({'Column1': 'EOF'})
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, header=False)

def main(folder_path, seed, output_filename_prefix, num_samples, output_folder):
    for i in range(num_samples):
        all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        random.seed(seed + i)
        selected_file_with_extension = random.choice(all_files)
        selected_file_without_extension = selected_file_with_extension.split('.')[0]

        print(f"Selected File: {selected_file_with_extension}")
        
        lrp_data = create_data(os.path.join(folder_path, selected_file_with_extension))
        
        # Randomly pick a depot
        chosen_depot = random.choice(lrp_data[2])

        total_customers = lrp_data[0]
        total_customer_demand = sum(lrp_data[6])  
        min_depot_capacity = min(lrp_data[5]) 
        # print(lrp_data[5])
        # print(f"Min depot capacity: {min_depot_capacity}")
        # print(f"Choosen depot: {chosen_depot}")

        num_depots_open = math.ceil(total_customer_demand / min_depot_capacity)
        mu = total_customers // num_depots_open
        sigma = 0.2 * mu
        
        print(f"Calculated mu: {mu}, sigma: {sigma}")

        num_customers_to_sample = min(max(int(np.random.normal(mu, sigma)), 1), total_customers)
        # print(f"Number of customers to sample: {num_customers_to_sample}")

        all_customers = [(int(x), int(y), demand) for (x, y), demand in zip(lrp_data[3], lrp_data[6])]

        customer_distances = []
        for customer in all_customers:
            distance = euclidean_distance(int(chosen_depot[0]), int(chosen_depot[1]), customer[0], customer[1])
            customer_distances.append((customer, distance))

        sorted_customers = sorted(customer_distances, key=lambda x: x[1])

        # print(sorted_customers)

        depot_index = lrp_data[2].index(chosen_depot)
        chosen_depot_capacity = lrp_data[5][depot_index] 
                    
        selected_customers = []
        D_total = 0

        for customer, distance in sorted_customers:
            if D_total + customer[2] <= chosen_depot_capacity and len(selected_customers) < num_customers_to_sample:
                selected_customers.append(customer)
                D_total += customer[2]
        # print(f"D total in the end is {D_total}")
        print(f"Actual number of customers selected {len(selected_customers)}")

        # try:
        #     plot_customers_and_depots([cust[:2] for cust in all_customers], lrp_data[2], chosen_depot, [cust[:2] for cust in selected_customers])
        # except Exception as e:
        #     print(f"Error in plotting: {e}")

        output_filename = os.path.join(output_folder, f"{output_filename_prefix}_{selected_file_without_extension}_customers_{len(selected_customers)}_seed_{seed + i}.txt")

        vehicle_capacity = lrp_data[4][0]
        
        updated_write_to_csv_cvrplib_format({
            'customers': selected_customers, 
            'depots': [chosen_depot]        
        }, output_filename, vehicle_capacity, len(selected_customers), seed + i)
        
main(

    folder_path='prodhon_dataset', 
    seed=42, 
    output_filename_prefix='ss2',
    num_samples=10,
    output_folder='training/sampledtraindata'
)

