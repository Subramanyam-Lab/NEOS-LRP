import math
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def plot_customers_and_depots(all_customers, depots, chosen_depot, closest_customers):

    customer_x, customer_y = zip(*[(cust[0], cust[1]) for cust in all_customers])
    depot_x, depot_y = zip(*depots)
    chosen_depot_x, chosen_depot_y = chosen_depot
    closest_customers_x, closest_customers_y = zip(*[(cust[0], cust[1]) for cust in closest_customers])

    plt.scatter(customer_x, customer_y, color='blue', label='Customers')
    
    plt.scatter(depot_x, depot_y, color='green', marker='s', label='Depots')

    plt.scatter(chosen_depot_x, chosen_depot_y, color='red', marker='s', label='Chosen Depot', s=100)

    plt.scatter(closest_customers_x, closest_customers_y, color='orange', label='Closest Customers', s=50)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Customers and Depots')
    plt.legend()
    plt.savefig("sampledata.png", dpi=350)

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

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_best_depot(sampled_customers, all_depots, all_depot_capacities):
    total_demand = sum(demand for _, _, demand in sampled_customers)
    best_depot = None
    min_distance = float('inf')
    
    print(f"Total demand from sampled customers: {total_demand}")
    
    for depot_idx, (depot_x, depot_y) in enumerate(all_depots):
        depot_capacity = all_depot_capacities[depot_idx]
        
        print(f"Evaluating depot at ({depot_x}, {depot_y}) with capacity {depot_capacity}")
        
        if depot_capacity >= total_demand:
            total_distance = sum(euclidean_distance(depot_x, depot_y, cust_x, cust_y) for cust_x, cust_y, _ in sampled_customers)
            
            print(f"Total distance from this depot to all sampled customers: {total_distance}")
            
            if total_distance < min_distance:
                min_distance = total_distance
                best_depot = (depot_x, depot_y)
                print(f"New best depot found at ({depot_x}, {depot_y}) with total distance: {total_distance}")
                
    return best_depot



def updated_sample_lrp_data(lrp_data, num_customers, seed=42):
    all_depots = [(int(x), int(y)) for x, y in lrp_data[2]]
    all_depot_capacities = lrp_data[5]
    all_customers = [(int(x), int(y), demand) for (x, y), demand in zip(lrp_data[3], lrp_data[6])]
    
    random.seed(seed)
    sampled_customers = random.sample(all_customers, num_customers)
    
    
    best_depot = find_best_depot(sampled_customers, all_depots, all_depot_capacities)
    
    return {
        'sampled_customers': sampled_customers,
        'best_depot': best_depot
    }

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


def find_closest_customers(chosen_depot, all_customers, num_customers):
    depot_x, depot_y = int(chosen_depot[0]), int(chosen_depot[1])
    
    # distances from the chosen depot to each customer
    distances = [(euclidean_distance(depot_x, depot_y, int(cust[0]), int(cust[1])), cust) for cust in all_customers]
    
    print("Calculated distances before sorting:", distances)

    # Sort the customers by distance
    distances.sort(key=lambda x: x[0])
    
    print("Distances after sorting:", distances)

    # Select the closest customers
    closest_customers = [cust for _, cust in distances[:num_customers]]
    
    print("Closest customers:", closest_customers)

    return closest_customers



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
        max_depot_capacity = max(lrp_data[5]) 

        num_depots_open = math.ceil(total_customer_demand / max_depot_capacity)
        mu = total_customers // num_depots_open
        sigma = 0.2 * mu
        
        print(f"Calculated mu: {mu}, sigma: {sigma}")
        num_customers_to_sample = min(max(int(np.random.normal(mu, sigma)), 1), total_customers)


        print(f"Number of customers to sample: {num_customers_to_sample}")
        all_customers = [(int(x), int(y), demand) for (x, y), demand in zip(lrp_data[3], lrp_data[6])]
        # print(all_customers)
        closest_customers = find_closest_customers(chosen_depot, all_customers, num_customers_to_sample)
        print(closest_customers)
        try:
            plot_customers_and_depots(all_customers, lrp_data[2], chosen_depot, closest_customers)
        except Exception as e:
            print(f"Error in plotting: {e}")

        output_filename = os.path.join(output_folder, f"{output_filename_prefix}_{selected_file_without_extension}_{i+1}_customers_{num_customers_to_sample}_seed_{seed + i}.txt")
        vehicle_capacity = lrp_data[4][0]
        
        updated_write_to_csv_cvrplib_format({
            'customers': closest_customers, 
            'depots': [chosen_depot]        
        }, output_filename, vehicle_capacity, num_customers_to_sample, seed + i)
        
main(
    folder_path='prodhon_dataset', 
    seed=42, 
    output_filename_prefix='ss1',
    num_samples=1,
    output_folder='training/sampledtraindata'
)

