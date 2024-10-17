# dataparse.py

import math
import numpy as np
import pandas as pd

def create_data(file_loc):
    data1 = []
    with open(file_loc, 'r') as file:
        for line in file:
            item = line.strip().split()
            if item:
                data1.append(item)
    
    # Number of Customers
    no_cust = int(float(data1[0][0]))
    
    # Number of Depots
    no_depot = int(float(data1[1][0]))
    
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
    vehicle_cap = int(data1[ed_cust][0])
    vehicle_cap = [vehicle_cap] * no_depot
    
    # Depot capacities #B1
    depot_cap = []
    start = ed_cust + 1
    end = start + no_depot
    for k in range(start, end):
        depot_cap.append(int(float(data1[k][0])))
   
    # Customer Capacities
    dem_end = end + no_cust
    cust_dem = []
    for l in range(end, dem_end):
        cust_dem.append(int(float(data1[l][0])))
    
    # Opening Cost of Depots #B1
    open_dep_cost = []
    cost_end = dem_end + no_depot
    for x in range(dem_end, cost_end):
        open_dep_cost.append(float(data1[x][0]))
    
    route_cost = float(data1[cost_end][0])
    
    rc_cal_index = int(data1[cost_end + 1][0])
    
    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost, rc_cal_index]

def dist_calc(cord_set1, cord_set2, rc_cal_index):
    distances = {}
    for from_counter, from_node in enumerate(cord_set1):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(cord_set2):
            if rc_cal_index == 0:
                distances[from_counter][to_counter] = int(100 * math.hypot(from_node[0] - to_node[0], from_node[1] - to_node[1]))
            else:
                distances[from_counter][to_counter] = int(math.hypot(from_node[0] - to_node[0], from_node[1] - to_node[1]))
    return distances

def new_dist_calc(cord_set, rc_cal_index):
    distances = {}
    sum1 = 0
    for j in cord_set:
        distances[j] = {}
        for i in cord_set[j]:
            if rc_cal_index == 0:
                distances[j][i] = 100 * math.sqrt(cord_set[j][i][0]**2 + cord_set[j][i][1]**2)
                sum1 += distances[j][i]
            else:
                distances[j][i] = math.sqrt(cord_set[j][i][0]**2 + cord_set[j][i][1]**2)
                sum1 += distances[j][i]
    return distances, sum1

def normalize_coord(cord_set1, cord_set2, rc_cal_index, min_max=None):
    """
    Normalizes coordinates using predefined min and max values if provided.
    
    Args:
        cord_set1 (list of tuples): Depot coordinates.
        cord_set2 (list of tuples): Customer coordinates.
        rc_cal_index (int): Route cost calculation index.
        min_max (dict, optional): Predefined min and max values for normalization.
    
    Returns:
        tuple: (normalized_coordinates, dist_fac_cust, big_m, normalization_factors)
    """
    coor = pd.DataFrame()
    mod_coord = {}
    fi_list = []
    
    for j in range(len(cord_set1)):
        mod_coord[j] = {}
        max_x = -math.inf
        min_x = math.inf
        max_y = -math.inf
        min_y = math.inf
        for i in range(len(cord_set2)):
            x = cord_set2[i][0] - cord_set1[j][0]
            y = cord_set2[i][1] - cord_set1[j][1]
            mod_coord[j][i] = (x, y)
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
        fi = max(max_x - min_x, max_y - min_y)
        fi_list.append(fi)
        print(f"Depot {j}: Max X={max_x}, Min X={min_x}, Max Y={max_y}, Min Y={min_y}, fi={fi}")
    
    # Normalize coordinates
    for j in mod_coord:
        for i in mod_coord[j]:
            x, y = mod_coord[j][i]
            if min_max:
                # Apply Min-Max normalization
                norm_x = (x - min_max['min_x']) / (min_max['max_x'] - min_max['min_x']) if (min_max['max_x'] - min_max['min_x']) != 0 else 0
                norm_y = (y - min_max['min_y']) / (min_max['max_y'] - min_max['min_y']) if (min_max['max_y'] - min_max['min_y']) != 0 else 0
                mod_coord[j][i] = [norm_x, norm_y]
            else:
                # Apply existing normalization
                fi = fi_list[j]
                mod_coord[j][i] = [x / fi, y / fi]
    
    dist_fac_cust, big_m = new_dist_calc(mod_coord, rc_cal_index)
    coor['normal'] = mod_coord
    print("Big M value:", big_m)
    print("Normalization factors for all depots:", fi_list)
    return coor, dist_fac_cust, big_m, fi_list

def norm_data(cord_set1, cord_set2, veh_cap, cust_dem, rc_cal_index, min_max=None):
    """
    Normalizes data and prepares facility dictionary.
    
    Args:
        cord_set1 (list of tuples): Depot coordinates.
        cord_set2 (list of tuples): Customer coordinates.
        veh_cap (list): Vehicle capacities.
        cust_dem (list): Customer demands.
        rc_cal_index (int): Route cost calculation index.
        min_max (dict, optional): Predefined min and max values for normalization.
    
    Returns:
        tuple: (facility_dict, big_m, cost_norm_factor)
    """
    facility_dict = {}
    normalized_coords, dist_fac_cust, big_m, cost_norm_factor = normalize_coord(cord_set1, cord_set2, rc_cal_index, min_max)
    
    for j in range(len(cord_set1)):
        norm_df = pd.DataFrame()
        norm_cust_dem = [cust_dem[i] / veh_cap[j] for i in range(len(cust_dem))]
        
        print(normalized_coords['normal'][j], '\n')
        norm_df['x'] = [normalized_coords['normal'][j][i][0] for i in range(len(cord_set2))]
        norm_df['y'] = [normalized_coords['normal'][j][i][1] for i in range(len(cord_set2))]
        norm_df['dem'] = norm_cust_dem
        facility_dict[j] = norm_df
    
    return facility_dict, big_m, cost_norm_factor
