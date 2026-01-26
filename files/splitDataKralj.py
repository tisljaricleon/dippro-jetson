import pandas as pd
import os
from geopy.distance import geodesic

def is_within_radius(lat1, lon1, lat2, lon2, radius_km):
    return geodesic((lat1, lon1), (lat2, lon2)).km <= radius_km

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    locations_data = pd.read_csv(os.path.join(dataset_path, 'locations-raw.csv'))

    cloudlet_nodes_list = [[] for _ in range(len(cloudlets))]

    for idx, sensor in locations_data.iterrows():
        sensor_loc = (sensor['Latitude'], sensor['Longitude'])
        closest_cloudlet = None
        min_distance = float('inf')

        for name, loc in cloudlets.items():
            if is_within_radius(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'], radius_km):
                distance = calculate_distance(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'])
                if distance < min_distance:
                    min_distance = distance
                    closest_cloudlet = loc['id']

        if closest_cloudlet is not None:
            cloudlet_nodes_list[closest_cloudlet].append(idx)

    # print(f"cloudlet_nodes_list: {cloudlet_nodes_list}")

    unassigned_sensors = [
        idx for idx, sensor in locations_data.iterrows()
        if all(idx not in nodes for nodes in cloudlet_nodes_list)
    ]
    if unassigned_sensors:
        print(f"WARNING: Following sensors were not assigned to any cloudlet due to range limitations: {unassigned_sensors}!")

    return cloudlet_nodes_list

def get_cloudlet_location_info_from_json(experiment_name, cloudlet_info_json):
    cloudlet_info = cloudlet_info_json.get(experiment_name) 
    if cloudlet_info: 
        cloudlets = cloudlet_info["cloudlets"] 
        radius_km = cloudlet_info["radius_km"] 
        return cloudlets, radius_km
    else: 
        raise ValueError(f"Experiment '{experiment_name}' not found in the json file.")