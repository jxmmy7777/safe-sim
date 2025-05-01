import matplotlib.pyplot as plt
from shapely.geometry import LineString,Point
from scipy.spatial.distance import euclidean
import torch
import numpy as np
import json

def plot_centerlines_and_cars(centerlines, car_positions):
    """
    Plot the centerlines and car positions on a single figure.

    :param centerlines: Tensor of shape [num_agents, num_points_per_agent, 2]
    :param car_positions: Tensor of shape [num_agents, 2]
    """
    # Create a new figure
    plt.figure(figsize=(10, 10))

    # Plot each agent's centerline
    for agent_centerline in centerlines:
        plt.plot(agent_centerline[:, 0], agent_centerline[:, 1], '-o', markersize=2, linewidth=1)

    # Plot each agent's current position
    for agent_position in car_positions:
        plt.scatter(agent_position[0], agent_position[1], s=50, c='red', edgecolors='black', label='Car Position')

    # Configure plot attributes
    plt.title('Centerlines and Car Positions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Show the plot
    plt.savefig("centerlines_scene.png")

def vector_relationship(vector_i, vector_j):
    dot_products = torch.sum(vector_i * vector_j)
    magnitude_i = torch.norm(vector_i)
    magnitude_j = torch.norm(vector_j)

    cos_theta = dot_products / (magnitude_i * magnitude_j)
    return cos_theta

def determine_centerline_relationship(obs, merge_threshold=5.0, distance_threshold=10.0, parallel_threshold=5.0, interaction_threshold=50):
    positions = obs["curr_agent_state"].position
    centerlines = obs["extras"]["centerline_world_xy"]
    num_cars = len(positions)
    relationships = {}
    interactions = {} 

   
    for i in range(num_cars):
        for j in range(i + 1, num_cars):
            if not obs["extras"]["has_lane"][i] or not obs["extras"]["has_lane"][j]:
                relationships[(i, j)] = "no relationship"
                continue  
            
            #check if they have same lane ref_polyline_ids
            if obs["extras"]["ref_polyline_ids"][i] == obs["extras"]["ref_polyline_ids"][j]:
                relationships[(i, j)] = "same lane"
                continue
            pointwise_distances = torch.cdist(centerlines[i], centerlines[j])
            # 1. Nearby Lanes Check
            initial_distance = euclidean(positions[i], positions[j])
            min_distance = pointwise_distances.min()
            # Compute vectors for each centerline in a batch
            vector_i_batch = centerlines[i, 1:] - centerlines[i, :-1]
            vector_j_batch = centerlines[j, 1:] - centerlines[j, :-1]
            # If there's substantial overlap
            # if min_distance < merge_threshold:
            #     overlapping_indices = (pointwise_distances < merge_threshold).nonzero(as_tuple=True)
            #     if len(overlapping_indices[0]) > overlap_length_threshold:  # threshold for consecutive overlapping points
            #         relationships[(i, j)] = "nearby lanes"
            #         continue
            # Compute cosine of the angles between the vectors in a batch
            # dot_products = (vector_i_batch * vector_j_batch).sum(dim=1)
            # magnitudes_i = torch.norm(vector_i_batch, dim=1)
            # magnitudes_j = torch.norm(vector_j_batch, dim=1)
            # cosine_values = dot_products / (magnitudes_i * magnitudes_j)

            cosines = torch.nn.functional.cosine_similarity(vector_i_batch, vector_j_batch, dim=1)
            avg_cosine_similarity = cosines.mean()

            if avg_cosine_similarity > 0.7:  # threshold for parallel lanes, close to 1
                relationships[(i, j)] = "nearby lanes"
                continue

            if initial_distance < distance_threshold and min_distance < parallel_threshold:
                relationships[(i, j)] = "nearby lanes"

            line1 = LineString(centerlines[i].cpu().numpy())
            line2 = LineString(centerlines[j].cpu().numpy())

            # 2. Intersection Check
            if line1.intersects(line2):
                intersection_point = line1.intersection(line2)
                if intersection_point.geom_type == "MultiPoint":
                    coords = intersection_point.geoms[0].coords[0]
                elif intersection_point.geom_type == "MultiLineString":
                    coords = intersection_point.geoms[0].coords[0]  # take coords of the first line segment
                elif intersection_point.geom_type == "MultiPolygon":
                    coords = intersection_point.geoms[0].exterior.coords[0]  # take coords of the exterior of the first polygon
                elif intersection_point.geom_type == "Point":
                    coords = intersection_point.coords[0]
                # elif intersection_point.geom_type == "GeometryCollection":
                #     for geom in intersection_point.geoms:
                #         if geom.geom_type == "Point":
                #             intersection_point = geom.coords[0]
                #         break
                else:
                    relationships[(i, j)] = "no relationship"
                    continue

                idx_i = torch.argmin(torch.norm(centerlines[i] - torch.tensor(coords), dim=1))
                idx_j = torch.argmin(torch.norm(centerlines[j] - torch.tensor(coords), dim=1))

                if idx_i != len(centerlines[0])-1 and idx_j != len(centerlines[0])-1:
                    cos_theta = vector_relationship(centerlines[i, idx_i+1] - centerlines[i, idx_i],
                                                    centerlines[j, idx_j+1] - centerlines[j, idx_j])
                    if cos_theta < torch.cos(torch.tensor(torch.pi/3)):  # 60 degrees
                        relationships[(i, j)] = "intersection"
                        continue

            # 3. Merging Check

            i_below_threshold_indices = (pointwise_distances < merge_threshold).nonzero(as_tuple=True)[0]
            j_below_threshold_indices = (pointwise_distances < merge_threshold).nonzero(as_tuple=True)[1]

            if len(i_below_threshold_indices) > 0 and len(j_below_threshold_indices) > 0:
                i_first_below_threshold_idx = i_below_threshold_indices[0]
                j_first_below_threshold_idx = j_below_threshold_indices[0]

                # Consider all points after the identified points
                i_subsequent_points = centerlines[i, i_first_below_threshold_idx:]
                j_subsequent_points = centerlines[j, j_first_below_threshold_idx:]
                # Compute direction vectors for i and j
                direction_i = i_subsequent_points[1:] - i_subsequent_points[:-1]
                direction_j = j_subsequent_points[1:] - j_subsequent_points[:-1]

                # Crop the longer sequence of direction vectors to match the size of the shorter one
                min_length = min(len(direction_i), len(direction_j))
                direction_i = direction_i[:min_length]
                direction_j = direction_j[:min_length]

                cosines = torch.nn.functional.cosine_similarity(direction_i, direction_j, dim=1)
            
                is_similar_heading = torch.where(cosines > 0.9)[0]  # You can adjust this threshold

                # Crop the longer sequence to match the size of the shorter one
                i_subsequent_points = i_subsequent_points[:min_length+1]
                j_subsequent_points = j_subsequent_points[:min_length+1]

                subsequent_distances = torch.cdist(i_subsequent_points, j_subsequent_points)
                min_subsequent_distances = subsequent_distances.min(dim=1).values

                # Check if a significant portion remains close and doesn't diverge beyond a threshold
                close_count = (min_subsequent_distances < merge_threshold).sum().item()
                divergence_count = (min_subsequent_distances > 1.5 * merge_threshold).sum().item()  # 1.5 times merge_threshold as a divergence threshold

                if close_count > 0.8 * len(min_subsequent_distances) and divergence_count == 0 and len(is_similar_heading) > 0.8 * len(cosines):  # 80% remain close and no divergence
                    relationships[(i, j)] = "merging"
                    continue
                else:
                    relationships[(i, j)] = "nearby lanes"
                    continue
                
            relationships[(i, j)] = "no relationship"
            
        for key in relationships.keys():
            (i,j) = key
            #------------------- determine interaction--------------------------

            if relationships[(i, j)] in ["merging", "intersection"]:

                # Compute pairwise distances between all points of the two centerlines
                distances_i_to_j = torch.norm(centerlines[i][:,None]-centerlines[j][None], dim=2)
                distances_j_to_i = torch.norm(centerlines[j][:,None]-centerlines[i][None], dim=2)

                # calculate indices that are less than the threshold
                # Get the indices of distances that are less than the threshold
                if relationships[(i, j)] == "merging":
                    if torch.any(torch.abs(distances_i_to_j)<0.5):
                        thresholds = torch.tensor([0.2, 0.5])
                        for threshold in thresholds:
                            close_points_i_to_j = torch.where(distances_i_to_j < threshold)
                            close_points_j_to_i = torch.where(distances_j_to_i < threshold)

                            if close_points_i_to_j[0].size(0) > 0:
                                conflict_point_i_idx = close_points_i_to_j[0][0]
                                conflict_point_j_idx = close_points_i_to_j[1][0]
                                break
                            elif close_points_j_to_i[0].size(0) > 0:
                                conflict_point_i_idx = close_points_j_to_i[1][0]
                                conflict_point_j_idx = close_points_j_to_i[0][0]
                                break
                        
                    else:
                        close_points_i_to_j = torch.where(distances_i_to_j < merge_threshold)
                        close_points_j_to_i = torch.where(distances_j_to_i < merge_threshold)
                        
                        if close_points_i_to_j[0].size(0) > 0:
                            conflict_point_i_idx = close_points_i_to_j[0][0]
                            conflict_point_j_idx = close_points_i_to_j[1][0]
                        elif close_points_j_to_i[0].size(0) > 0:
                            conflict_point_i_idx = close_points_j_to_i[1][0]
                            conflict_point_j_idx = close_points_j_to_i[0][0]
                        else:
                            interactions[key] = ("no_interaction", 0, 0, (np.nan,np.nan))
                            continue
                    # plt
                else:
                    # # Find the indices of the closest points
                    min_idx_i_to_j = torch.argmin(distances_i_to_j.view(-1))
                    min_idx_j_to_i = torch.argmin(distances_j_to_i.view(-1))

                    # Convert the flattened index back to 2D indices
                    conflict_point_i_idx = min_idx_i_to_j // distances_i_to_j.size(1)
                    conflict_point_j_idx = min_idx_j_to_i // distances_j_to_i.size(1)

                    # # Compute the average of the two points to get the conflict point
                conflict_point = (centerlines[i][conflict_point_i_idx] + centerlines[j][conflict_point_j_idx]) / 2

                s1 = torch.norm(conflict_point - positions[i])
                s2 = torch.norm(conflict_point - positions[j])

                car_i_position_idx = torch.argmin(torch.norm(centerlines[i] - positions[i], dim=1))
                car_j_position_idx = torch.argmin(torch.norm(centerlines[j] - positions[j], dim=1)) 

                if car_i_position_idx > conflict_point_i_idx or car_j_position_idx > conflict_point_j_idx:
                    interactions[key] = ("already_passed_conflict", s1.item(), s2.item(),conflict_point)
                elif torch.abs(s1 - s2) > interaction_threshold:
                    interactions[key] = ("no_interaction",  s1.item(), s2.item(),conflict_point)
                elif torch.abs(s1-s2) <= 10:
                    interactions[key] = ("equal",  s1.item(), s2.item(),conflict_point)
                elif s1 > s2 and (s1 - s2) < interaction_threshold:
                    interactions[key] = ("car1_behind_car2",  s1.item(), s2.item(), conflict_point)
                else:
                    interactions[key] = ("car2_behind_car1",  s1.item(), s2.item(), conflict_point)
            
            elif relationships[(i, j)] == "nearby lanes":
                
                curr_state = obs["curr_agent_state"]
                curr_yaw = curr_state.heading[...,0]
                curr_pos = curr_state.position
                
                ego_idx = i
                agent_indices = j
                # egos_pos, egos_yaw, agents_pos, agents_yaws = ... (Get these values as necessary)
                rel_pos, rel_orient, dists = get_relative_positions_and_orientations(
                    curr_pos[ego_idx], 
                    curr_yaw[ego_idx],
                    curr_pos[agent_indices], 
                    curr_yaw[agent_indices])
                
                
                pos_cats, heading_cats, dist_cats = classify_agents(rel_pos, dists, rel_orient, distance_threshold=10.0)
                interactions[key] = (pos_cats.item(), dist_cats.item(), dists.item(),0)
            else:
                interactions[key] = ("no_interaction", 0, 0,0)
    return relationships,interactions

def transform_to_ego_perspective(rel_positions, ego_yaw):
    # Create rotation matrix
    cos_yaw = torch.cos(-ego_yaw)
    sin_yaw = torch.sin(-ego_yaw)
    
    rotation_matrix = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=-1).reshape( 2, 2)
    
    # Use batch matrix multiplication for rotation
    transformed_positions =rel_positions @ rotation_matrix

    return transformed_positions

def get_relative_positions_and_orientations(ego_pos, ego_yaw, agent_positions, agent_yaws):
    rel_positions = agent_positions - ego_pos
    rel_positions_ego_perspective = transform_to_ego_perspective(rel_positions, ego_yaw)

    # Using agent_yaws to compute the relative orientations
    relative_orientations = (agent_yaws - ego_yaw + 2 * torch.pi) % (2 * torch.pi)
    distances = torch.norm(rel_positions, dim=-1)
    
    return rel_positions_ego_perspective, relative_orientations, distances

def classify_agents(rel_positions, distances, relative_orientations, distance_threshold=10.0):  # default threshold as example
    # Determine Relative Positions using transformed rel_positions
    # Extract x and y from the relative positions
    rel_x = rel_positions[..., 0]
    rel_y = rel_positions[..., 1]

    # Define boolean masks for each direction
    front = rel_x > 0
    back = rel_x < 0
    left = rel_y < 0
    right = rel_y > 0

    # Determine Relative Heading Angles
    approaching = relative_orientations > torch.pi / 2
    moving_away = relative_orientations <= torch.pi / 2
    same_direction = (relative_orientations < torch.pi / 4) | (relative_orientations > 3 * torch.pi / 4)
    opposite_direction = (relative_orientations >= torch.pi / 4) & (relative_orientations <= 3 * torch.pi / 4)

    # Distance-based classification
    near = distances < distance_threshold
    far = ~near

    # Combine categories and return
    pos_categories = torch.zeros_like(distances, dtype=torch.long)
    heading_categories = torch.zeros_like(distances, dtype=torch.long)
    distance_categories = torch.zeros_like(distances, dtype=torch.long)

    # Assigning values for position categories
    # Initial categories with zeros (unknown)
    pos_categories[front & left]  = 0
    pos_categories[front & right] = 1
    pos_categories[back & left]   = 2
    pos_categories[back & right]  = 3

    # Assigning values for heading categories
    heading_categories[approaching] = 1
    heading_categories[moving_away] = 2
    heading_categories[same_direction] = 3
    heading_categories[opposite_direction] = 4

    # Assign values for distance categories
    distance_categories[near] = 0
    distance_categories[far] = 1

    return pos_categories, heading_categories, distance_categories
def plot_relationships_separately(positions, centerlines, relationships, interactions, scene_name = ""):
    # Plot all centerlines in light gray first as the background
    
    for (i, j), relationship in relationships.items():
        #only plot if relationship is not no relationship
        if relationship in ["no relationship", "nearby lanes", "same lane"]:
            continue
        plt.figure(figsize=(10, 10))
        
        # Plot centerlines with color transition
        plt.scatter(centerlines[i,:,0], centerlines[i,:,1], c=np.arange(len(centerlines[i])), cmap='coolwarm', label=f"Car {i}-{relationship}")
        plt.scatter(centerlines[j,:,0], centerlines[j,:,1], c=np.arange(len(centerlines[j])), cmap='coolwarm', label=f"Car {j}-{relationship}")
        
        # Plot car positions
        plt.scatter(positions[i,0], positions[i,1], color='black', marker='x', s=100, label=f"Position Car {i}")
        plt.scatter(positions[j,0], positions[j,1], color='red', marker='x', s=100, label=f"Position Car {j}")
        
        

        # Add colorbar to indicate progression of centerline
        plt.colorbar(label='Progression of Centerline')

        # Remove duplicate labels in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        plt.xlabel('X')
        plt.ylabel('Y')

        interaction_type = interactions.get((i, j), interactions.get((j, i), None))
        if interaction_type:
            interaction_label,s1,s2,conflict_pt = interaction_type
            plt.plot(conflict_pt[0],conflict_pt[1],marker='o',color='black',markersize=10)
            plt.title(f'Centerline Relationships Visualization - {relationship} - Interaction: {interaction_label}(s1: {s1:.2f}, s2: {s2:.2f})')
        else:
            plt.title(f'Centerline Relationships Visualization - {relationship}')

        plt.grid(True)
        plt.savefig(f"scene_classify/{scene_name[i]}_{relationship}_{i}{j}.png")
        plt.close()
        
def load_predefined_scene_init_from_json(relationship_list, interactions_list, train= False):
      
    merged_dict = {}
    for relationship in relationship_list:
        for interaction in interactions_list:
            if train:
                filename = f"scene_agent_relation/TRAIN_{relationship}__{interaction}.json"
            else:
                filename = f"scene_agent_relation/{relationship}__{interaction}.json"
            try:
                with open(filename, 'r') as file:
                    data = json.load(file)
                    
                    # Merge dictionaries
                    for key, value in data.items():
                        if key in merged_dict:
                            merged_dict[key]["indices"].extend(value["indices"])
                        else:
                            merged_dict[key] = value
                            
            except FileNotFoundError:
                print(f"No JSON file found for relationship: {relationship} and interaction: {interaction}")
    return merged_dict
def filter_scenes(data, relationship, interactions):
    filtered_scenes = {}
    
    for scene, scene_dict in data.items():
        for (i,j) in scene_dict["relationships"].keys():
            # Check if the scene_interaction matches any interaction in the list
            if scene_dict["relationships"][(i,j)] == relationship and scene_dict["interactions"][(i,j)][0] == interactions:
                # The logic here assumes 'i' is always the ego vehicle
                # and 'j' is the controlled vehicle.
                # Adjust if this assumption is not correct.
                if scene not in filtered_scenes:
                    filtered_scenes[scene] = {}
                    filtered_scenes[scene]["indices"] = [{"ego_idx": i, "ctrl_indices": [j]}]
                    filtered_scenes[scene]["ref_polyline_ids"] = scene_dict["ref_polyline_ids"]
                else:
                    filtered_scenes[scene]["indices"] += [{"ego_idx": i, "ctrl_indices": [j]}]
                    
    
    return filtered_scenes

def save_filtered_data_as_json(filtered_data, relationship, interaction):
    # Constructing the filename
    filename = f"scene_agent_relation/{relationship}__{interaction}.json"
    
    # Saving the data
    with open(filename, 'w') as file:
        json.dump(filtered_data, file, indent=4)
        
    print(f"Data saved to {filename}!")

def find_conflict_point(centerline1, centerline2, relationship="intersection", merge_threshold=0.5):
    distances_i_to_j = torch.norm(centerline1[:, None] - centerline2[None], dim=2)
    distances_j_to_i = torch.norm(centerline2[:, None] - centerline1[None], dim=2)
    conflict_point_i_idx = None
    conflict_point_j_idx = None

    if relationship == "merging":
        if torch.any(torch.abs(distances_i_to_j) < 0.5):
            thresholds = torch.tensor([0.2, 0.5])
            for threshold in thresholds:
                close_points_i_to_j = torch.where(distances_i_to_j < threshold)
                close_points_j_to_i = torch.where(distances_j_to_i < threshold)

                if close_points_i_to_j[0].size(0) > 0:
                    conflict_point_i_idx = close_points_i_to_j[0][0]
                    conflict_point_j_idx = close_points_i_to_j[1][0]
                    break
                elif close_points_j_to_i[0].size(0) > 0:
                    conflict_point_i_idx = close_points_j_to_i[1][0]
                    conflict_point_j_idx = close_points_j_to_i[0][0]
                    break
            else:
                close_points_i_to_j = torch.where(distances_i_to_j < merge_threshold)
                close_points_j_to_i = torch.where(distances_j_to_i < merge_threshold)

                if close_points_i_to_j[0].size(0) > 0:
                    conflict_point_i_idx = close_points_i_to_j[0][0]
                    conflict_point_j_idx = close_points_i_to_j[1][0]
                elif close_points_j_to_i[0].size(0) > 0:
                    conflict_point_i_idx = close_points_j_to_i[1][0]
                    conflict_point_j_idx = close_points_j_to_i[0][0]

    elif relationship == "intersection":
        min_idx_i_to_j = torch.argmin(distances_i_to_j.view(-1))
        min_idx_j_to_i = torch.argmin(distances_j_to_i.view(-1))

        conflict_point_i_idx = min_idx_i_to_j // distances_i_to_j.size(1)
        conflict_point_j_idx = min_idx_j_to_i // distances_j_to_i.size(1)

    conflict_point = (centerline1[conflict_point_i_idx] + centerline2[conflict_point_j_idx]) / 2

    return conflict_point, conflict_point_i_idx, conflict_point_j_idx
