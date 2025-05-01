import numpy as np
from PIL import Image, ImageDraw


from trajdata.maps.raster_map import RasterizedMap

from tbsim.utils.tensor_utils import map_ndarray
from tbsim.utils.geometry_utils import get_box_world_coords_np,transform_points
import tbsim.utils.tensor_utils as TensorUtils


COLORS = {
    "agent_contour": "#247BA0",
    "agent_fill": "#56B1D8",
  
    "ctrl_agent_contour": "#911A12",  # Dark Green for the contour
    "ctrl_agent_fill": "#FE5F55",     # Green for the fill
    "ego_contour": "#FCAE1E",
    "ego_fill": "#FCAE1E",
}
# COLORS = {
#     "agent_contour": "#247BA0",
#     "agent_fill": "#56B1D8",
#     "ego_contour": "#911A12",
#     "ego_fill": "#FE5F55",
#     "ctrl_agent_contour": "#006400",  # Dark Green for the contour
#     "ctrl_agent_fill": "#008000"     # Green for the fill
# }


def agent_to_raster_np(pt_tensor, trans_mat):
    pos_raster = transform_points(pt_tensor[None], trans_mat)[0]
    return pos_raster


def draw_actions(
        state_image,
        trans_mat,
        pred_action=None,
        pred_plan=None,
        pred_plan_info=None,
        ego_action_samples=None,
        plan_samples=None,
        action_marker_size=3,
        plan_marker_size=8,
):
    im = Image.fromarray((state_image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    if pred_action is not None:
        raster_traj = agent_to_raster_np(
            pred_action["positions"].reshape(-1, 2), trans_mat)
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#FE5F55", outline="#911A12")
    if ego_action_samples is not None:
        raster_traj = agent_to_raster_np(
            ego_action_samples["positions"].reshape(-1, 2), trans_mat)
        # for point in raster_traj:
        #     circle = np.hstack([point - action_marker_size, point + action_marker_size])
        #     draw.ellipse(circletolist(), #fill="#808080",
        #                  outline="#911A12")
        # Define the colors for each batch
        colors = ["#0000FF"] * 32 + ["#911A12"] * (19 * 32)
        
        # Reshape the trajectory and colors for easier looping
        reshaped_raster_traj = raster_traj.reshape(-1, 2)
        circles = np.hstack([reshaped_raster_traj - action_marker_size, reshaped_raster_traj + action_marker_size])
        
        # Draw the ellipses
        for circle, color in zip(circles, colors):
            draw.ellipse(circle.tolist(), outline=color)

    if pred_plan is not None:
        pos_raster = agent_to_raster_np(
            pred_plan["positions"][:, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    if plan_samples is not None:
        pos_raster = agent_to_raster_np(
            plan_samples["positions"][0, :, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    im = np.asarray(im)
    # visualize plan heat map
    if pred_plan_info is not None and "location_map" in pred_plan_info:
        import matplotlib.pyplot as plt

        cm = plt.get_cmap("jet")
        heatmap = pred_plan_info["location_map"][0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        heatmap = cm(heatmap)

        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(size=(im.shape[1], im.shape[0]))
        heatmap = np.asarray(heatmap)[..., :3]
        padding = np.ones((im.shape[0], 200, 3), dtype=np.uint8) * 255

        composite = heatmap.astype(np.float32) * \
            0.3 + im.astype(np.float32) * 0.7
        composite = composite.astype(np.uint8)
        im = np.concatenate((im, padding, heatmap, padding, composite), axis=1)

    return im


def draw_agent_boxes(image, pos, yaw, extent, raster_from_agent, outline_color, fill_color, index = None):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2)).astype(int)

    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    for i,b in enumerate(boxes_raster):
        im_draw.polygon(xy=b.reshape(-1).tolist(),
                        outline=outline_color, fill=fill_color)
        # Annotate the box with its index
        box_center = ((b[:, 0].max() + b[:, 0].min()) / 2, 
                      (b[:, 1].max() + b[:, 1].min()) / 2)
        if isinstance(index,np.ndarray):
            im_draw.text(box_center, str(index[i]), fill="black")
    #label the index
    if isinstance(index,np.int64):
        im_draw.text(box_center, str(index), fill="black")

    im = np.asarray(im).astype(np.float32) / 255.
    return im

def draw_references(image, batch_idx, path, raster_from_agent, agent_from_world, world_from_agent, outline_color, right_path_color='blue', left_path_color='green', target_idx = None):
    # Expand dimensions of agent_from_world[batch_idx]
    agent_from_world_expand = np.expand_dims(agent_from_world[batch_idx], axis=0)
    raster_from_agent_expand = np.expand_dims(raster_from_agent[batch_idx], axis=0)
    
    # Use broadcasting in the np.einsum operation
    ego_from_all_agent = np.einsum('bij,bjk->bik', agent_from_world_expand, world_from_agent)
    raster_from_all_agent = np.einsum('bij,bjk->bik', raster_from_agent_expand, ego_from_all_agent)
    
    if target_idx is not None:
        path_raster = transform_points(path[target_idx], raster_from_all_agent[target_idx])

    else: 
        path_raster = transform_points(path, raster_from_all_agent)
    
    path_raster = path_raster.reshape((-1, 150, 2)).astype(int)

    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    
    for p in path_raster:
        im_draw.point(xy=p.reshape(-1).tolist(), fill=outline_color)
    # Draw right_path and left_path as continuous lines
    # for p in path_raster:
    #     im_draw.point(xy=p.reshape(-1).tolist(), fill=right_path_color)
    # for p in path_raster:
    #     im_draw.point(xy=p.reshape(-1).tolist(), fill=left_path_color)


    im = np.asarray(im).astype(np.float32) / 255.
    return im

def draw_poss_refs(image, batch_idx, ctrl_idx, all_pos_refs, num_refs, raster_from_agent, agent_from_world, world_from_agent, outline_color):
    # Expand dimensions of agent_from_world[batch_idx]
    if num_refs == 0:
        return image
    agent_from_world_expand = np.expand_dims(agent_from_world[batch_idx], axis=0)
    raster_from_agent_expand = np.expand_dims(raster_from_agent[batch_idx], axis=0)
    
    # Use broadcasting in the np.einsum operation
    ego_from_all_agent = np.einsum('bij,bjk->bik', agent_from_world_expand, world_from_agent)
    raster_from_all_agent = np.einsum('bij,bjk->bik', raster_from_agent_expand, ego_from_all_agent)
    
    path_raster = transform_points(all_pos_refs[:num_refs], raster_from_all_agent[[ctrl_idx]])
   
    path_raster = path_raster.reshape((-1, 150, 2)).astype(int)
  
    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    
    for p in path_raster:
        im_draw.point(xy=p.reshape(-1).tolist(), fill=outline_color)
    # Draw right_path and left_path as continuous lines
  

    im = np.asarray(im).astype(np.float32) / 255.
    return im


def render_state_trajdata(
        batch: dict,
        batch_idx: int,
        action,
        i: int = 0,
        ctrl_idx= None,
) -> np.ndarray:
    pos = batch["history_positions"][batch_idx, -1]
    yaw = batch["history_yaws"][batch_idx, -1]
    extent = batch["extent"][batch_idx, :2]
    
    scene_index = batch["scene_index"][batch_idx]
    agent_scene_index= scene_index == batch["scene_index"]
    first_true_index = np.argmax(agent_scene_index)

  
    image = RasterizedMap.to_img(
        TensorUtils.to_tensor(batch["maps"][batch_idx]),
        [[0], [1], [2]],
    )

    image = draw_agent_boxes(
        image,
        pos=pos[None, :],
        yaw=yaw[None, :],
        extent=extent[None, :],
        raster_from_agent=batch["raster_from_agent"][batch_idx],
        outline_color=COLORS["ego_contour"],
        fill_color=COLORS["ego_fill"],
        index = batch_idx-first_true_index
    )
   

    if "centerline_xy" in batch["extras"]:
        # image = draw_references(
        #     image,
        #     batch_idx= batch_idx,
        #     path = batch["extras"]["centerline_xy"],
        #     right_path = batch["extras"]["right_centerline_xy"],
        #     left_path = batch["extras"]["left_centerline_xy"],
        #     raster_from_agent = batch["raster_from_agent"],
        #     world_from_agent = batch["world_from_agent"],
        #     agent_from_world =  batch["agent_from_world"],
        #     outline_color = "black"
        # )
        # just draw ctrl and target idx
        
        if ctrl_idx is not None:
            target_idx = [batch_idx, ctrl_idx]
            image = draw_references(
                image,
                batch_idx= batch_idx,
                path = batch["extras"]["centerline_xy"],
                raster_from_agent = batch["raster_from_agent"],
                world_from_agent = batch["world_from_agent"],
                agent_from_world =  batch["agent_from_world"],
                outline_color = "black",
                target_idx = target_idx
            )
            
            image = draw_poss_refs(
                image, 
                batch_idx= batch_idx,
                ctrl_idx = ctrl_idx,
                all_pos_refs =  batch["extras"]["all_poss_refs"][ctrl_idx],
                num_refs = int(batch["extras"]["num_poss_refs"][ctrl_idx]),
                raster_from_agent = batch["raster_from_agent"],
                world_from_agent = batch["world_from_agent"],
                agent_from_world =  batch["agent_from_world"],
                outline_color = "green"
                
            )
        
        
        
        


    agent_scene_index[batch_idx] = 0  # don't plot ego
    neigh_pos = batch["centroid"][agent_scene_index]
    neigh_yaw = batch["yaw"][agent_scene_index]
    neigh_extent = batch["extent"][agent_scene_index, :2]

    #turn True False agent_scene_index to indices
    agent_scene_inds = np.where(agent_scene_index)[0]
    if neigh_pos.shape[0] > 0:
        image = draw_agent_boxes(
            image,
            pos=neigh_pos,
            yaw=neigh_yaw[:, None],
            extent=neigh_extent,
            raster_from_agent=batch["raster_from_world"][batch_idx],
            outline_color=COLORS["agent_contour"],
            fill_color=COLORS["agent_fill"],
            index = agent_scene_inds-first_true_index
        )
    if ctrl_idx is not None:
        ctrl_pos = batch["centroid"][ctrl_idx]
        ctrl_yaw = batch["yaw"][ctrl_idx:ctrl_idx+1]
        ctrl_extent = batch["extent"][ctrl_idx, :2]

        image = draw_agent_boxes(
            image,
            pos=ctrl_pos[None, :],
            yaw=ctrl_yaw[None, :],
            extent=ctrl_extent[None, :],
            raster_from_agent=batch["raster_from_world"][batch_idx],
            outline_color=COLORS["ctrl_agent_contour"],
            fill_color=COLORS["ctrl_agent_fill"],
            index = ctrl_idx-first_true_index
        )
    plan_info = None
    plan_samples = None
    action_samples = None
    if action.ego is not None:
        action_samples = {'positions':action.ego.positions[i],'yaws':action.ego.yaws[i],}
        vis_action = action_samples
    else:
        if "plan_info" in action.agents_info:
            plan_info = TensorUtils.map_ndarray(action.agents_info["plan_info"], lambda x: x[[batch_idx]])
        if "plan_samples" in action.agents_info:
            plan_samples = TensorUtils.map_ndarray(action.agents_info["plan_samples"], lambda x: x[[batch_idx]])
        if "action_samples" in action.agents_info:
            action_samples = TensorUtils.map_ndarray(action.agents_info["action_samples"], lambda x: x[[batch_idx]])

        vis_action = TensorUtils.map_ndarray(action.agents.to_dict(), lambda x: x[batch_idx])
    
    image = draw_actions(
        image,
        trans_mat=batch["raster_from_agent"][batch_idx],
        pred_action=vis_action,
        pred_plan_info=plan_info,
        ego_action_samples=action_samples,
        plan_samples=plan_samples,
        action_marker_size=2,
        plan_marker_size=3
    )

    ## Draw reference

    return image

