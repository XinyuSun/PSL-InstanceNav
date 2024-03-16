import numpy as np
from typing import Dict, List, Optional, Tuple

from habitat.utils.visualizations.utils import tile_images, draw_collision
from habitat.utils.visualizations import maps


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        if "cam_visual" in sensor_name:
            cam_visual = observation[sensor_name]
            if not isinstance(cam_visual, np.ndarray):
                cam_visual = cam_visual.cpu().numpy()

            render_obs_images.append(cam_visual)
        elif "depth" in sensor_name:
            depth_map = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            render_obs_images.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        render_obs_images.append(rgb)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame