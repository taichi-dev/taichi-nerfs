import taichi as ti
import torch
from taichi.math import vec3

from .utils import NEAR_DISTANCE


@ti.kernel
def ray_aabb_intersect(
    hits_t: ti.types.ndarray(),
    rays_o: ti.types.ndarray(),
    rays_d: ti.types.ndarray(),
    centers: ti.types.ndarray(),
    half_sizes: ti.types.ndarray(),
):
    ti.loop_config(block_dim=512)
    for r in ti.ndrange(hits_t.shape[0]):
        ray_o = vec3([rays_o[r, 0], rays_o[r, 1], rays_o[r, 2]])
        ray_d = vec3([rays_d[r, 0], rays_d[r, 1], rays_d[r, 2]])
        inv_d = 1.0 / ray_d

        center = vec3([centers[0, 0], centers[0, 1], centers[0, 2]])
        half_size = vec3(
            [half_sizes[0, 0], half_sizes[0, 1], half_sizes[0, 1]])

        t_min = (center - half_size - ray_o) * inv_d
        t_max = (center + half_size - ray_o) * inv_d

        _t1 = ti.min(t_min, t_max)
        _t2 = ti.max(t_min, t_max)
        t1 = _t1.max()
        t2 = _t2.min()

        if t2 > 0.0:
            hits_t[r, 0] = ti.max(t1, NEAR_DISTANCE)
            hits_t[r, 1] = t2


def ray_aabb_intersection(rays_o, rays_d, center, half_size):
    
    hits_t = torch.full(
        (rays_o.size(0), 2),
        -1, 
        device=rays_o.device, 
        dtype=rays_o.dtype
    )

    ray_aabb_intersect(
        hits_t, 
        rays_o, 
        rays_d, 
        center,
        half_size
    )

    return hits_t