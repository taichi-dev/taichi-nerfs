import torch
import taichi as ti
from taichi.math import vec3, vec2
from .utils import NEAR_DISTANCE

mat3x4 = ti.types.matrix(3, 4, ti.f32)
mat1x3 = ti.types.matrix(1, 3, ti.f32)


@ti.func
def __get_rays(c2w, direction):

    mat_result = direction @ c2w[:, :3].transpose()
    ray_d = vec3(mat_result[0, 0], mat_result[0, 1], mat_result[0, 2])
    ray_o = c2w[:, 3]

    return ray_o, ray_d

@ti.kernel
def get_rays_test_kernel(
    pose: ti.types.ndarray(dtype=mat3x4, ndim=0),
    directions: ti.types.ndarray(dtype=mat1x3, ndim=1),
    rays_o: ti.types.ndarray(dtype=vec3, ndim=1),
    rays_d: ti.types.ndarray(dtype=vec3, ndim=1),
):
    for i in ti.ndrange(directions.shape[0]):
        c2w = pose[None]
        direction = directions[i]

        ray_o, ray_d = __get_rays(c2w, direction)

        rays_o[i] = ray_o
        rays_d[i] = ray_d

@ti.kernel
def get_rays_train_kernel(
    pose: ti.types.ndarray(dtype=mat3x4, ndim=1),
    directions: ti.types.ndarray(dtype=mat1x3, ndim=1),
    rays_o: ti.types.ndarray(dtype=vec3, ndim=1),
    rays_d: ti.types.ndarray(dtype=vec3, ndim=1),
):
    for i in ti.ndrange(directions.shape[0]):
        c2w = pose[i]
        direction = directions[i]
        
        ray_o, ray_d = __get_rays(c2w, direction)

        rays_o[i] = ray_o
        rays_d[i] = ray_d

def get_rays(directions, pose):
    rays_o = torch.empty_like(directions)
    rays_d = torch.empty_like(directions)

    if len(pose.shape) == 3:
        get_rays_train_kernel(
            pose,
            directions.unsqueeze(1),
            rays_o, 
            rays_d,  
        )
    else:
        get_rays_test_kernel(
            pose,
            directions.unsqueeze(1),
            rays_o, 
            rays_d,  
        )

    return rays_o, rays_d

@ti.kernel
def ray_aabb_intersect(
    hits_t: ti.types.ndarray(dtype=vec2, ndim=1),
    rays_o: ti.types.ndarray(dtype=vec3, ndim=1),
    rays_d: ti.types.ndarray(dtype=vec3, ndim=1),
    scale: ti.f32
):
    xyz_max = ti.Vector([scale, scale, scale])
    xyz_min = -xyz_max
    half_size = (xyz_max - xyz_min) / 2
    center = ti.Vector([0.0, 0.0, 0.0])

    ti.loop_config(block_dim=512)
    for r in ti.ndrange(hits_t.shape[0]):
        ray_o = rays_o[r]
        ray_d = rays_d[r]
        inv_d = 1.0 / ray_d

        t_min = (center - half_size - ray_o) * inv_d
        t_max = (center + half_size - ray_o) * inv_d

        _t1 = ti.min(t_min, t_max)
        _t2 = ti.max(t_min, t_max)
        t1 = _t1.max()
        t2 = _t2.min()

        if t2 > 0.0:
            hits_t[r] = ti.Vector([ti.max(t1, NEAR_DISTANCE), t2])
        else:
            hits_t[r] = ti.Vector([-1.0, -1.0])


def ray_aabb_intersection(rays_o, rays_d, scale):
    
    hits_t = torch.empty(
        rays_o.size(0), 2,
        device=rays_o.device, 
        dtype=rays_o.dtype
    )

    ray_aabb_intersect(
        hits_t, 
        rays_o, 
        rays_d,
        scale
    )

    return hits_t
