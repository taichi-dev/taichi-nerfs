import taichi as ti
from taichi.math import vec3

@ti.kernel
def get_rays(pose: ti.types.ndarray(),
             directions: ti.types.ndarray(),
             rays_o: ti.types.ndarray(),
             rays_d: ti.types.ndarray()):
    #print(directions.shape)
    for i in ti.ndrange(directions.shape[0]):
    #for i in range(10):
        c2w = pose[None]
        mat_result = directions[i] @ c2w[:, :3].transpose()
        ray_d = vec3(mat_result[0, 0], mat_result[0, 1], mat_result[0, 2])
        ray_o = c2w[:, 3]

        rays_o[i] = ray_o
        rays_d[i] = ray_d
