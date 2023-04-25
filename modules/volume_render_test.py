import taichi as ti
from taichi.math import vec3

@ti.kernel
def composite_test(
           sigmas: ti.types.ndarray(ndim=1),
             rgbs: ti.types.ndarray(ndim=2),
           deltas: ti.types.ndarray(ndim=1),
               ts: ti.types.ndarray(ndim=1),
        pack_info: ti.types.ndarray(ndim=2),
    alive_indices: ti.types.ndarray(ndim=1),
      T_threshold: float,
          opacity: ti.types.ndarray(ndim=1),
            depth: ti.types.ndarray(ndim=1),
              rgb: ti.types.ndarray(ndim=2)
):
    
    ti.loop_config(block_dim=256)
    for n in alive_indices:
        start_idx = pack_info[n, 0]
        steps = pack_info[n, 1]
        ray_idx = alive_indices[n]
        if steps == 0:
            alive_indices[n] = -1
        else:
            T = 1 - opacity[ray_idx]

            rgb_temp = vec3(0.0)
            depth_temp = 0.0
            opacity_temp = 0.0

            for s in range(steps):
                s_n = start_idx + s
                delta = deltas[s_n]
                a = 1.0 - ti.exp(-sigmas[s_n]*delta)

                w = a * T
                tmid = ts[s_n]
                rgbs_vec3 = vec3(
                    rgbs[s_n, 0], rgbs[s_n, 1], rgbs[s_n, 2]
                )
                rgb_temp += w * rgbs_vec3
                depth_temp += w * tmid
                opacity_temp += w
                T *= 1.0 - a

                if T <= T_threshold:
                    alive_indices[n] = -1
                    break

            rgb[ray_idx, 0] += rgb_temp[0]
            rgb[ray_idx, 1] += rgb_temp[1]
            rgb[ray_idx, 2] += rgb_temp[2]
            depth[ray_idx] += depth_temp
            opacity[ray_idx] += opacity_temp