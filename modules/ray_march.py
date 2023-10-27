import taichi as ti
import torch
from taichi.math import vec3

from .utils import __morton3D, calc_dt, mip_from_dt, mip_from_pos, torch_type


@ti.kernel
def raymarching_train_kernel(
    rays_o: ti.types.ndarray(),
    rays_d: ti.types.ndarray(),
    hits_t: ti.types.ndarray(),
    density_bitfield: ti.types.ndarray(),
    noise: ti.types.ndarray(),
    counter: ti.types.ndarray(),
    rays_a: ti.types.ndarray(),
    xyzs: ti.types.ndarray(),
    dirs: ti.types.ndarray(),
    deltas: ti.types.ndarray(),
    ts: ti.types.ndarray(),
    cascades: int,
    grid_size: int,
    scale: float,
    exp_step_factor: float,
    max_samples: float,
):
    ti.loop_config(block_dim=128)
    for r in noise:
        ray_o = vec3(rays_o[r, 0], rays_o[r, 1], rays_o[r, 2])
        ray_d = vec3(rays_d[r, 0], rays_d[r, 1], rays_d[r, 2])
        d_inv = 1.0 / ray_d

        t1, t2 = hits_t[r, 0], hits_t[r, 1]

        grid_size3 = grid_size**3
        grid_size_inv = 1.0 / grid_size

        if t1 >= 0:
            dt = calc_dt(t1, exp_step_factor, grid_size, scale)
            t1 += dt * noise[r]

        t = t1
        N_samples = 0

        while (0 <= t) & (t < t2) & (N_samples < max_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            mip = ti.max(mip_from_pos(xyz, cascades),
                         mip_from_dt(dt, grid_size, cascades))

            mip_bound = ti.min(ti.pow(2., mip - 1), scale)
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(
                0.5 * (xyz * mip_bound_inv + 1) * grid_size,
                xmin=0.0,
                xmax=grid_size - 1.0
            )

            idx = mip * grid_size3 + __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx // 8)] & (1 << ti.u32(idx % 8))

            if occ:
                t += dt
                N_samples += 1
            else:
                # t += dt
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)

        start_idx = ti.atomic_add(counter[0], N_samples)
        ray_count = ti.atomic_add(counter[1], 1)

        rays_a[ray_count, 0] = r
        rays_a[ray_count, 1] = start_idx
        rays_a[ray_count, 2] = N_samples

        t = t1
        samples = 0

        while (t < t2) & (samples < N_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            mip = ti.max(mip_from_pos(xyz, cascades),
                         mip_from_dt(dt, grid_size, cascades))

            mip_bound = ti.min(ti.pow(2., mip - 1), scale)
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(
                0.5 * (xyz * mip_bound_inv + 1) * grid_size,
                xmin=0.0,
                xmax=grid_size - 1.0
            )

            idx = mip * grid_size3 + __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx // 8)] & (1 << ti.u32(idx % 8))

            if occ:
                s = start_idx + samples
                xyzs[s, 0] = xyz[0]
                xyzs[s, 1] = xyz[1]
                xyzs[s, 2] = xyz[2]
                dirs[s, 0] = ray_d[0]
                dirs[s, 1] = ray_d[1]
                dirs[s, 2] = ray_d[2]
                ts[s] = t
                deltas[s] = dt
                t += dt
                samples += 1
            else:
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)


def raymarching_train(
        rays_o,
        rays_d,
        hits_t,
        density_bitfield,
        cascades,
        scale,
        exp_step_factor,
        grid_size,
        max_samples
    ):
    # noise to perturb the first sample of each ray
    noise = torch.rand_like(rays_o[:, 0])
    counter = torch.zeros(
        2,
        device=rays_o.device,
        dtype=torch.int32
    )
    rays_a = torch.empty(
        rays_o.shape[0], 3,
        device=rays_o.device,
        dtype=torch.int32,
    )
    xyzs = torch.empty(
        rays_o.shape[0] * max_samples, 3,
        device=rays_o.device,
        dtype=torch_type,
    )
    dirs = torch.empty(
        rays_o.shape[0] * max_samples, 3,
        device=rays_o.device,
        dtype=torch_type,
    )
    deltas = torch.empty(
        rays_o.shape[0] * max_samples,
        device=rays_o.device,
        dtype=torch_type,
    )
    ts = torch.empty(
        rays_o.shape[0] * max_samples,
        device=rays_o.device,
        dtype=torch_type,
    )

    raymarching_train_kernel(
        rays_o.contiguous(),
        rays_d.contiguous(),
        hits_t.contiguous(),
        density_bitfield,
        noise,
        counter,
        rays_a,
        xyzs,
        dirs,
        deltas,
        ts,
        cascades, grid_size, scale,
        exp_step_factor, max_samples
    )

    # total samples for all rays
    total_samples = counter[0]
    # remove redundant output
    xyzs = xyzs[:total_samples]
    dirs = dirs[:total_samples]
    deltas = deltas[:total_samples]
    ts = ts[:total_samples]

    return rays_a, xyzs, dirs, deltas, ts, total_samples


@ti.kernel
def raymarching_test_kernel(
    rays_o: ti.types.ndarray(),
    rays_d: ti.types.ndarray(),
    hits_t: ti.types.ndarray(),
    alive_indices: ti.types.ndarray(),
    density_bitfield: ti.types.ndarray(),
    cascades: int,
    grid_size: int,
    scale: float,
    exp_step_factor: float,
    max_samples: int,
    ray_indices: ti.types.ndarray(),
    valid_mask: ti.types.ndarray(),
    deltas: ti.types.ndarray(),
    ts: ti.types.ndarray(),
    samples_counter: ti.types.ndarray(),
):

    for n in alive_indices:
        r = alive_indices[n]
        grid_size3 = grid_size**3
        grid_size_inv = 1.0 / grid_size

        ray_o = vec3(rays_o[r, 0], rays_o[r, 1], rays_o[r, 2])
        ray_d = vec3(rays_d[r, 0], rays_d[r, 1], rays_d[r, 2])
        d_inv = 1.0 / ray_d

        t = hits_t[r, 0]
        t2 = hits_t[r, 1]

        s = 0
        start_based = n * max_samples
        while (0 < t) & (t < t2) & (s < max_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            mip = ti.max(
                mip_from_pos(xyz, cascades),
                mip_from_dt(dt, grid_size, cascades)
            )

            mip_bound = ti.min(ti.pow(2., mip - 1), scale)
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(
                0.5 * (xyz * mip_bound_inv + 1) * grid_size,
                xmin=0.0,
                xmax=grid_size - 1.0
            )

            idx = mip * grid_size3 + __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx // 8)] & (1 << ti.u32(idx % 8))

            if occ:
                idx = start_based + s
                ray_indices[idx] = r
                valid_mask[idx] = 1
                ts[idx] = t
                deltas[idx] = dt
                t += dt
                hits_t[r, 0] = t
                s += 1
            else:
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)

        samples_counter[n] = s

def raymarching_test(
    rays_o,
    rays_d,
    hits_t,
    alive_indices,
    density_bitfield,
    cascades,
    scale,
    exp_step_factor,
    grid_size,
    max_samples,
):

    N_rays = alive_indices.size(0)
    ray_indices = torch.empty(
        N_rays*max_samples,
        device=rays_o.device,
        dtype=torch.long,
    )
    valid_mask = torch.zeros(
        N_rays*max_samples,
        device=rays_o.device,
        dtype=torch.uint8,
    )
    deltas = torch.empty(
        N_rays*max_samples,
        device=rays_o.device,
        dtype=rays_o.dtype
    )
    ts = torch.empty(
        N_rays*max_samples,
        device=rays_o.device,
        dtype=rays_o.dtype
    )
    samples_counter = torch.empty(
        N_rays,
        device=rays_o.device,
        dtype=torch.int32
    )

    raymarching_test_kernel(
        rays_o.contiguous(),
        rays_d.contiguous(),
        hits_t.contiguous(),
        alive_indices.contiguous(),
        density_bitfield,
        cascades,
        grid_size,
        scale,
        exp_step_factor,
        max_samples,
        ray_indices,
        valid_mask,
        deltas,
        ts,
        samples_counter
    )
    valid_mask = valid_mask.bool()
    cumsum = torch.cumsum(samples_counter, 0)
    packed_info = torch.stack([
        cumsum - samples_counter,
        samples_counter],
        dim=-1,
    )
    return packed_info, ray_indices[valid_mask], deltas[valid_mask], ts[valid_mask]
