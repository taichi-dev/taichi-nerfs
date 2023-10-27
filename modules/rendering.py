import torch
from einops import rearrange

from .intersection import ray_aabb_intersection
from .ray_march import raymarching_test, raymarching_train
from .volume_render_test import composite_test

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


def render(
    model,
    rays_o,
    rays_d,
    test_time=False,
    exp_step_factor=0,
    T_threshold=1e-4,
    max_samples=MAX_SAMPLES,
):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)
    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
    Outputs:
        result: dictionary containing final rgb and depth
    """

    hits_t = ray_aabb_intersection(
        rays_o.contiguous(),
        rays_d.contiguous(),
        model.scale
    )

    if test_time:
        return __render_rays_test(
            model,
            rays_o,
            rays_d,
            hits_t,
            exp_step_factor=exp_step_factor,
            T_threshold=T_threshold,
            max_samples=max_samples,
        )
    else:
        return __render_rays_train(
            model,
            rays_o,
            rays_d,
            hits_t,
            exp_step_factor=exp_step_factor,
            T_threshold=T_threshold,
        )



@torch.no_grad()
def __render_rays_test(
    model,
    rays_o,
    rays_d,
    hits_t,
    exp_step_factor=0,
    T_threshold=1e-4,
    max_samples=MAX_SAMPLES,
):
    """
    Render rays by
    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor == 0 else 4

    while samples < max_samples:
        N_alive = len(alive_indices)
        if N_alive == 0:
            break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), min_samples)
        samples += N_samples

        (
            pack_info,
            ray_indices,
            deltas,
            ts,
        ) = raymarching_test(
            rays_o,
            rays_d,
            hits_t,
            alive_indices,
            model.density_bitfield,
            model.cascades,
            model.scale,
            exp_step_factor,
            model.grid_size,
            N_samples
        )
        if ray_indices.shape[0] == 0:
            break
        ray_o_local = rays_o[ray_indices, :3]
        ray_d_local = rays_d[ray_indices, :3]
        xyzs = ray_o_local + ts[:, None] * ray_d_local
        dirs = ray_d_local

        sigmas, rgbs = model(xyzs, dirs)

        composite_test(
            sigmas,
            rgbs,
            deltas,
            ts,
            pack_info,
            alive_indices,
            T_threshold,
            opacity,
            depth,
            rgb
        )
        # remove converged rays
        alive_indices = alive_indices[alive_indices >= 0]
        total_samples += pack_info[:, 1].sum()

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples  # total samples for all rays

    if exp_step_factor == 0:  # synthetic
        rgb_bg = torch.ones(3, device=device)
    else:  # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg * rearrange(1 - opacity, 'n -> n 1')

    return results


def __render_rays_train(
    model,
    rays_o,
    rays_d,
    hits_t,
    exp_step_factor=0,
    T_threshold=1e-4,
):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    results = {}

    (
        rays_a,
        xyzs,
        dirs,
        results['deltas'],
        results['ts'],
        results['rm_samples']
    ) = raymarching_train(
        rays_o,
        rays_d,
        hits_t,
        model.density_bitfield,
        model.cascades,
        model.scale,
        exp_step_factor,
        model.grid_size,
        MAX_SAMPLES
    )

    sigmas, rgbs = model(xyzs, dirs)

    (
        results['vr_samples'],
        results['opacity'],
        results['depth'],
        results['rgb'],
        results['ws']
    ) = model.render_func(
        sigmas,
        rgbs,
        results['deltas'],
        results['ts'],
        rays_a,
        T_threshold
    )

    results['rays_a'] = rays_a

    if exp_step_factor == 0:
        # synthetic
        rgb_bg = torch.ones(3, device=rays_o.device)
    else:
        # real
        rgb_bg = torch.zeros(3, device=rays_o.device)
    results['rgb'] = results['rgb'] + \
                     rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

    return results
