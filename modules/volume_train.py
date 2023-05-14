import torch
import taichi as ti

from .utils import torch_type

@ti.kernel
def volume_rendering_kernel(
    sigmas: ti.types.ndarray(),
    rgbs: ti.types.ndarray(),
    deltas: ti.types.ndarray(),
    ts: ti.types.ndarray(),
    rays_a: ti.types.ndarray(),
    T_threshold: float,
    T: ti.types.ndarray(),
    total_samples: ti.types.ndarray(),
    opacity: ti.types.ndarray(),
    depth: ti.types.ndarray(),
    rgb: ti.types.ndarray(),
    ws: ti.types.ndarray()
):
    ti.loop_config(block_dim=128)
    for n in opacity:
        ray_idx = rays_a[n, 0]
        start_idx = rays_a[n, 1]
        N_samples = rays_a[n, 2]

        rgb[ray_idx, 0] = 0.0
        rgb[ray_idx, 1] = 0.0
        rgb[ray_idx, 2] = 0.0
        depth[ray_idx] = 0.0
        opacity[ray_idx] = 0.0
        total_samples[ray_idx] = 0

        T[start_idx] = 1.0
        for sample_ in range(N_samples):
            s = start_idx + sample_
            T_ = T[s]
            if T_ > T_threshold:
                a = 1.0 - ti.exp(-sigmas[s] * deltas[s])
                w = a * T_
                rgb[ray_idx, 0] += w * rgbs[s, 0]
                rgb[ray_idx, 1] += w * rgbs[s, 1]
                rgb[ray_idx, 2] += w * rgbs[s, 2]
                depth[ray_idx] += w * ts[s]
                opacity[ray_idx] += w
                ws[s] = w
                T[s+1] = T_ * (1.0 - a)
                total_samples[ray_idx] += 1



class VolumeRenderer(torch.nn.Module):

    def __init__(self):
        super(VolumeRenderer, self).__init__()

        self._volume_rendering_kernel = volume_rendering_kernel
        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(
                    ctx, 
                    sigmas, 
                    rgbs, 
                    deltas, 
                    ts, 
                    rays_a, 
                    T_threshold
                ):
                ctx.T_threshold = T_threshold
                n_rays = rays_a.shape[0]
                total_samples = torch.empty_like(rays_a[:, 0])
                opacity = torch.empty(
                    n_rays, 
                    dtype=torch_type,
                    device=rays_a.device, 
                    requires_grad=True
                )
                depth = torch.empty(
                    n_rays, 
                    dtype=torch_type,
                    device=rays_a.device, 
                    requires_grad=True
                )
                rgb = torch.empty(
                    n_rays, 3,
                    dtype=torch_type,
                    device=rays_a.device, 
                    requires_grad=True
                )
                ws = torch.empty_like(
                    sigmas, 
                    requires_grad=True
                )
                T_recap = torch.zeros_like(
                    sigmas, 
                    requires_grad=True
                )

                self._volume_rendering_kernel(
                    sigmas, 
                    rgbs, 
                    deltas, 
                    ts, 
                    rays_a, 
                    T_threshold,
                    T_recap,
                    total_samples,
                    opacity,
                    depth,
                    rgb,
                    ws,
                )
                ctx.save_for_backward(
                    sigmas, 
                    rgbs, 
                    deltas, 
                    ts, 
                    rays_a, 
                    T_recap, 
                    total_samples, 
                    opacity, 
                    depth, 
                    rgb, 
                    ws,
                )

                return total_samples.sum(), opacity, depth, rgb, ws

            @staticmethod
            def backward(
                    ctx, 
                    dL_dtotal_samples, 
                    dL_dopacity, 
                    dL_ddepth,
                    dL_drgb, dL_dws
                ):

                # get the saved tensors
                T_threshold = ctx.T_threshold
                (
                    sigmas,
                    rgbs,
                    deltas,
                    ts,
                    rays_a,
                    T_recap,
                    total_samples,
                    opacity,
                    depth,
                    rgb,
                    ws,
                ) = ctx.saved_tensors
                # put the gradients into the tensors before calling the grad kernel
                opacity.grad = dL_dopacity
                depth.grad = dL_ddepth
                rgb.grad = dL_drgb
                ws.grad = dL_dws

                self._volume_rendering_kernel.grad(
                    sigmas, 
                    rgbs, 
                    deltas, 
                    ts, 
                    rays_a, 
                    T_threshold,
                    T_recap,
                    total_samples,
                    opacity,
                    depth,
                    rgb,
                    ws,
                )

                return sigmas.grad, rgbs.grad, None, None, None, None

        self._module_function = _module_function.apply

    def forward(
            self, 
            sigmas, 
            rgbs, 
            deltas, 
            ts, 
            rays_a, 
            T_threshold
        ):
        return self._module_function(
            sigmas.contiguous(), 
            rgbs.contiguous(), 
            deltas.contiguous(), 
            ts.contiguous(), 
            rays_a.contiguous(), 
            T_threshold,
        )
