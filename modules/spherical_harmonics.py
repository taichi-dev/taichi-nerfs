import torch
import taichi as ti

data_type = ti.f32
torch_type = torch.float32

@ti.kernel
def dir_encoder(
    dirs: ti.types.ndarray(), 
    embedding: ti.types.ndarray(), 
    B: ti.i32,
):
    # spherical_harmonics
    ti.loop_config(block_dim=512)
    for i in ti.ndrange(B):
        x = dirs[i, 0]
        y = dirs[i, 1]
        z = dirs[i, 2]

        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z

        embedding[i, 0] = (0.28209479177387814)
        embedding[i, 1] = (-0.48860251190291987 * y)
        embedding[i, 2] = (0.48860251190291987 * z)
        embedding[i, 3] = (-0.48860251190291987 * x)
        embedding[i, 4] = (1.0925484305920792 * xy)
        embedding[i, 5] = (-1.0925484305920792 * yz)
        embedding[i, 6] = (0.94617469575755997 * z2 - 0.31539156525251999)
        embedding[i, 7] = (-1.0925484305920792 * xz)
        embedding[i, 8] = (0.54627421529603959 * x2 - 0.54627421529603959 * y2)
        embedding[i, 9] = (0.59004358992664352 * y * (-3.0 * x2 + y2))
        embedding[i, 10] = (2.8906114426405538 * xy * z)
        embedding[i, 11] = (0.45704579946446572 * y * (1.0 - 5.0 * z2))
        embedding[i, 12] = (0.3731763325901154 * z * (5.0 * z2 - 3.0))
        embedding[i, 13] = (0.45704579946446572 * x * (1.0 - 5.0 * z2))
        embedding[i, 14] = (1.4453057213202769 * z * (x2 - y2))
        embedding[i, 15] = (0.59004358992664352 * x * (-x2 + 3.0 * y2))

        # embedding[i, 0] = data_type(0.28209479177387814)
        # embedding[i, 1] = data_type(-0.48860251190291987 * y)
        # embedding[i, 2] = data_type(0.48860251190291987 * z)
        # embedding[i, 3] = data_type(-0.48860251190291987 * x)
        # embedding[i, 4] = data_type(1.0925484305920792 * xy)
        # embedding[i, 5] = data_type(-1.0925484305920792 * yz)
        # embedding[i, 6] = data_type(0.94617469575755997 * z2 - 0.31539156525251999)
        # embedding[i, 7] = data_type(-1.0925484305920792 * xz)
        # embedding[i, 8] = data_type(0.54627421529603959 * x2 - 0.54627421529603959 * y2)
        # embedding[i, 9] = data_type(0.59004358992664352 * y * (-3.0 * x2 + y2))
        # embedding[i, 10] = data_type(2.8906114426405538 * xy * z)
        # embedding[i, 11] = data_type(0.45704579946446572 * y * (1.0 - 5.0 * z2))
        # embedding[i, 12] = data_type(0.3731763325901154 * z * (5.0 * z2 - 3.0))
        # embedding[i, 13] = data_type(0.45704579946446572 * x * (1.0 - 5.0 * z2))
        # embedding[i, 14] = data_type(1.4453057213202769 * z * (x2 - y2))
        # embedding[i, 15] = data_type(0.59004358992664352 * x * (-x2 + 3.0 * y2))


class DirEncoder(torch.nn.Module):

    def __init__(self):
        super(DirEncoder, self).__init__()

        self._dir_encoder_kernel = dir_encoder
        self.out_dim = 16

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_dir):
                output_embedding = torch.empty(
                    input_dir.shape[0], 16,
                    dtype=torch_type,
                    device=input_dir.device,
                    requires_grad=True,
                )
                self._dir_encoder_kernel(
                    input_dir, 
                    output_embedding,
                    input_dir.shape[0]
                )
                ctx.save_for_backward(input_dir, output_embedding)
                return output_embedding

            @staticmethod
            def backward(ctx, doutput):
                input_dir, output_embedding = ctx.saved_tensors
                output_embedding.grad = doutput
                self._dir_encoder_kernel.grad(
                    input_dir, 
                    output_embedding,
                    input_dir.shape[0]
                )
                return input_dir.grad

        self._module_function = _module_function.apply

    def forward(self, dirs):
        return self._module_function(dirs.contiguous())
