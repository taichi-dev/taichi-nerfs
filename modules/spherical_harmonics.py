import taichi as ti
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import torch2ti  # data_type, torch_type
from .utils import ti2torch, ti2torch_grad, torch2ti_grad

data_type = ti.f16
torch_type = torch.float16


@ti.kernel
def dir_encoder(dirs: ti.template(), embedding: ti.template(), B: ti.i32):
    # spherical_harmonics
    ti.loop_config(block_dim=256)
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


class DirEncoder(torch.nn.Module):

    def __init__(self, batch_size=8192):
        super(DirEncoder, self).__init__()

        self.input_fields = ti.field(dtype=data_type,
                                     shape=(batch_size * 1024, 3),
                                     needs_grad=True)
        self.output_fields = ti.field(dtype=data_type,
                                      shape=(batch_size * 1024, 16),
                                      needs_grad=True)

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_dir):
                # If no output gradient is provided, no need to
                # automatically materialize it as torch.zeros.

                # ctx.set_materialize_grads(False) # maybe not needed
                # input_dir = input_dir.to(torch.float16)
                ctx.input_size = input_dir.shape
                output_embedding = torch.zeros(input_dir.shape[0],
                                               16,
                                               dtype=torch_type,
                                               device=input_dir.device)

                # ti.sync()
                torch2ti(self.input_fields, input_dir.contiguous())
                dir_encoder(self.input_fields, self.output_fields,
                            input_dir.shape[0])
                ti2torch(self.output_fields, output_embedding)
                # ti.sync()

                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                input_size = ctx.input_size
                grad = torch.zeros(*input_size,
                                   device=doutput.device,
                                   dtype=torch_type)
                # doutput *= 128

                # zero out the gradient
                self.input_fields.grad.fill(0)
                # ti.sync()
                torch2ti_grad(self.output_fields, doutput.contiguous())
                dir_encoder.grad(self.input_fields, self.output_fields,
                                 doutput.shape[0])
                ti2torch_grad(self.input_fields, grad)
                # ti.sync()

                return grad

        self._module_function = _module_function

    def forward(self, dirs):
        return self._module_function.apply(dirs)
