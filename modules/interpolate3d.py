
# https://gist.github.com/Kulbear/af6499e83382df88c2a2c42fb3143652
import torch
import numpy as np

def gather_nd_torch(params, indices, batch_dim=1):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


def interpolate(grid_3d,
                sampling_points):
    """Trilinear interpolation on a 3D regular grid.

    This is a porting of TensorFlow Graphics implementation of trilinear interpolation.
    Check https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/trilinear.py
    for more details.

    Args:
      grid_3d: A tensor with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
        height, width, depth of the grid and C is the number of channels.
      sampling_points: A tensor with shape `[A1, ..., An, M, 3]` where M is the
        number of sampling points. Sampling points outside the grid are projected
        in the grid borders.

    Returns:
      A tensor of shape `[A1, ..., An, M, C]`
    """

    grid_3d_shape = grid_3d.size()
    sampling_points_shape = sampling_points.size()
    voxel_cube_shape = grid_3d_shape[-4:-1]  # [H, W, D]
    batch_dims = sampling_points_shape[:-2]  # [A1, ..., An]
    num_points = sampling_points_shape[-2]  # M

    bottom_left = torch.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = bottom_left.type(torch.int32)
    top_right_index = top_right.type(torch.int32)

    x0_index, y0_index, z0_index = torch.unbind(bottom_left_index, dim=-1)
    x1_index, y1_index, z1_index = torch.unbind(top_right_index, dim=-1)
    index_x = torch.concat([x0_index, x1_index, x0_index, x1_index,
                            x0_index, x1_index, x0_index, x1_index], dim=-1)
    index_y = torch.concat([y0_index, y0_index, y1_index, y1_index,
                            y0_index, y0_index, y1_index, y1_index], dim=-1)
    index_z = torch.concat([z0_index, z0_index, z0_index, z0_index,
                            z1_index, z1_index, z1_index, z1_index], dim=-1)
    indices = torch.stack([index_x, index_y, index_z], dim=-1)

    clip_value_max = (torch.tensor(list(voxel_cube_shape)) - 1).to(device=sampling_points.device)
    clip_value_min = torch.zeros_like(clip_value_max).to(device=sampling_points.device)
    indices = torch.clamp(indices, min=clip_value_min, max=clip_value_max)

    content = gather_nd_torch(
        params=grid_3d, indices=indices.long(), batch_dim=len(batch_dims))

    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = torch.unbind(distance_to_bottom_left, dim=-1)
    x1_x, y1_y, z1_z = torch.unbind(distance_to_top_right, dim=-1)
    weights_x = torch.concat([x1_x, x_x0, x1_x, x_x0,
                              x1_x, x_x0, x1_x, x_x0], dim=-1)
    weights_y = torch.concat([y1_y, y1_y, y_y0, y_y0,
                              y1_y, y1_y, y_y0, y_y0], dim=-1)
    weights_z = torch.concat([z1_z, z1_z, z1_z, z1_z,
                              z_z0, z_z0, z_z0, z_z0], dim=-1)

    weights = weights_x * weights_y * weights_z
    weights = weights.unsqueeze(-1)

    interpolated_values = weights * content

    return sum(torch.split(interpolated_values, [num_points] * 8, dim=-2))