import torch
from torch_geometric.data import Data

from helpers.basic_classes import DataSet
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations


@torch.no_grad()
def prepare_raw_data(data, dataset: DataSet, loader, SO3_rotate: bool) -> Data:
    pos, y = data
    batch_size, n, _ = pos.shape

    pos = pos.reshape(-1, pos.shape[2])
    y = y.squeeze(1).to(dtype=torch.int64)
    batch = torch.arange(0, batch_size).repeat_interleave(n, dim=0)

    data = Data(pos=pos, y=y, batch=batch, dtype=torch.int64)

    data.edge_index = torch.tensor([[0], [0]])
    delattr(data, 'edge_attr')

    # data rotation
    batch_size = 1
    dim = data.pos.shape[1]
    points = data.pos.view(batch_size, -1, dim)
    if SO3_rotate:
        trot = Rotate(R=random_rotations(batch_size))
    else:
        trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True)
    data.pos = trot.transform_points(points).view(-1, dim)
    return data


@torch.no_grad()
def apply_noise(x: torch.Tensor, additive_noise: float, scale_noise: float) -> torch.Tensor:
    """
    Apply noise to the features

    :param x: torch.Tensor
    :param additive_noise: float
    :param scale_noise: float
    :return: noisy_x: float
    """
    if additive_noise:
        additive = 2 * additive_noise * torch.rand(x.shape, dtype=x.dtype, device=x.device) - additive_noise
    else:
        additive = 0

    if scale_noise == 1.0:
        scale = 1.0
    else:
        min_scale = 1 / scale_noise
        scale = (scale_noise - min_scale) * torch.rand(x.shape, dtype=x.dtype, device=x.device) + min_scale

    return scale * x + additive
