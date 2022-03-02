import torch
import math as m

from universal_blocks.kronecker import KroneckerModel
from universal_blocks.contraction import ContractionLayer
from universal_blocks.linear import CustomLinear
from universal_blocks.vn_relu import VNLeakyReLU
from universal_blocks.vn_pooling import get_pooling_cls
from model import Head, Model
from helpers.basic_classes import PoolType, KroneckerArgs, ReLUArgs, GeneralHeadArgs, Task

# Note #1: The following script tests that our models are indeed rotation equivariant
# Note #2: for large values of in_channel, out_channel or k, the numerical error starts to hurt the model

# seed
seed = 10
torch.manual_seed(seed)

# KroneckerArgs
n_neighbors = 5
dynamic_knn = True

# ReLUArgs
relu_args = ReLUArgs(add=True, eps=0.0, share=False, negative_slope=0.0)

# GeneralHeadArgs
big_k = 6
smaller_k = 2
in_channel = 128
out_channel = 64

drop_out = 0.0

# Loading Synthetic data
device = torch.device('cpu')
n = 50


def general_3d_rotation_matrix(alpha: float = 30, beta: float = 40, gamma: float = 50) -> torch.Tensor:
    """
    :param alpha: float
        yaw angle, rotation around the z axis
    :param beta: float
        pitch angle, rotation around the y axis
    :param gamma: float
        roll angle, rotation around the x axis
    :return: rot: torch.Tensor
        the resulting rotation matrix
    """

    def rot_x(theta):
        return torch.Tensor([[1, 0, 0],
                             [0, m.cos(theta), -m.sin(theta)],
                             [0, m.sin(theta), m.cos(theta)]])

    def rot_y(theta):
        return torch.Tensor([[m.cos(theta), 0, m.sin(theta)],
                             [0, 1, 0],
                             [-m.sin(theta), 0, m.cos(theta)]])

    def rot_z(theta):
        return torch.Tensor([[m.cos(theta), -m.sin(theta), 0],
                             [m.sin(theta), m.cos(theta), 0],
                             [0, 0, 1]])

    return rot_z(theta=alpha) @ rot_y(theta=beta) @ rot_x(theta=gamma)


def general_2d_rotation_matrix(alpha: float = 30) -> torch.Tensor:
    """
    :param alpha: float
        yaw angle, rotation around the z axis
    :return: rot: torch.Tensor
        the resulting rotation matrix
    """
    return torch.Tensor([[m.cos(alpha), -m.sin(alpha)],
                         [m.sin(alpha), m.cos(alpha)]])


def rotation_matrix_times_k(rot: torch.Tensor, k: int) -> torch.Tensor:
    """
    Creates the rotation tensor for dimension R ** (3 ** k)
    :param rot:
        R ** 3 rotation matrix
    :param k: int
        the power to which we raise the feature dimension
    :return: rot_power_k: torch.Tensor
        rotation tensor for dimension R ** (3 ** k)
    """
    rot_power_k = rot.clone()
    for _ in range(k - 1):
        rot_power_k = torch.kron(rot_power_k, rot)
    return rot_power_k


def test_kronecker_rotation_equivariance(x: torch.Tensor, kronecker_args: KroneckerArgs, k: int, rtol: float):
    """
    tests rotation equivariance for our KroneckerModel
    :param x: torch.Tensor (batch_size * n, m, 1)
        input matrix
    :param kronecker_args: KroneckerArgs
    :param k: int
    :param rtol: float
    """
    kronecker_model = KroneckerModel(channel=in_channel, kronecker_args=kronecker_args,
                                     k=k).double().to(device=device)
    rot = general_3d_rotation_matrix()  # shape (3, 3)
    rot = rot.type(x.dtype)
    rot_power_k = rotation_matrix_times_k(rot=rot, k=k)  # shape (3 ** k, 3 ** k)

    model_output_for_rotated_input = kronecker_model(x=x @ rot)
    # shape (batch_size * n, 3 ** k, in_channel)

    model_output = kronecker_model(x=x)  # shape (batch_size * n, 3 ** k, in_channel)
    rotated_model_output = torch.einsum('bik,ij->bjk', model_output, rot_power_k)
    # shape (batch_size * n, 3 ** k, in_channel)

    assert torch.allclose(model_output_for_rotated_input, rotated_model_output, rtol=rtol), \
        f"{kronecker_model.__class__.__name__} is not rotation equivariant"


def test_contraction_rotation_equivariance(x: torch.Tensor):
    """
    tests rotation equivariance for our ContractionModel
    :param x: torch.Tenor (n, m ** k, in_channel)
        input data
    """
    contraction_model = ContractionLayer(channel=in_channel, k=big_k).double().to(device=device)
    rot = general_3d_rotation_matrix()  # shape (3, 3)
    rot = rot.type(x.dtype)
    rot_power_k = rotation_matrix_times_k(rot=rot, k=contraction_model.k)  # shape (3 ** k, 3 ** k)
    rot_power_k_minus_2 = rotation_matrix_times_k(rot=rot,
                                                  k=contraction_model.k - 2)  # shape (3 ** (k - 2), 3 ** (k - 2))
    starting_x_shape = x.shape

    view_x = x.view(starting_x_shape[0], starting_x_shape[1] ** big_k, starting_x_shape[-1])  # (n, m ** k, in_channel)
    rotated_input = torch.einsum('bik,ij->bjk', view_x, rot_power_k)  # (n, m ** k, in_channel)
    model_output_for_rotated_input = contraction_model(
        rotated_input.view(*starting_x_shape))  # shape (n, 3_1, 3_2, ..., 3_{k-2}, in_channel)

    model_output = contraction_model(x)  # shape (n, 3_1, 3_2, ..., 3_{k-2}, in_channel)
    model_output = model_output.view(starting_x_shape[0], starting_x_shape[1] ** (big_k - 2),
                                     starting_x_shape[-1])  # shape (n, 3 ** (k - 2), in_channel)
    rotated_model_output = torch.einsum('bik,ij->bjk', model_output,
                                        rot_power_k_minus_2)  # shape (n, 3 ** (k - 2), in_channel)
    rotated_model_output = rotated_model_output.view(*starting_x_shape[:-3], starting_x_shape[-1])
    # shape (n, 3_1, 3_2, ..., 3_{k-2}, in_channel)

    assert torch.allclose(model_output_for_rotated_input, rotated_model_output, rtol=5e-2), \
        f"{contraction_model.__class__.__name__} is not rotation equivariant"


def test_linear_rotation_equivariance(x: torch.Tensor):
    """
    tests rotation equivariance for our LinearModelK2
    :param x: torch.Tenor (n, m ** 2, in_channel)
        input data
    """
    linear_model = CustomLinear(in_features=in_channel, k=2, out_features=out_channel).double().to(device=device)
    rot = general_3d_rotation_matrix()  # shape (3, 3)
    rot = rot.type(x.dtype)
    rot_power_2 = rotation_matrix_times_k(rot=rot, k=2)  # shape (3 ** k, 3 ** k)

    rotated_input = torch.einsum('bik,ij->bjk', x, rot_power_2)  # shape (n, 3 ** k, in_channel)
    model_output_for_rotated_input = linear_model(rotated_input)  # shape (n, 3 ** k, out_channel)

    model_output = linear_model(x)  # shape (n, 3 ** k, out_channel)
    rotated_model_output = torch.einsum('bik,ij->bjk', model_output, rot_power_2)  # shape (n, 3 ** k, out_channel)

    assert torch.allclose(rotated_model_output, model_output_for_rotated_input, rtol=1e-2), \
        f"{linear_model.__class__.__name__} is not rotation equivariant"


def test_relu_rotation_equivariance(x: torch.Tensor):
    """
    tests rotation equivariance for our LinearModelK2

    :param x: torch.Tenor (n, m ** k, in_channel)
        input data
    """
    relu_model = VNLeakyReLU(channel=in_channel, k=big_k, relu_args=relu_args).double().to(device=device)
    rot = general_3d_rotation_matrix()  # shape (3, 3)
    rot = rot.type(x.dtype)

    rot_power_k = rotation_matrix_times_k(rot=rot, k=big_k)  # shape (3 ** k, 3 ** k)

    rotated_input = torch.einsum('bik,ij->bjk', x, rot_power_k)  # shape (n, 3 ** k, in_channel)
    model_output_for_rotated_input = relu_model(rotated_input)  # shape (n, 3 ** k, in_channel)

    model_output = relu_model(x)  # shape (n, 3 ** k, in_channel)
    rotated_model_output = torch.einsum('bik,ij->bjk', model_output, rot_power_k)  # shape (n, 3 ** k, in_channel)

    assert torch.allclose(rotated_model_output, model_output_for_rotated_input, rtol=1e-2), \
        f"{relu_model.__class__.__name__} is not rotation equivariant"


def test_pooling_rotation_equivariance(x: torch.Tensor, pool_type: PoolType):
    """
    tests rotation equivariance for our LinearModelK2
    :param x: torch.Tenor (batch_size * n, m ** k, in_channel)
        input data
    :param pool_type: PoolType
    """
    pooling_model = get_pooling_cls(pool_type=pool_type)(channel=x.shape[2], k=big_k).double().to(device=device)
    rot = general_3d_rotation_matrix()  # shape (3, 3)
    rot = rot.type(x.dtype)

    rot_power_k = rotation_matrix_times_k(rot=rot, k=big_k)  # shape (3 ** k, 3 ** k)

    rotated_input = torch.einsum('bik,ij->bjk', x, rot_power_k)  # shape (batch_size * n, 3 ** k, in_channel)
    model_output_for_rotated_input = pooling_model(rotated_input)
    # shape (batch_size * n, 3 ** k, in_channel)

    model_output = pooling_model(x)  # shape (batch_size, 3 ** k, in_channel)
    rotated_model_output = torch.einsum('bik,ij->bjk', model_output, rot_power_k)
    # shape (batch_size, 3 ** k, in_channel)

    assert torch.allclose(rotated_model_output, model_output_for_rotated_input, rtol=1e-4), \
        f"{pooling_model.__class__.__name__} is not rotation invariant"


def test_rotation_invariance(x: torch.Tensor, z_align: bool, full_model: bool):
    """
    tests rotation invariance
    :param x: torch.Tenor (batch_size * n, m)
        input data
    :param z_align: bool
            whether or not to assume that the objects are aligned with respect to the gravity axis
    """
    kronecker_args = KroneckerArgs(n_neighbors=n_neighbors, dynamic_knn=dynamic_knn)
    general_head_args = GeneralHeadArgs(k=smaller_k, add_linears=True, u_shape=True, z_align=z_align,
                                        pool_type=PoolType.MEAN, in_channel=in_channel)

    if z_align:
        rot = general_2d_rotation_matrix()  # shape (2, 2)
        rot = rot.type(x.dtype)
        x_rotated = torch.cat((x[:, :2] @ rot, x[:, 2].unsqueeze(1)), dim=-1)
    else:
        rot = general_3d_rotation_matrix()  # shape (3, 3)
        rot = rot.type(x.dtype)
        x_rotated = x @ rot

    if full_model:
        model = Model(task=Task.Classification, kronecker_args=kronecker_args, relu_args=relu_args,
                      general_head_args=general_head_args, drop_out=drop_out,
                      out_channel=out_channel).double().to(device=device)
    else:
        model = Head(kronecker_args=kronecker_args, relu_args=relu_args,
                     general_head_args=general_head_args).double().to(device=device)

    model_output = model(x=x)  # shape (n, out_channel)
    model_output_for_rotated_input = model(x=x_rotated)  # shape (n, out_channel)
    assert torch.allclose(model_output, model_output_for_rotated_input, rtol=1e-3), \
        f"{model.__class__.__name__} with z_align {z_align} is not rotation invariant"


if __name__ == '__main__':
    # kronecker
    kronecker_x = torch.randn(size=(n, 3)).double().to(device)
    kronecker_args = KroneckerArgs(n_neighbors=0, dynamic_knn=dynamic_knn)
    test_kronecker_rotation_equivariance(x=kronecker_x, kronecker_args=kronecker_args, k=big_k, rtol=1e-5)

    kronecker_args = KroneckerArgs(n_neighbors=n_neighbors, dynamic_knn=dynamic_knn)
    test_kronecker_rotation_equivariance(x=kronecker_x, kronecker_args=kronecker_args, k=big_k, rtol=1e-3)

    # contraction
    contraction_x = torch.randn(size=[n] + ([3] * big_k) + [in_channel]).double().to(device)
    test_contraction_rotation_equivariance(x=contraction_x)

    # linear
    linear_x = torch.randn(size=(n, 3 ** 2, in_channel)).double().to(device)
    test_linear_rotation_equivariance(x=linear_x)

    # relu
    relu_x = torch.randn(size=(n, 3 ** big_k, in_channel)).double().to(device)
    test_relu_rotation_equivariance(x=relu_x)

    # pooling
    pooling_x = torch.randn(size=(n, 3 ** big_k, in_channel)).double().to(device)
    test_pooling_rotation_equivariance(x=pooling_x, pool_type=PoolType.MAX)
    test_pooling_rotation_equivariance(x=pooling_x, pool_type=PoolType.MEAN)

    # model
    model_x = torch.randn(size=(n, 3)).double().to(device)
    test_rotation_invariance(x=model_x, z_align=False, full_model=False)
    test_rotation_invariance(x=model_x, z_align=True, full_model=False)
    test_rotation_invariance(x=model_x, z_align=False, full_model=True)
    test_rotation_invariance(x=model_x, z_align=True, full_model=True)
