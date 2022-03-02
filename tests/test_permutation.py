import torch
import numpy as np

from universal_blocks.kronecker import KroneckerModel
from universal_blocks.contraction import ContractionLayer
from universal_blocks.linear import CustomLinear
from universal_blocks.vn_relu import VNLeakyReLU
from universal_blocks.vn_pooling import VNMaxPool, MeanPool
from model import Head, Model
from helpers.basic_classes import PoolType, KroneckerArgs, ReLUArgs, GeneralHeadArgs, Task

# Note: The following script tests that our models are indeed permutation invariant

# seed
seed = 0
torch.manual_seed(seed)

# KroneckerArgs
n_neighbors = 5
dynamic_knn = True

# ReLUArgs
relu_args = ReLUArgs(add=True, eps=0.0, share=False, negative_slope=0.0)

# GeneralHeadArgs
k = 6
in_channel = 128
out_channel = 64

drop_out = 0.0

# Loading Synthetic data
device = torch.device('cpu')
n = 50


def test_permutation_invariance(x: torch.Tensor, idx_permutation: np.array, model, rtol: float,
                                permute_output: bool = True):
    """
    tests permutation invariance
    :param x: torch.Tensor
        input data
    :param idx_permutation: np.array
        new order of points
    :param rtol: float
    :param permute_output: bool
    """
    permuted_model_output = model(x)
    if permute_output:
        permuted_model_output = permuted_model_output[idx_permutation]

    model_output_for_permuted_input = model(x[idx_permutation])
    assert torch.allclose(permuted_model_output, model_output_for_permuted_input, rtol=rtol), \
        f"{model.__class__.__name__} is not permutation invariant"


if __name__ == '__main__':
    idx_permutation = torch.from_numpy(np.random.choice(np.arange(n), n, replace=False))

    # kronecker
    kronecker_x = torch.randn(size=(n, 3)).double().to(device)
    kronecker_args = KroneckerArgs(n_neighbors=0, dynamic_knn=dynamic_knn)
    kronecker_model = KroneckerModel(channel=in_channel, kronecker_args=kronecker_args,
                                     k=k).double().to(device=device)
    test_permutation_invariance(x=kronecker_x, idx_permutation=idx_permutation,
                                model=kronecker_model, rtol=1e-5)

    kronecker_args = KroneckerArgs(n_neighbors=n_neighbors, dynamic_knn=dynamic_knn)
    kronecker_model = KroneckerModel(channel=in_channel, kronecker_args=kronecker_args,
                                     k=k).double().to(device=device)
    test_permutation_invariance(x=kronecker_x, idx_permutation=idx_permutation,
                                model=kronecker_model, rtol=1e-5)

    # contraction
    contraction_x = torch.randn(size=[n] + ([3] * k) + [in_channel]).double().to(device)
    contraction_model = ContractionLayer(channel=in_channel, k=k).double().to(device=device)
    test_permutation_invariance(x=contraction_x, idx_permutation=idx_permutation,
                                model=contraction_model, rtol=1e-5)

    # linear
    linear_x = torch.randn(size=(n, 3 ** 2, in_channel)).double().to(device)
    lineark2_model = CustomLinear(in_features=in_channel, k=k, out_features=out_channel).double().to(device=device)
    test_permutation_invariance(x=linear_x, idx_permutation=idx_permutation,
                                model=lineark2_model, rtol=1e-5)

    # relu
    relu_x = torch.randn(size=(n, 3 ** k, in_channel)).double().to(device)
    relu_model = VNLeakyReLU(channel=in_channel, k=k, relu_args=relu_args).double().to(device=device)
    test_permutation_invariance(x=relu_x, idx_permutation=idx_permutation,
                                model=relu_model, rtol=1e-4)

    # pooling
    pooling_x = torch.randn(size=(n, 3 ** k, in_channel)).double().to(device)
    max_pooling_model = VNMaxPool(channel=in_channel, k=k).double().to(device=device)
    mean_pooling_model = MeanPool(channel=in_channel, k=k).double().to(device=device)
    test_permutation_invariance(x=pooling_x, idx_permutation=idx_permutation,
                                model=max_pooling_model, rtol=1e-5, permute_output=False)
    test_permutation_invariance(x=pooling_x, idx_permutation=idx_permutation,
                                model=mean_pooling_model, rtol=1e-5, permute_output=False)

    # head
    model_x = torch.randn(size=(n, 3)).double().to(device)
    general_head_args_3d = GeneralHeadArgs(k=k, add_linears=True, u_shape=True, z_align=False,
                                           pool_type=PoolType.MEAN, in_channel=in_channel)
    head_3d = Head(kronecker_args=kronecker_args, relu_args=relu_args,
                   general_head_args=general_head_args_3d).double().to(device=device)
    test_permutation_invariance(x=model_x, idx_permutation=idx_permutation,
                                model=head_3d, rtol=1e-2)

    general_head_args_2d = GeneralHeadArgs(k=k, add_linears=True, u_shape=True, z_align=True,
                                           pool_type=PoolType.MEAN, in_channel=in_channel)
    head_2d = Head(kronecker_args=kronecker_args, relu_args=relu_args,
                   general_head_args=general_head_args_3d).double().to(device=device)
    test_permutation_invariance(x=model_x, idx_permutation=idx_permutation,
                                model=head_2d, rtol=1e-2)

    # Model
    model_3d = Model(task=Task.Classification, kronecker_args=kronecker_args, relu_args=relu_args,
                     general_head_args=general_head_args_3d, drop_out=drop_out,
                     out_channel=out_channel).double().to(device=device)
    test_permutation_invariance(x=model_x, idx_permutation=idx_permutation,
                                model=model_3d, rtol=1e-2, permute_output=False)

    model_2d = Model(task=Task.Classification, kronecker_args=kronecker_args, relu_args=relu_args,
                     general_head_args=general_head_args_2d, drop_out=drop_out,
                     out_channel=out_channel).double().to(device=device)
    test_permutation_invariance(x=model_x, idx_permutation=idx_permutation,
                                model=model_3d, rtol=1e-2, permute_output=False)
