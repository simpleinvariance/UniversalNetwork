import torch
from sklearn.neighbors import NearestNeighbors
from torch.nn import Module, ModuleList
import numpy as np

from helpers.basic_classes import PoolType, KroneckerArgs, ReLUArgs, GeneralHeadArgs, Task
from universal_blocks.kronecker import KroneckerLayer
from universal_blocks.contraction import ContractionLayer
from universal_blocks.linear import CustomLinear
from universal_blocks.vn_relu import VNLeakyReLU
from universal_blocks.vn_pooling import get_pooling_cls
from helpers.mlp import MLP


class Head(Module):
    """
    Our model for feature extractions

    :param kronecker_args: KroneckerArgs
    :param relu_args: ReLUArgs
    :param general_head_args: GeneralHeadArgs
    """

    def __init__(self, kronecker_args: KroneckerArgs, relu_args: ReLUArgs, general_head_args: GeneralHeadArgs):
        super(Head, self).__init__()
        n_neighbors, dynamic_knn = kronecker_args
        add, eps, share, negative_slope = relu_args
        k, input_in_channel, add_linears, u_shape, z_align, pool_type = general_head_args

        assert k > 1, "k has to be equal or above 2"

        # The KNearestNeighbors preprocessing
        self.nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors) if n_neighbors else None

        # The layers of kronecker products
        self.kronecker_layers = ModuleList()
        kronecker_channels = np.array([input_in_channel] * (k + 1))
        for in_channel, out_channel, out_k in zip(kronecker_channels[:-1], kronecker_channels[1:], range(1, k + 1)):
            self.kronecker_layers.append(KroneckerLayer(channel=in_channel, kronecker_args=kronecker_args))
            relu_channel = in_channel

            if add_linears or out_k == k:
                self.kronecker_layers.append(CustomLinear(in_features=in_channel, k=out_k, out_features=out_channel))
                relu_channel = out_channel
            if add:
                self.kronecker_layers.append(VNLeakyReLU(channel=relu_channel, k=out_k, relu_args=relu_args))
        # pooling
        pool_channels = 0
        if pool_type is not PoolType.NONE:
            pool_channels = kronecker_channels[-1]
            self.pooling_layer = get_pooling_cls(pool_type=pool_type)(channel=pool_channels, k=k)

        # The layers of contraction
        self.contraction_layers = ModuleList()
        contraction_channels = np.array([input_in_channel + pool_channels] * (k // 2 + 1))
        for idx, out_k in enumerate(range(k, 0, -2)):
            if u_shape and idx != 0:
                contraction_channels[idx:] += kronecker_channels[-2 * idx]

            self.contraction_layers.append(ContractionLayer(channel=contraction_channels[idx], k=out_k))
            relu_channel = contraction_channels[idx]
            if out_k - 2 != 0:
                if add_linears:
                    self.contraction_layers.append(CustomLinear(in_features=contraction_channels[idx], k=out_k - 2,
                                                                out_features=contraction_channels[idx + 1]))
                    relu_channel = contraction_channels[idx + 1]
                if add:
                    self.contraction_layers.append(VNLeakyReLU(channel=relu_channel, k=out_k - 2, relu_args=relu_args))

        self.k, self.add_linears, self.u_shape, self.z_align = k, add_linears, u_shape, z_align
        self.n_neighbors, self.dynamic_knn, self.pool_type = n_neighbors, dynamic_knn, pool_type
        self.in_channel = input_in_channel
        self.out_channel = contraction_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward thru the kronecker product and the contractions

        :param x: torch.Tensor (n, m)
        :return: head_output: torch.Tensor (batch_size * n, channel)
        """
        n, m = x.shape
        # Choosing the rotations to be in 3D or 2D
        if self.z_align:
            assert x.shape[1] == 3, 'Matrix X is not a 3-d point cloud'
            z_axis = x[:, 2]
            x = x[:, :2]
            m -= 1

        # forward thru the kronecker product
        assert len(list(x.size())) == 2, 'Matrix X is not a 2-d matrix'
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        v = torch.ones(size=(n, 1, self.in_channel), dtype=x.dtype, device=device)
        x = x.unsqueeze(2)

        kronecker_mid_results = []
        out_k = 0
        allow_once = True
        for layer in self.kronecker_layers:
            # KNearestNeighbors
            if self.n_neighbors:
                if allow_once or self.dynamic_knn:
                    neighbor_space = x if allow_once else v
                    neighbor_space = neighbor_space.detach().cpu().numpy().reshape(n, -1)
                    nbrs = self.nearest_neighbors.fit(X=neighbor_space)
                    neighbors_indices = nbrs.kneighbors(X=neighbor_space)[1]
                    neighbors_indices = torch.from_numpy(neighbors_indices).to(device=device)
                allow_once = False
            else:
                neighbors_indices = None

            if layer.__class__.__name__ == 'KroneckerLayer':
                v = layer(x=x, v=v, neighbors_indices=neighbors_indices)  # shape (batch_size * n, m ** i, channel)
                # where n is the number of point in a single graph
                out_k += 1

                if self.u_shape and not self.add_linears and (out_k % 2 == 0 and out_k != self.k):
                    kronecker_mid_results.append((v, layer.channel))
            else:
                v = layer(x=v)
                if self.u_shape and self.add_linears and layer.__class__.__name__ == 'CustomLinear' \
                        and (out_k % 2 == 0 and out_k != self.k):
                    kronecker_mid_results.append((v, layer.out_features))
        x = v  # shape (batch_size * n, m ** k, channel)

        # pooling
        if self.pool_type is not PoolType.NONE:
            x_global = self.pooling_layer(x).repeat(n, 1, 1)  # shape (batch_size * n, m ** k, 2 * channel)
            x = torch.cat((x, x_global), dim=2)

        # forward thru the contractions
        assert len(list(x.size())) == 3, 'Matrix X is a 3-d tensor'
        assert x.shape[1] == m ** self.k, \
            f'Dimension 1 of matrix X should equal to {m} to the power of {self.k}'

        view_dims = [n] + [m] * self.k
        x = x.view(*view_dims, x.shape[-1])  # shape (batch_size * n, m_1, ..., m_k, channel)

        result_idx = 1
        for idx, layer in enumerate(self.contraction_layers):
            if layer.__class__.__name__ == 'ContractionLayer' and self.u_shape and idx != 0:
                view_dims = view_dims[:-2]
                result, channel = kronecker_mid_results[-result_idx]
                x = torch.cat((x, result.view(*view_dims, channel)), dim=-1)
                result_idx += 1
            x = layer(x=x)  # shape (batch_size * n, m_1, ..., m_k, channel)

        if self.z_align:
            x = torch.cat((x, z_axis.unsqueeze(1)), dim=-1)
        return x  # shape (batch_size * n, channel)


class Model(Module):
    """
    Our model for classification tasks

    :param task: Task
    :param kronecker_args: KroneckerArgs
    :param relu_args: ReLUArgs
    :param general_head_args: GeneralHeadArgs
    :param drop_out: float
    :param out_channel: int
    """

    def __init__(self, task: Task, kronecker_args: KroneckerArgs, relu_args: ReLUArgs,
                 general_head_args: GeneralHeadArgs, drop_out: float, out_channel: int):
        super(Model, self).__init__()
        self.head = Head(kronecker_args=kronecker_args, relu_args=relu_args, general_head_args=general_head_args)
        head_out_channel = self.head.out_channel + general_head_args.z_align

        self.mlp = MLP(channels=[head_out_channel, general_head_args.in_channel, general_head_args.in_channel,
                       out_channel], drop_out=drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward thru Head + a pooling layer + MLP

        :param x: torch.Tensor (n, m)
        :return: model_output: torch.Tensor (batch_size, channels[-1])
        """
        head_output = self.head(x=x)  # shape (batch_size * n, channels[-1])
        x_invariant = head_output.mean(dim=0, keepdim=True)  # shape (batch_size, channels[-1])
        return self.mlp(x_invariant)  # shape (batch_size, channels[-1])
