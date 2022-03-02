import torch
from torch.nn import Module, Parameter, ModuleList
from sklearn.neighbors import NearestNeighbors

from helpers.basic_classes import KroneckerArgs
from helpers.mlp import MLP


class KroneckerLayer(Module):
    def __init__(self, channel: int, kronecker_args: KroneckerArgs):
        """
        Our dimension up sampling layer

        :param channel: int
                number of channels
        """
        super(KroneckerLayer, self).__init__()

        self.channel = channel
        self.n_neighbors = kronecker_args.n_neighbors

        self.theta1 = Parameter(data=torch.randn(size=(channel,)), requires_grad=True)
        self.theta2 = Parameter(data=torch.randn(size=(channel,)), requires_grad=True)

        if kronecker_args.n_neighbors:
            self.theta3 = Parameter(data=torch.randn(size=(channel,)), requires_grad=True)

    def forward(self, x: torch.Tensor, v: torch.Tensor, neighbors_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate the kronecker product

        :param x: torch.Tensor (n, m, 1)
                input matrix
        :param v: torch.Tensor (n, d, channel)
                Output tensor from last iteration
        :param neighbors_indices: torch.Tensor (n, n_neighbors)
        :return: kron_result: torch.Tensor (n, m * d, channel)
        """
        n = x.shape[0]
        assert x.shape[2] == 1, 'Dimension 2 of matrix X should equal 1'
        assert v.shape[2] == self.channel, f'Dimension 2 of matrix V should equal c: {self.channel}'
        assert v.shape[0] == n, f'Dimensions 0 of matrix V should equal n: {n}'

        assert neighbors_indices is None or neighbors_indices.shape[0] == n, \
            f'Dimensions 0 of matrix neighbors_indices should equal n: {n}'
        assert neighbors_indices is None or neighbors_indices.shape[1] == self.n_neighbors,\
            f'Dimensions 1 of matrix neighbors_indices should equal n_neighbors: {self.n_neighbors}'

        kron_products = (x[:, :, None, :, None] * v[:, None, :, None, :])\
            .view(n, x.shape[1] * v.shape[1], self.channel)  # shape (n, m * d, channel)

        if self.n_neighbors:
            return self.theta1 * kron_products + (self.theta2 * kron_products).mean(dim=0)\
                   + (self.theta3 * kron_products[neighbors_indices, :, :]).mean(dim=1)  # shape (m * d, channel)
        else:
            return self.theta1 * kron_products + (self.theta2 * kron_products).mean(dim=0)  # shape (m * d, channel)


class KroneckerModel(Module):
    def __init__(self, channel: int, kronecker_args: KroneckerArgs, k: int):
        """
        Our dimension up sampling model - THIS CLASS IS USED TO HELP TESTS
        :param channel: int
                number of channels
        :param kronecker_args: KroneckerArgs
        :param k: int
        """
        super(KroneckerModel, self).__init__()

        self.channel = channel
        self.n_neighbors = kronecker_args.n_neighbors

        if kronecker_args.n_neighbors:
            self.nearest_neighbors = NearestNeighbors(n_neighbors=kronecker_args.n_neighbors)
        else:
            self.nearest_neighbors = None
        self.kronecker_layers = ModuleList([KroneckerLayer(channel=channel, kronecker_args=kronecker_args)
                                            for _ in range(k)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward thru the kronecker product
        :param x: torch.Tensor (n, m)
        :return: head_output: torch.Tensor (n, 3 ** k, m)
        """
        assert len(list(x.size())) == 2, 'Matrix X is not a 2-d matrix'
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        if self.n_neighbors:
            nbrs = self.nearest_neighbors.fit(X=x.cpu().numpy())
            neighbors_indices = torch.from_numpy(nbrs.kneighbors(X=x)[1]).to(device=device)
        else:
            neighbors_indices = None

        v = torch.ones(size=(x.shape[0], 1, self.channel), dtype=x.dtype, device=x.device)
        x = x.unsqueeze(2)

        for layer in self.kronecker_layers:
            v = layer(x=x, v=v, neighbors_indices=neighbors_indices)  # shape (n, m ** i, channel)
        return v
