import torch
import torch.nn as nn

from helpers.basic_classes import PoolType


def get_pooling_cls(pool_type: PoolType):
    if pool_type is PoolType.MAX:
        return VNMaxPool
    elif pool_type is PoolType.MEAN:
        return MeanPool


class VNMaxPool(nn.Module):
    def __init__(self, channel: int, k: int):
        """
        VN's MaxPool implementation

        :param channel: int
                number of input features
        :param k: int
                the power to which we raise the feature dimension
        """
        super(VNMaxPool, self).__init__()

        self.channel = channel
        self.k = k

        self.map_to_dir = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        """
        Forward VN's MaxPool implementation

        :param x: torch.Tensor (n, m_1,..., m_k, channel) or (n, m ** k, channel)
                    input matrix
        :return: x_out: torch.Tensor (1, m_1,..., m_k, channel) or (1, m ** k, channel)
        """
        starting_x_shape = x.shape[1:]
        if len(list(x.size())) == 3:  # x is of shape (n, m ** k, channel)
            m_float = x.shape[1] ** (1 / self.k)
            assert x.shape[1] == round(m_float) ** self.k, \
                f'Dimension 1 of matrix X should equal to an integer to the power of {self.k}'
            assert x.shape[2] == self.channel, 'Dimension 2 of matrix X should equal to channel'

        else:  # x is of shape (n, m_1,..., m_k, channel)
            assert len(list(x.size())) == 2 + self.k, 'Matrix X is not a (k + 2)-d tensor'
            assert (torch.Tensor(list(x.shape[1:-1])) == x.shape[1]).sum() == self.k, \
                'Matrix X dimensions 1 to -2 have to be the same'
            assert x.shape[-1] == self.channel, 'Dimension -1 of matrix X should equal to channel'

            x = x.view(x.shape[0], -1, self.channel)  # shape (n, m ** k, channel)

        d = self.map_to_dir(x)
        dotprod = (x * d).sum(dim=1, keepdim=True)
        idx = dotprod.max(dim=0, keepdim=False)[1]
        index_tuple = (idx,) + torch.meshgrid([torch.arange(j) for j in x.size()[1:]])
        x_max = x[index_tuple]
        return x_max.view(1, *starting_x_shape)


class MeanPool(nn.Module):
    def __init__(self, channel, k):
        super(MeanPool, self).__init__()

    def forward(self, x):
        return x.mean(dim=0, keepdim=True)
