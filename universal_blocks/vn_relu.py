import torch
import torch.nn as nn

from helpers.basic_classes import ReLUArgs


class VNLeakyReLU(nn.Module):
    def __init__(self, channel: int, k: int, relu_args: ReLUArgs):
        """
        VN's LeakyReLU implementation

        :param channel: int
            number of input features
        :param k: int
            the power to which we raise the feature dimension
        :param relu_args: ReLUArgs
        """
        super(VNLeakyReLU, self).__init__()

        self.channel = channel
        self.k = k
        self.share_nonlinearity = relu_args.share
        self.negative_slope = relu_args.negative_slope
        self.eps = relu_args.eps

        if relu_args.share:
            self.map_to_dir = nn.Linear(channel, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        """
        Forward VN's LeakyReLU implementation

        :param x: torch.Tensor (n, m_1,..., m_k, channel) or (n, m ** k, channel)
                    input matrix
        :return: x_out: torch.Tensor (n, m_1,..., m_k, channel) or (n, m ** k, channel)
        """
        starting_x_shape = x.shape
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

        k = self.map_to_dir(x)
        dotprod = (x * k).sum(dim=1, keepdim=True)
        mask = (dotprod < 0).float()
        k_norm_sq = (k * k).sum(dim=1, keepdim=True)
        model_output = x - mask * (1 - self.negative_slope) * dotprod * k / (
                    k_norm_sq + self.eps)  # shape (n, m ** k, channel)
        return model_output.view(*starting_x_shape)
