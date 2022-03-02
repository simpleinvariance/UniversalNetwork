import torch
import itertools
from torch.nn import Module, Parameter


class ContractionLayer(Module):
    def __init__(self, channel: int, k: int):
        """
        Our dimension down sampling layer

        :param channel: int
                number of channels
        :param k: int
                the power to which we raised the feature dimension
        """
        super(ContractionLayer, self).__init__()
        assert k > 1, "k has to be equal or above 2"
        assert k % 2 == 0, 'Can not contract an odd k'

        self.channel = channel
        self.k = k

        self.k_choose_2 = int((k * (k - 1)) / 2)

        # theta needs additional dimension as the multiplication at the end of the forward is not "broadcast-able"
        self.theta = Parameter(data=torch.randn(size=(self.k_choose_2, channel)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates a linear combination of all possible dual contractions, for each channel

        :param x: torch.Tensor (n, m_1,..., m_k, channel)
                    input matrix
        :return: model_output: torch.Tensor (n, m_1,..., m_{k-2}, channel)
        """
        assert len(list(x.size())) == 2 + self.k, 'Matrix X is not a (k + 2)-d tensor'
        assert (torch.Tensor(list(x.shape[1:-1])) == x.shape[1]).sum() == self.k, \
            'Matrix X dimensions 1 to -2 have to be the same'

        current_device = x.get_device() if x.is_cuda else torch.device('cpu')
        combinations_of_dimensions = itertools.combinations(range(1, self.k + 1), 2)
        linear_combination_of_diagonals = torch.zeros(size=list(x.shape[:-3]) + [x.shape[-1]], dtype=x.dtype,
                                                      device=current_device)

        for dims, theta_per_trace in zip(combinations_of_dimensions, self.theta):
            linear_combination_of_diagonals += theta_per_trace * torch.diagonal(x, dim1=dims[0], dim2=dims[1]).sum(-1)
        return linear_combination_of_diagonals / self.k_choose_2  # shape (n, m_1,..., m_{k-2}, channel)
