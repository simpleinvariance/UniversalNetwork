import torch
from torch.nn import Module, Linear


class LinearModelK2(Module):
    def __init__(self, in_features: int, k: int, out_features: int):
        """
        A linear block for k = 2

        :param in_features: int
                number of input features
        :param k: int
                the power to which we raise the feature dimension
        :param out_features: int
                number of output features
        """
        super(LinearModelK2, self).__init__()
        assert k == 2, "LinearModelK2 work only for k=2"

        self.in_features = in_features
        self.k = k
        self.out_features = out_features

        self.alpha = Linear(in_features=in_features, out_features=out_features, bias=False)
        self.beta = Linear(in_features=in_features, out_features=out_features, bias=False)
        self.gamma = Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward thru the weighted sum of all linear and equivariant transformations

        :param x: torch.Tensor (n, m, m, in_features) or (n, m ** 2, in_features)
                    input matrix
        :return: model_output: torch.Tensor (n, m, m, out_features) or (n, m ** 2, out_features)
        """
        starting_x_shape = x.shape[:-1]
        if len(list(x.size())) == 3:  # x is of shape (n, m ** 2, in_features)
            m_float = x.shape[1] ** (1 / self.k)
            assert x.shape[1] == round(m_float) ** self.k,\
                f'Dimension 1 of matrix X should equal to an integer to the power of {self.k}'
            assert x.shape[2] == self.in_features, 'Dimension 2 of matrix X should equal to in_features'

            x = x.view([x.shape[0]] + [round(m_float)] * self.k + [self.in_features])  # shape (n, m, m, in_features)
        else:  # x is of shape (n, m, m, in_features)
            assert len(list(x.size())) == 2 + self.k, 'Matrix X is not a (k + 2)-d tensor'
            assert (torch.Tensor(list(x.shape[1:-1])) == x.shape[1]).sum() == self.k, \
                'Matrix X dimensions 1 to -2 have to be the same'
            assert x.shape[3] == self.in_features, 'Dimension 3 of matrix X should equal to in_features'

        # dimensions
        m = x.shape[1]
        dimensions_to_forward = (x.shape[0], m ** self.k, x.shape[3])

        # creating x_t
        x_t = x.transpose(1, 2).contiguous().view(dimensions_to_forward)  # shape (n, m ** 2, in_features)

        # creating tr_x
        all_traces = torch.diagonal(x, dim1=1, dim2=2).sum(dim=-1)  # shape (n, in_features)
        eye_matrix = torch.eye(m, device=x.get_device() if x.is_cuda else torch.device('cpu'))  # shape (m, m)
        trace_x = (all_traces[:, None, None, :] * eye_matrix[None, :, :, None]).contiguous().view(*x_t.shape)
        # shape (n, m ** 2, in_features)

        # apply a weighted sum of the transformations (the linear layers)
        model_output = self.alpha(x.view(dimensions_to_forward)) + \
                       self.beta(x_t) + self.gamma(trace_x)  # shape (n, m ** 2, out_features)
        return model_output.view(*starting_x_shape, self.out_features)


class CustomLinear(Module):
    """
        The current overall custom linear block

        :param in_features: int
                number of input features
        :param k: int
                the power to which we raise the feature dimension
        :param out_features: int
                number of output features
    """

    def __init__(self, in_features: int, k: int, out_features: int):
        super(CustomLinear, self).__init__()

        self.in_features = in_features
        self.k = k
        self.out_features = out_features

        if k == 2:
            self.parent = LinearModelK2(in_features=in_features, k=k, out_features=out_features)
        else:
            self.parent = Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x):
        return self.parent.forward(x)
