import torch

class LinearNorm(torch.nn.Module):
    """
    Creates a linear layer.
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        """
        The linear layer with in dimension = in_dim  and out dimension = out_dim
        Weights are initialised using the uniform Xavier function
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """
    Creates a convolutional layer (with normalisation ??)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        """
        The convolutional layer with in dimension = in_channels and out dimension = out_channels
        Kernel size is default 1.
        Weights are initialised using the Uniform Xavier function
        """
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal