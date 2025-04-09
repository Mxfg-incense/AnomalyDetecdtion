import torch


class LSTMEncoder(torch.nn.Module):
    """
    Encoder of a time series using a LSTM, ccomputing a linear transformation
    of the output of an LSTM

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    Only works for one-dimensional time series.
    """
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1, hidden_size=256, num_layers=2
        )
        self.linear = torch.nn.Linear(256, 160)

    def forward(self, x):
        return self.linear(self.lstm(x.permute(2, 0, 1))[0][-1])
