from torch import nn

class LinearRegression(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.linear = nn.Linear(input_shape, output_shape)
    def forward(self, X):
        return self.linear(X)