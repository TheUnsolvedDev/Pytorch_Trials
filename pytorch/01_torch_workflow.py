import torch
import numpy
from torch import nn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.weights, x) + self.bias


class LinearRegressionV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


if __name__ == '__main__':
    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02

    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight*X + bias

    train_split = int(0.8*len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    model_name = '01_pytorch_workflow.pth'

    torch.manual_seed(42)
    model_0 = LinearRegressionV2().to(device)
    print(list(model_0.parameters()))

    # training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    epochs = 5000
    for epoch in range(1, epochs + 1):
        model_0.train()
        y_pred = model_0(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_0.eval()
        with torch.inference_mode():
            test_pred = model_0(X_test)
            test_loss = loss_fn(test_pred, y_test)

        if epoch % 100 == 0:
            print(
                f'Epoch: {epoch:02d}, Loss: {loss:.8f},Test Loss: {test_loss:.8f}')
            print(f'saving model...')
            torch.save(model_0.state_dict(), 'model/'+model_name)

    print(list(model_0.parameters()))

    model = LinearRegressionV2()
    model.load_state_dict(torch.load('model/'+model_name))
    print(model.state_dict())
