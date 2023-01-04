import numpy as np
import torch
from torch import nn
import torchinfo
import tensorflow as tf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.output(x)
        return x


def accuracy_fn(y_pred, y_true):
    y_pred_tags = y_pred.argmax(dim=1)
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc*100


def train(model, train_data, train_labels, test_data, test_labels, epochs=1000, BATCH_SIZE=1000):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_name = '02_pytorch_nn.pth'
    histories = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        for batch_ind in range(len(train_data)//BATCH_SIZE):
            data, target = train_data[batch_ind*BATCH_SIZE:(
                batch_ind+1)*BATCH_SIZE], train_labels[batch_ind*BATCH_SIZE:(batch_ind+1)*BATCH_SIZE]
            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            train_loss = loss_fn(pred, target)
            train_accuracy = accuracy_fn(pred, target)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        model.eval()
        for batch_ind in range(len(test_data)//BATCH_SIZE):
            data, target = test_data[batch_ind*BATCH_SIZE:(
                batch_ind+1)*BATCH_SIZE], test_labels[batch_ind*BATCH_SIZE:(batch_ind+1)*BATCH_SIZE]
            data = data.to(device)
            target = target.to(device)

            with torch.inference_mode():
                pred = model(data)
                test_loss = loss_fn(pred, target)
                test_accuracy = accuracy_fn(pred, target)
                histories['loss'].append(test_loss.item())
                histories['accuracy'].append(test_accuracy.item())

        if epoch % 100 == 0:
            print()
            print(f'Epoch: {epoch:02d}')
            print(
                f'Loss: {train_loss:.8f},Test Loss: {test_loss:.8f}')
            print(
                f'Accuracy: {train_accuracy:.8f}%,Test Accuracy: {test_accuracy:.8f}%')
            print(
                f'Loss: {np.mean(histories["loss"]):.8f}, Accuracy: {np.mean(histories["accuracy"])}')

        if epoch % 250 == 0:
            print(f'saving model...')
            torch.save(model.state_dict(), 'model/'+model_name)
    return model


if __name__ == "__main__":
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()

    train_data = torch.tensor(train_data) / 255.0
    test_data = torch.tensor(test_data) / 255.0

    train_data = train_data.reshape(-1, 784)
    test_data = test_data.reshape(-1, 784)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    model = NeuralNetwork(input_size=784, hidden_size=128,
                          output_size=10).to(device)
    # print(model.state_dict())
    print(model)
    torchinfo.summary(model, input_size=(1000, 784))

    train(model, train_data, train_labels, test_data, test_labels)
