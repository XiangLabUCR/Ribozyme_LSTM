import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout_rate=0.2
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            hidden_size,
            hidden_size * 2,
            batch_first=True
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            hidden_size * 2 * 7,
            1000
        )

        self.fc2 = nn.Linear(
            1000,
            output_size
        )


    def forward(self, x):

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x