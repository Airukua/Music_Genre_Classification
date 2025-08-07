import torch.nn as nn

class Net5(nn.Module):
    def __init__(self, input_size=33, hidden_size=128, output_size=10, num_layers=2, dropout=0.5):
        super(Net5, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out