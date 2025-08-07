#Tutorial : https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
import torch.nn as nn

class Net1(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net1,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits