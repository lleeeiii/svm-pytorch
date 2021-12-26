import torch
from torch import nn

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, input):
        # Expect input to be shape (N, self.input_dim)
        output = self.linear(input) # Shape (N, 1)
        output = output.flatten() # Shape (N)
        return output 

def HingeLoss(pred, truth):
    # Expect both of shape (N)
    loss_tensor = nn.ReLU()(1-pred*truth)
    return torch.mean(loss_tensor)



