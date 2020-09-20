import torch
from torch.autograd import Variable
import proximal_gradient.proximalGradient as pg

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# D_in is input dimension; H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 2, 100, 1

# Inputs and Outputs for xor
inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))

# Construct model
model = TwoLayerNet(D_in, H, D_out)

# Neurons to Start (H)
print("H initially (model.linear1.weight):", (model.linear1.weight.nonzero()[:,0]).unique().size(0))

# Loss, Optimizer, and Proximal Gradient
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for t in range(5000):
    for input, target in zip(inputs, targets):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(input)

        # Compute loss
        loss = criterion(y_pred, target)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Proximal Gradient Step
        pg.linf1(model.linear1.weight, model.linear1.bias, reg=0.005)

# Neurons Left (H)
print("H (model.linear1.weight):", (model.linear1.weight.nonzero()[:,0]).unique().size(0))
print("Final results:")
for input, target in zip(inputs, targets):
    output = model(input)
    print("Input:", input, "Target:", target, "Predicted:", output)
