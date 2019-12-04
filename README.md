# ProxGradPyTorch
ProxGradPyTorch is a PyTorch implementation of many of the proximal gradient algorithms from [Parikh and Boyd (2014)](https://web.stanford.edu/~boyd/papers/prox_algs.html). In particular, many of these algorithms are useful for Auto-Sizing Neural Networks [(Murray and Chiang 2015)](https://www.aclweb.org/anthology/D15-1107).

If you use this toolkit, we would appreciate it if you could cite:

    @inproceedings{murray2019autosizing,
        author={Murray, Kenton and Kinnison, Jeffery and Nguyen, Toan Q. and Scheirer, Walter and Chiang, David},
        title={Auto-Sizing the Transformer Network: Improving Speed, Efficiency, and Performance for Low-Resource Machine Translation},
        year=2019,
        booktitle={Proceedings of the Third Workshop on Neural Generation and Translation},
    }

## Installation
A PyPI release is on the way, but for now, to build from source, simply clone this repository. Currently, there is a dependency on pytorch >=0.4.1 On Linux, it's easiest to add the repo to your shared library path:

```
export LD_LIBRARY_PATH="[install_dir]/ProxGradPytorch/prox-grad-pytorch:$LD_LIBRARY_PATH"
```

In the headers for any file that you want to use ProxGradPytorch, add the following line:

```
import proximalGradient as pg
```

## Running

Proximal Gradient Algorithms make use of a two-step process. First, normal backpropogation is run on your network:

```
# Zero gradients, perform a backward pass, and update the weights.
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This is just a standard pytorch update. Second, you run the proximal gradient algorithm. Many of these algorithms have a closed form solution and do not rely on stored gradients. For instance, to apply L2,1 regularization to a tensor named model.linear1, you run the following code:

```
pg.l21(model.linear1.weight, model.linear1.bias, reg=0.005)
```

This will apply a group regularizer over each row. Assuming that the row is the input to a non-linearity where f(0) = 0 (and is all of the inputs to a neuron), then this will auto-size that layer. There are many other regularizers implemented as well that are not just for auto-sizing (for instance L_infinity, L_2, etc.).

## Auto-Sizing

[Murray et al. (2019)](https://www.aclweb.org/anthology/D19-5634/), make use of these algorithms for auto-sizing. Auto-sizing is a method for deleting the number of neurons in a network subject to a few assumptions. At a basic level, if all the weights of a neuron are 0.0, it does not matter what the input to that neuron is -- everything will be 0.0. If the non-linearity maps f(0) to 0, such as tanh or ReLU, the output is 0.0 and it is as if the neuron does not exist. Auto-sizing relies on the use of sparse group regularizers in order to drive these weights to 0. As sparse regularizers are often non-differentiable, the authors rely on the proximal gradient methods in this toolkit. For a more complete description of auto-sizing, see either that paper or [Murray and Chiang (2015)](https://www.aclweb.org/anthology/D15-1107).

As an example of auto-sizing, let's look at simple xor example build with a two layer network (also available in the examples):

```
import torch
from torch.autograd import Variable

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

# Neurons Left (H)
print("H (model.linear1.weight):", (model.linear1.weight.nonzero()[:,0]).unique().size(0))

print("Final results:")
for input, target in zip(inputs, targets):
    output = model(input)
    print("Input:", input, "Target:", target, "Predicted:", output)
``` 

To auto-size this network, which will reduce the dimension of H, only requires two lines of code. First, we import this toolkit:

```
import proximalGradient as pg
```

Then, we simply apply the proximal gradient step after optimizer.step():

```
pg.linf1(model.linear1.weight, model.linear1.bias, reg=0.1)
```

So, the final code is:


```
import torch
from torch.autograd import Variable
import proximalGradient as pg


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
print("H remaining (model.linear1.weight):", (model.linear1.weight.nonzero()[:,0]).unique().size(0))

print("Final results:")
for input, target in zip(inputs, targets):
    output = model(input)
    print("Input:", input, "Target:", target, "Predicted:", output)
``` 

Though random initializations vary, frequently there are around 15 of the 100 neurons (H) left.
