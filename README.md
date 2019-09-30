# ProxGradPytorch
ProxGradPytorch is a pytorch implementation of many of the proximal gradient algorithms from [Parikh and Boyd (2014)](https://web.stanford.edu/~boyd/papers/prox_algs.html). In particular, many of these algorithms are useful for Auto-Sizing Neural Networks [(Murray and Chiang 2015)](https://www.aclweb.org/anthology/D15-1107).

If you use this toolkit, we would appreciate it if you could cite:

    @inproceedings{murray19autosizing,
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
pg.l21(model.linear1.weight, model.linear1.bias, reg=0.1)
```

This will apply a group regularizer over each row. Assuming that the row is the input to a non-linearity where f(0) = 0 (and is all of the inputs to a neuron), then this will auto-size that layer. There are many other regularizers implemented as well that are not just for auto-sizing (for instance L_infinity, L_2, etc.).


