# PyTorch Fluent Models

A small package that provides a fluent interface for creating pytorch models.

## Warning

This package is not complete yet and has not been pushed to pypi yet.

## Summary

A fluent interface is roughly one where you chain method calls. Read more about
fluent interfaces [here](https://en.wikipedia.org/wiki/Fluent_interface).

This library allows for dense layers, convolution layers, max pooling,
and nonlinearities or other operators (i.e. normalization). This calculates
the new shape after each layer, meaning you do not have to redundantly
specify features.

Consider the following pure PyTorch code:

```py
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.Linear(128, 10)
)
```

The input to the second layer (128) must always match the output of the first
layer. This redundancy is very small but can be improved. The issue becomes
even more apparent when you consider convolution layers.

Furthermore, the official PyTorch library does not include some common glue
code for extensive sequential blocks. One possible reason for this is that
Fluent API's are unlikely to be as exhaustive as conventional API's so
one will often have to fall back on the more verbose module definition anyway.

Finally, this has the extremely versatile `then` and `then_with` which
work for transposed convolution layers and unpooling while still avoiding
redundant layer sizes or channel numbers.

## API Reference

https://tjstretchalot.github.io/torchluent/

## Usage

Create an instance of `torchluent.FluentModule` with the shape of your input.
There are a few meta functions on FluentModule, such as `.verbose()` which
will print how the shape changes through progressive calls. For layers which
change the number of features one can call `.transform` in the generic sense
or use one of the provided functions such as `.dense` which will calculate the
new number of features. For layers which do not change the shape of the data,
rather than including a function for each one you may use `.operator` which
accepts the name of the attribute in `torch.nn` as well as an arguments or
keyword arguments.

## Installation

```pip install torchluent```

## Examples

```py
from torchluent import FluentModule

print('Network:')
net = (
    FluentModule((1, 28, 28))
    .verbose()
    .conv2d(32, kernel_size=5)
    .maxpool2d(kernel_size=3)
    .operator('LeakyReLU', negative_slope=0.05)
    .flatten()
    .dense(128)
    .operator('ReLU')
    .dense(10)
    .operator('ReLU')
    .build()
)

print(net)
```

Produces:

```
Network:
  (1, 28, 28)
  Conv2d -> (32, 24, 24)
  MaxPool2d -> (32, 8, 8)
  LeakyReLU
  Reshape -> (2048,)
  Linear -> (128,)
  ReLU
  Linear -> (10,)
  ReLU

Sequential(
  (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
  (1): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
  (2): LeakyReLU(negative_slope=0.05)
  (3): Reshape(2048)
  (4): Linear(in_features=2048, out_features=128, bias=True)
  (5): ReLU()
  (6): Linear(in_features=128, out_features=10, bias=True)
  (7): ReLU()
)
```

## Wrapping and Unwrapping

One concept which is not in PyTorch by default is a way to consider the hidden
state of an arbitrary network in an abstract way. The idea is
basically that it is often nice if a module returns an array in addition to
the transformed output, where each element in the returned array is a snapshot
of the input as it propagated through the network.

The following is a contrived example that illustrates what such a module might
look like:

```py
import torch.nn as nn

class HiddenStateModule(nn.Module):
    def forward(self, x):
        result = []
        result.append(x) # initial state always there
        x = x ** 2
        result.append(x) # where relevant
        x = x * 3 + 2
        x = torch.relu(x)
        result.append(x)
        return x, result
```

This module means to expose this concept without having to modify the
underlying transformations (i.e. `nn.Linear`) nor be forced to fallback on
creating a custom Module just for this extremely common situation.

However, another problem that arises with this type of module is that this
result will break much of your codebase if it expects a single output. This
is most problematic when combined with some abstract training paradigm such as
PyTorch Ignite. Luckily, it's very easy to just drop the second output from
such a module, as if by the following

```py
import torch.nn as nn

class StrippedStateModule(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x)[0]
```

By including the array in the main implementation and then using such an
"unwrapping" module you can get the best of both worlds. For training and
generic usage which does not need the hidden state, use the stripped version.
For analysis which desires the hidden state, use the pre-stripped version.

With this context in mind, the following code snippet will produce both the
wrapped and unwrapped versions of the network:

```py
from torchluent import FluentModule

print('Network:')
net, stripped_net = (
    FluentModule((28*28,))
    .verbose()
    .wrap(with_input=True) # create array and initialize with input
    .dense(128)
    .operator('ReLU')
    .save_state() # pushes to the array
    .dense(128)
    .operator('ReLU')
    .save_state()
    .dense(10)
    .operator('ReLU')
    .save_state()
    .build(with_stripped=True)
)
print()
print(net)
```

Produces

```
Network:
  (784,)
  Linear -> (128,)
  ReLU
  Linear -> (128,)
  ReLU
  Linear -> (10,)
  ReLU

Sequential(
  (0): InitListModule(include_first=True)
  (1): WrapModule(
    (child): Linear(in_features=784, out_features=128, bias=True)
  )
  (2): WrapModule(
    (child): ReLU()
  )
  (3): SaveStateModule()
  (4): WrapModule(
    (child): Linear(in_features=128, out_features=128, bias=True)
  )
  (5): WrapModule(
    (child): ReLU()
  )
  (6): SaveStateModule()
  (7): WrapModule(
    (child): Linear(in_features=128, out_features=10, bias=True)
  )
  (8): WrapModule(
    (child): ReLU()
  )
  (9): SaveStateModule()
)
```

## Limitations

For non-trivial networks there will likely be significant usage of the `then`
and `then_with` functions which aren't quite as nice as the examples shown
above, but I believe they are still a significant improvement.
