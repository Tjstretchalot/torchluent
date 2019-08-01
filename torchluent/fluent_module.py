"""Contains the FluentModule class"""

import pytypeutils as tus
import typing
import torch
import torch.nn as nn
import operator
from functools import reduce

class Reshape(nn.Module):
    """Reshapes the input to match the given shape, using view. This preserves
    the first dimension which is assumed to be the batch dimension.

    :Example:

    .. code-block:: python

        import torchluent
        import torch

        a = torchluent.Reshape(28*28)

        data = torch.randn(5, 28, 28)
        reshaped = a(data)
        print(reshaped.shape) # torch.Size[5, 784]


    :ivar tuple[int] shape: the new shape for the input
    """
    def __init__(self, *args):
        super().__init__()
        tus.check(args=(args, tuple))
        tus.check_listlike(args=(args, int, (1, None)))

        self.shape = args

    def forward(self, x):
        """Changes the view of x to the desired shape"""
        real_new_shape = [x.shape[0]]
        real_new_shape.extend(self.shape)
        return x.view(real_new_shape)

    def extra_repr(self):
        return ', '.join(str(i) for i in self.shape)

class Transpose(nn.Module):
    """Transposes two dimensions. Does not effect the batch
    dimension.

    :Example:

    .. code-block:: python

        import torchluent
        import torch

        transposer = torchluent.Transpose(0, 1)

        data = torch.randn(5, 100, 50)
        newdata = transposer(data)
        print(newdata.shape) # torch.Size[5, 50, 100]

    :ivar int dim1: the first dimension to transpose
    :ivar int dim2: the second dimension to transpose
    """
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        tus.check(dim1=(dim1, int), dim2=(dim2, int))
        if dim1 < 0:
            raise ValueError(f'dim1={dim1} must be nonnegative')
        if dim2 < 0:
            raise ValueError(f'dim2={dim2} must be nonnegative')
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.tranpose(self.dim1 + 1, self.dim2 + 1)

    def extra_repr(self):
        return f'{self.dim1}, {self.dim2}'

class InitListModule(nn.Module):
    """Initializes a list of states, optionally with the state its
    passed in.

    :ivar bool include_first: True to include x in the list, False to make
        an empty list.
    """
    def __init__(self, include_first: bool):
        super().__init__()
        tus.check(include_first=(include_first, bool))
        self.include_first = include_first

    def forward(self, x):
        return (x, [x]) if self.include_first else (x, [])

    def extra_repr(self):
        return f'include_first={self.include_first}'

class WrapModule(nn.Module):
    """Wraps a module which is expecting just x, passing the list through it

    :ivar nn.Module child: the wrapped module
    """
    def __init__(self, child: nn.Module):
        super().__init__()
        tus.check(child=(child, nn.Module))
        self.child = child

    def forward(self, x_and_arr):
        return self.child(x_and_arr[0]), x_and_arr[1]

class SaveStateModule(nn.Module):
    """Stores the state into the array.
    """
    def forward(self, x_and_arr):
        x_and_arr[1].append(x_and_arr[0])
        return x_and_arr

class StrippingModule(nn.Module):
    """Strips the array from the output of the child

    :ivar nn.Module child: the child who we are stripping
    """
    def __init__(self, child: nn.Module):
        super().__init__()
        tus.check(child=(child, nn.Module))
        self.child = child

    def forward(self, x):
        return self.child(x)[0]

class FluentModule:
    """
    This constructs torch modules in a fluent-style interface.

    :Example:

    .. code-block:: python

        from torchluent import FluentModule
        net = (
            FluentModule(28*28)
            .dense(128)
            .operator('ReLU')
            .dense(10)
            .operator('ReLU')
            .build()
        )

    .. note::

        This modules shape and all shape arguments are in practice prefixed by
        a batch dimension. The batch dimension is not altered by any of these
        calls, including reshaping, unless otherwise specified.

    :ivar list[nn.Module] sequence: the actual sequence of modules that
        we have constructed so far.

    :ivar tuple[int] shape: the current feature shape

    :ivar bool is_verbose: if we are currently outputting each function call and
        the corresponding effects

    :ivar bool wrapped: if we are currently storing a list of hidden states
    """
    def __init__(self, shape: typing.Tuple[int], assume_wrapped: bool = False):
        tus.check(shape=(shape, (list, tuple)),
                  assume_wrapped=(assume_wrapped, bool))
        tus.check_listlike(shape=(shape, int, (1, None)))
        for features in shape:
            if features <= 0:
                raise ValueError(f'shape={shape} must be positive')

        self.shape = tuple(shape)
        self.sequence = []
        self.is_verbose = False
        self.wrapped = assume_wrapped

    def verbose(self) -> 'FluentModule':
        """Turns on verbose mode, which cases this to output every function
        call and the resulting shape.

        :returns: self
        :rtype: FluentModule
        """
        self.is_verbose = True
        print(f'  {self.shape}')
        return self

    def silent(self) -> 'FluentModule':
        """Disables verbose mode

        :returns: self
        :rtype: FluentModule
        """
        self.is_verbose = False
        return self

    def wrap(self, with_input: bool = False) -> 'FluentModule':
        """Changes the output to the form (x, arr) where an arr is a list of
        states stored in locations specified with save_state()

        :param with_input: if True we immediately save_state()
        :type with_input: bool
        :returns: self
        :rtype: FluentModule
        """
        tus.check(with_input=(with_input, bool))
        if self.wrapped:
            raise ValueError('already wrapped')
        self.wrapped = True
        self.sequence.append(InitListModule(with_input))
        return self

    def _wrap(self, mod):
        return WrapModule(mod) if self.wrapped else mod

    def save_state(self):
        """Stores the current state into the list for the result. Requires that
        wrap() has already been called.

        :returns: self
        :rtype: FluentModule
        """
        if not self.wrapped:
            raise ValueError('cannot save_state() without wrap()')
        self.sequence.append(SaveStateModule())
        return self

    def dense(self, out_features: int, bias: bool = True) -> 'FluentModule':
        """A dense layer, also known as a linear layer or a fully connected
        layer. A dense layer requires that this already be in flattened
        form, i.e., len(self.shape) == 1.

        :param out_features: the number of neurons to project to
        :param bias: determines if a bias (additive) term is applied to each
            of the output features
        :type out_features: int
        :type bias: bool
        :returns: self
        :rtype: FluentModule
        """
        tus.check(out_features=(out_features, int),
                  bias=(bias, bool))
        if out_features <= 0:
            raise ValueError(f'out_features={out_features} must be positive')
        if len(self.shape) != 1:
            raise ValueError(
                f'cannot perform operation {self.shape} -> dense -> '
                + f'{out_features} (current shape is not flat). consider '
                + 'calling flatten() first')
        self.sequence.append(self._wrap(nn.Linear(self.shape[0], out_features, bias)))
        self.shape = (out_features,)
        if self.is_verbose:
            print(f'  Linear -> {self.shape}')
        return self

    def reshape(self, shape: typing.Tuple[int]) -> 'FluentModule':
        """Reshapes the data to the specified shape. Must correspond to the
        same total number of features.

        .. note::

            The batch dimension is preserved.

        :param shape: the new shape for the data
        :type shape: tuple[int]
        :returns: self
        :rtype: FluentModule
        """
        tus.check(shape=(shape, (list, tuple)))
        tus.check_listlike(shape=(shape, int, (1, None)))
        for features in shape:
            if features <= 0:
                raise ValueError(f'shape={shape} must be positive')

        old_num_features = reduce(operator.mul, self.shape)
        new_num_features = reduce(operator.mul, shape)
        if old_num_features != new_num_features:
            raise ValueError(
                f'cannot view {self.shape} as {shape}: expected '
                + f'{old_num_features} but got {new_num_features}')

        self.sequence.append(self._wrap(Reshape(*shape)))
        self.shape = tuple(shape)
        if self.is_verbose:
            print(f'  Reshape -> {self.shape}')
        return self

    def flatten(self) -> 'FluentModule':
        """Reshapes this such that the data has only one dimension.

        .. note::

            The batch dimension is preserved.

        :returns: self
        :rtype: FluentModule
        """
        return self.reshape((reduce(operator.mul, self.shape),))

    def transpose(self, dim1: int, dim2: int) -> 'FluentModule':
        """Transposes the two specified dimensions, where dimension 0 is the
        first dimension after the batch dimension (i.e., really index 0
        in self.shape).

        :Example:

            from torchluent import FluentModule
            import torch

            net = FluentModule((1, 12, 24)).transpose(0, 2).build()
            inp = torch.randn((5, 1, 12, 24))
            out = net(inp)
            print(out.shape) # torch.Size[5, 12, 24, 1]

        :returns: self
        :rtype: FluentModule
        """
        tus.check(dim1=(dim1, int), dim2=(dim2, int))
        if not 0 <= dim1 < len(self.shape) or not 0 <= dim2 < len(self.shape):
            raise ValueError(f'cannot transpose {dim1} and {dim2} for '
                             + f'shape {self.shape}')

        self.sequence.append(self._wrap(Transpose(dim1, dim2)))
        newshape = list(self.shape)
        tmp = newshape[dim1]
        newshape[dim1] = newshape[dim2]
        newshape[dim2] = tmp
        self.shape = list(newshape)
        if self.is_verbose:
            print(f'  Transpose[{dim1}, {dim2}] -> {self.shape}')
        return self

    def operator(self, oper, *args, **kwargs) -> 'FluentModule':
        """An operator is some operation which does not change the shape of the
        data. The operator may be specified as a string, in which it should be
        a module in torch.nn, or it may be the module itself which has not yet
        be initialized (i.e. 'ReLU' or nn.ReLU but not nn.ReLU())

        :Example:

        .. code-block:: python

            from torchluent import FluentModule
            net = (
                FluentModule(28*28)
                .dense(10)
                .operator('LeakyReLU', negative_slope=0.05)
                .build()
            )

        :param oper: the name of the operator or a callable which returns one
        :param args: passed to the operator
        :param kwargs: passed to the operator
        :returns: self
        :rtype: FluentModule
        """
        if isinstance(oper, str):
            if not hasattr(nn, oper):
                raise ValueError(f'torch.nn has no attribute {oper}')
            oper = getattr(nn, oper)

        mod = oper(*args, **kwargs)
        if self.is_verbose:
            print(f'  {type(mod).__name__}')
        tus.check(**{'oper(*args, **kwargs)': (mod, nn.Module)})
        self.sequence.append(self._wrap(mod))
        return self

    def then(self, module, *args, **kwargs) -> 'FluentModule':
        """Applies a generic torch module transformation. To determine the
        output shape, this just runs some data through the module. If the
        module is a string then it it is assumed to be the name of an
        attribute in torch.nn, and it is initialized with the specified
        arguments.

        :param module: the module that should modify the data
        :rtype module: union[nn.Module, str, type]
        :returns: self
        :rtype: FluentModule
        """
        if isinstance(module, str):
            if not hasattr(nn, module):
                raise ValueError(f'torch.nn has no attribute {module}')
            module = getattr(nn, module)
        if not isinstance(module, nn.Module):
            module = module(*args, **kwargs)

        module.eval()
        with torch.no_grad():
            data = torch.randn(self.shape).unsqueeze(0)
            output = module(data)
            tus.check(output=(output, torch.Tensor))
            if output.shape[0] != 1:
                raise ValueError('module killed batch dimension; '
                                 + f'output shape: {output.shape}')

            new_shape = list(output.shape)
            new_shape.pop(0)

        self.sequence.append(self._wrap(module))
        self.shape = tuple(new_shape)
        if self.is_verbose:
            print(f'  {type(module).__name__} -> {self.shape}')

        return self

    def then_with(self, dims, mod, *args, **kwargs) -> 'FluentModule':
        """This applies the given nn.Module or string for an attribute in nn
        with the given dimensions passed as inputs. dims should either be a
        single number, which is treated like a tuple of a single element, or
        a tuple of numbers, which is treated as if each element is (i, num)
        where i is the index, or a tuple of (arg_index, num).

        Our current shape is injected into args such that for each pair
        (arg_index, num) in dims, args[arg_index] = self.shape[num]. This
        allows for an extremely generic interface for modules which do not have
        a dedicated function for them.

        :Example:

        .. code-block:: python

            from torchluent import FluentModule

            net = (
                FluentModule((1, 7, 7))
                .verbose()
                .then_with(0, 'ConvTranspose2d', 16,
                           kernel_size=2, stride=2, padding=2)
                .operator('LeakyReLU')
                .then_with(0, 'ConvTranspose2d', 32,
                           kernel_size=2, stride=2, padding=2)
                .operator('LeakyReLU')
                .then_with(0, 'ConvTranspose2d', 1,
                           kernel_size=3, stride=2, padding=2)
                .operator('LeakyReLU')
                .build()
            )


        :ivar dims: one of int, tuple[int], and tuple[tuple[int, int]]. each
            element is treated as if by (arg_index, num) where num is the
            dimension in self.shape that corresponds to args[arg_index]
        :ivar mod: either a str (for an attribute in nn) or a callable which
            returns a module.

        :returns: self
        :rtype: FluentModule
        """
        if isinstance(dims, int):
            dims = (dims,)
        tus.check(dims=(dims, (list, tuple)))
        dims = list(dims)
        for i in range(len(dims)):
            if isinstance(dims[i], int):
                dims[i] = (i, dims[i])
            tus.check(**{f'dims[{i}]': (dims[i], tuple)})
            tus.check_listlike(**{f'dims[{i}]': (dims[i], int, 2)})
            if dims[i][0] < 0:
                raise ValueError(f'dims[{i}][0] = {dims[i][0]} '
                                 + 'should be nonnegative')
            if dims[i][0] >= len(args) + len(dims):
                raise ValueError(f'dims[{i}][0] = {dims[i][0]} requires more '
                                 + 'arguments than were specified')
            if not 0 <= dims[i][1] < len(self.shape):
                raise ValueError(f'dims[{i}][1]={dims[i][1]} is not valid for '
                                 + f'the current shape {self.shape}')

        if len(set(arg_index for arg_index, num in dims)) != len(dims):
            raise ValueError(f'arg_index must be unique in dims={dims}')

        if isinstance(mod, str):
            if not hasattr(nn, mod):
                raise ValueError(f'no module {mod} in torch.nn')
            mod = getattr(nn, mod)

        dims.sort(key=lambda x: x[0])
        newargs = []
        newargs.extend(args)
        for arg_index, shape_index in dims:
            newargs.insert(arg_index, self.shape[shape_index])

        return self.then(mod(*newargs, **kwargs))

    def conv1d(self, *args, **kwargs) -> 'FluentModule':
        """Applies a 1d convolution to the current data. The current shape
        should be in the form (channels, length). This accepts all the same
        arguments as nn.Conv1d exception for in_channels which it will
        calculate from the current shape.

        .. seealso::

            `torch.nn.Conv1d <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d>`_

        :returns: self
        :rtype: FluentModule
        """
        if len(self.shape) != 2:
            raise ValueError(f'cannot perform conv1d on shape {self.shape} - '
                             + 'expected shape (channels, length)')

        return self.then_with(0, 'Conv1d', *args, **kwargs)

    def conv2d(self, *args, **kwargs) -> 'FluentModule':
        """Applies a convolution to the current data. The current shape should
        be in the form (channels, height, width). This accepts all the same
        arguments as nn.Conv2d except for in_channels, which it will calculate
        from the current shape.

        .. seealso::

            `torch.nn.Conv2d <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`_

        :returns: self
        :rtype: FluentModule
        """
        if len(self.shape) != 3:
            raise ValueError(f'cannot perform conv2d on shape {self.shape} - '
                             + 'expected shape (channels, height, width)')

        return self.then_with(0, 'Conv2d', *args, **kwargs)

    def conv3d(self, *args, **kwargs) -> 'FluentModule':
        """Applies a convolution to the current data. The current shape should
        be in the form (channels, depth, height, width). This accepts all the same
        arguments as nn.Conv3d except for in_channels, which it will calculate
        from the current shape.

        .. seealso::

            `torch.nn.Conv3d <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv3d>`_

        :returns: self
        :rtype: FluentModule
        """
        if len(self.shape) != 4:
            raise ValueError(f'cannot perform conv3d on shape {self.shape} - '
                             + 'expected shape '
                             + '(channels, depth, height, width)')

        return self.then_with(0, 'Conv3d', *args, **kwargs)

    def maxpool1d(self, *args, **kwargs) -> 'FluentModule':
        """The arguments and keyword arguments are identical to MaxPool1d

        .. seealso::

            `torch.nn.MaxPool1d
            <https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool1d>`_

        :returns: self
        :rtype: FluentModule
        """
        if len(self.shape) != 2:
            raise ValueError(f'cannot perform maxpool1d on shape {self.shape} - '
                             + 'expected shape (channels, length)')
        return self.then('MaxPool1d', *args, **kwargs)

    def maxpool2d(self, *args, **kwargs) -> 'FluentModule':
        """The arguments and keyword arguments are identical to MaxPool2d

        .. seealso::

            `torch.nn.MaxPool2d
            <https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d>`_

        :returns: self
        :rtype: FluentModule
        """
        if len(self.shape) != 3:
            raise ValueError(f'cannot perform maxpool2d on shape '
                             + f'{self.shape} - expected shape '
                             + '(channels, height, width)')

        return self.then('MaxPool2d', *args, **kwargs)

    def maxpool3d(self, *args, **kwargs) -> 'FluentModule':
        """The arguments and keyword arguments are identical to MaxPool3d

        .. seealso::

            `torch.nn.MaxPool3d
            <https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool3d>`_

        :returns: self
        :rtype: FluentModule
        """
        if len(self.shape) != 4:
            raise ValueError(f'cannot perform maxpool3d on shape '
                             + f'{self.shape} - expected shape '
                             + '(channels, depth, height, width)')

        return self.then('MaxPool3d', *args, **kwargs)

    def build(self, with_stripped=False) -> nn.Module:
        """Constructs the actual torch module created through other invocations
        to this instance.

        :param with_stripped: if True, wrap() must have been called and the
            output changes to (net, stripped_net).
        :type with_stripped: bool

        :returns: a ready-to-use torch module
        :rtype: nn.Module
        """
        tus.check(with_stripped=(with_stripped, bool))
        if with_stripped and not self.wrapped:
            raise ValueError('cannot strip unless already wrapped')

        res = nn.Sequential(*self.sequence)
        if with_stripped:
            return res, StrippingModule(res)
        return res
