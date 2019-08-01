"""Contains the FluentModule, which is the main interface for torchluent"""
import typing
import torch.nn as nn

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

    :ivar list[nn.Module] sequence: the actual sequence of modules that
        we have constructed so far.

    :ivar tuple[int] shape: the current feature shape

    """
    def __init__(self, shape: typing.Tuple[int]):
        self.shape = shape
        self.sequence = []

    def build(self) -> nn.Module:
        """Constructs the actual torch module created through other invocations
        to this instance.

        :returns: a ready-to-use torch module
        :rtype: nn.Module
        """
        return nn.Sequence(self.sequence)
