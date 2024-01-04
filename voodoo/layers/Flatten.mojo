from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Flatten(BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.flatten()
