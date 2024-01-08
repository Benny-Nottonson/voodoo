from voodoo import Tensor, shape
from .BaseLayer import BaseLayer


struct Flatten(BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.flatten()
