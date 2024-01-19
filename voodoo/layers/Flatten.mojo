from voodoo import Tensor, shape
from .BaseLayer import BaseLayer


struct Flatten[](BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    @always_inline
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        return x.flatten()
