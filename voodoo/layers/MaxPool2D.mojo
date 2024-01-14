from voodoo import Tensor, shape
from .BaseLayer import BaseLayer


struct MaxPool2D[
    pool_size: Int = 2,
    stride: Int = 2,
    padding: Int = 0,
](BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    @always_inline
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        return x.max_pool_2d(self.pool_size, self.pool_size, self.stride, self.padding)
