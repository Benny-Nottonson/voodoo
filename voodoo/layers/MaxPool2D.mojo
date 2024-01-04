from voodoo import Tensor, conv_2d, get_activation_code, shape
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

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.max_pool_2d(self.pool_size, self.pool_size, self.stride, self.padding)
