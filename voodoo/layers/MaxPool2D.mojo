from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct MaxPool2D[
    pool_size: Int = 2,
](BaseLayer):
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.max_pool_2d(self.pool_size, self.pool_size)
