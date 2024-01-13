from voodoo import Tensor, shape
from .BaseLayer import BaseLayer


struct Dropout[
    dropout_rate: Float32 = 0.5,
    noise_shape: DynamicVector[Int] = DynamicVector[Int](),
    # TODO: add noise shape functionality
](BaseLayer):
    @always_inline
    fn __init__(
        inout self,
    ) raises:
        ...

    @always_inline
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        return x.dropout[dropout_rate, noise_shape]()
