from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Dropout[
    dropout_rate: Float32 = 0.5,
    noise_shape: DynamicVector[Int] = DynamicVector[Int](),
    # TODO: add noise shape functionality
](BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.dropout[dropout_rate, noise_shape]()
