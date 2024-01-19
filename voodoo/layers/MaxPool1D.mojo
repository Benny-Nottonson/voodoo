from voodoo import Tensor, shape
from .BaseLayer import BaseLayer


struct MaxPool1D[
    kernel_width: Int,
    stride: Int = 1,
    padding: Int = 0,
](BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        let res = x.maxpool_1d(
            kernel_width,
            stride,
            padding,
        )
        return res
