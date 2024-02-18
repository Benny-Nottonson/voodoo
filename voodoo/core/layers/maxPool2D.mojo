from tensor import TensorShape

from voodoo.core import Tensor, NoneInitializer, NoneConstraint


struct MaxPool2D[
    kernel_width: Int,
    kernel_height: Int,
    stride: Int = 1,
    padding: Int = 0,
]():
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[
        TensorShape(
            x.shape[0],
            (x.shape[1] - kernel_width + 2 * padding) // stride + 1,
            (x.shape[2] - kernel_height + 2 * padding) // stride + 1,
            x.shape[3],
        ),
        NoneInitializer,
        NoneConstraint,
        False,
        False,
    ]:
        let res = x.maxpool_2d[
            TensorShape(
                x.shape[0],
                (x.shape[1] - kernel_width + 2 * padding) // stride + 1,
                (x.shape[2] - kernel_height + 2 * padding) // stride + 1,
                x.shape[3],
            )
        ](
            StaticIntTuple[2](kernel_width, kernel_height),
            stride,
            padding,
        )
        return res
