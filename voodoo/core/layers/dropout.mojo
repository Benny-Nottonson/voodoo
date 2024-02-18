from tensor import TensorShape

from voodoo.core import Tensor, NoneInitializer, NoneConstraint


struct Dropout[
    dropout_rate: Float32 = 0.5, noise_shape: TensorShape = TensorShape(0)
]():
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[x.shape, NoneInitializer, NoneConstraint, False, False]:
        let res = x.dropout[dropout_rate, noise_shape]()
        return res
