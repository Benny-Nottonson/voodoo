from tensor import TensorShape

from voodoo.core import Tensor, NoneInitializer, NoneConstraint


struct Flatten[]():
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[
        TensorShape(x.shape[0], x.shape.num_elements() // x.shape[0]),
        NoneInitializer,
        NoneConstraint,
        False,
        False,
    ]:
        let res = x.flatten()
        return res
