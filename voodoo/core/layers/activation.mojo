from voodoo.core import Tensor, NoneInitializer, NoneConstraint
from voodoo.utils import get_activation_code


struct Activation[
    in_neurons: Int,
    out_neurons: Int,
    activation: String,
]():
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[x.shape, NoneInitializer, NoneConstraint, False, False]:
        let res = x.compute_activation[get_activation_code[activation]()]()
        return res
