from voodoo import Tensor, get_activation_code
from voodoo.initializers import NoneInitializer
from voodoo.constraints import NoneConstraint


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
        return x.compute_activation[get_activation_code[activation]()]()
