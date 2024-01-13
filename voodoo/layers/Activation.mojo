from voodoo import Tensor, get_activation_code
from .BaseLayer import BaseLayer


struct Activation[
    in_neurons: Int,
    out_neurons: Int,
    activation: String = "none",
](BaseLayer):
    @always_inline
    fn __init__(
        inout self,
    ) raises:
        ...

    @always_inline
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        return x.compute_activation[get_activation_code[activation]()]()
