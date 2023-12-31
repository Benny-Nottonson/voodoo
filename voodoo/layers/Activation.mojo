from voodoo import Tensor, get_activation_code
from .BaseLayer import BaseLayer


struct Activation[
    in_neurons: Int,
    out_neurons: Int,
    activation: String = "none",
](BaseLayer):
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.compute_activation[get_activation_code[activation]()]()
