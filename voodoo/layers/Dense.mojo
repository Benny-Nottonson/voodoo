from voodoo import Tensor, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Dense[
    in_neurons: Int,
    out_neurons: Int,
    activation: String = "none",
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "zeros",
    weight_mean: Float32 = 0.0,
    weight_std: Float32 = 0.05,
    bias_mean: Float32 = 0.0,
    bias_std: Float32 = 0.05,
    # TODO: Add regularizers and constraints
](BaseLayer):
    var W: Tensor
    var bias: Tensor

    # TODO: Might need .requires_grad() for weights and bias
    fn __init__(
        inout self,
    ) raises:
        self.W = (
            Tensor(shape(in_neurons, out_neurons))
            .initialize[weight_initializer, weight_mean, weight_std]()
            .requires_grad()
        )

        @parameter
        if self.use_bias:
            self.bias = (
                Tensor(shape(out_neurons))
                .initialize[bias_initializer, bias_mean, bias_std]()
                .requires_grad()
            )
        else:
            self.bias = Tensor(shape(out_neurons))

    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        let computed = x @ self.W + (self.bias * Float32(self.use_bias))

        @parameter
        if self.activation != "none":
            return computed.compute_activation[get_activation_code[activation]()]()

        return computed
