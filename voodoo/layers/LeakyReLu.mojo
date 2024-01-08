from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct LeakyReLu[
    in_neurons: Int,
    out_neurons: Int,
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_mean: Float32 = 0.0,
    weight_std: Float32 = 0.05,
    bias_mean: Float32 = 0.0,
    bias_std: Float32 = 0.05,
    alpha: Float32 = 0.2,
](BaseLayer):
    var W: Tensor
    var bias: Tensor

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
            self.bias = Tensor(shape(out_neurons)).initialize["zeros", 0.0]()

    fn forward(self, x: Tensor) raises -> Tensor:
        return (x @ self.W + (self.bias * Float32(self.use_bias))).compute_activation[
            "lrelu", self.alpha
        ]()
