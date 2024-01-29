from voodoo import Tensor
from tensor import TensorShape
from .BaseLayer import BaseLayer


struct LeakyReLu[
    in_neurons: Int,
    out_neurons: Int,
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "zeros",
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
            Tensor(TensorShape(in_neurons, out_neurons))
            .initialize[weight_initializer, weight_mean, weight_std]()
            .requires_grad()
        )

        @parameter
        if self.use_bias:
            self.bias = (
                Tensor(TensorShape(out_neurons))
                .initialize[bias_initializer, bias_mean, bias_std]()
                .requires_grad()
            )
        else:
            self.bias = Tensor(TensorShape(out_neurons)).initialize["zeros", 0.0]()

    @always_inline("nodebug")
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        var computed = x @ self.W

        @parameter
        if self.use_bias:
            computed = computed + self.bias

        return (computed).compute_activation["lrelu", self.alpha]()
