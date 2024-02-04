from voodoo import Tensor, get_activation_code
from tensor import TensorShape
from .BaseLayer import BaseLayer
from ..initializers import Initializer, GlorotUniform, Zeroes
from ..constraints import Constraint, NoneConstraint


struct Dense[
    in_neurons: Int,
    out_neurons: Int,
    activation: String = "none",
    use_bias: Bool = True,
    weight_initializer: Initializer = GlorotUniform,
    weight_initializer_arg0: Float64 = 0.0,
    weight_initializer_arg1: Float64 = 0.0,
    weight_constraint: Constraint = NoneConstraint,
    weight_constraint_arg0: Float64 = 0.0,
    weight_constraint_arg1: Float64 = 0.0,
    bias_initializer: Initializer = Zeroes,
    bias_initializer_arg0: Float64 = 0.0,
    bias_initializer_arg1: Float64 = 0.0,
    bias_constraint: Constraint = NoneConstraint,
    bias_constraint_arg0: Float64 = 0.0,
    bias_constraint_arg1: Float64 = 0.0,
](BaseLayer):
    var W: Tensor[TensorShape(in_neurons, out_neurons)]
    var bias: Tensor[TensorShape(out_neurons)]

    fn __init__(
        inout self,
    ) raises:
        self.W = (
            Tensor[TensorShape(in_neurons, out_neurons)]()
            .initialize[
                weight_initializer, weight_initializer_arg0, weight_initializer_arg1
            ]()
            .constrain[
                weight_constraint, weight_constraint_arg0, weight_constraint_arg1
            ]()
            .requires_grad()
        )

        @parameter
        if self.use_bias:
            self.bias = (
                Tensor[TensorShape(out_neurons)]()
                .initialize[
                    bias_initializer, bias_initializer_arg0, bias_initializer_arg1
                ]()
                .constrain[
                    bias_constraint, bias_constraint_arg0, bias_constraint_arg1
                ]()
                .requires_grad()
            )
        else:
            self.bias = Tensor[TensorShape(out_neurons)]()

    @always_inline("nodebug")
    fn forward(self, x: Tensor) raises -> Tensor[x.shape, False, False]:
        var computed = x @ self.W

        @parameter
        if self.use_bias:
            computed = computed + self.bias

        @parameter
        if self.activation != "none":
            return computed.compute_activation[get_activation_code[activation]()]()

        return computed
