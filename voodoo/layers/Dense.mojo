from voodoo import Tensor, get_activation_code
from tensor import TensorShape
from .BaseLayer import BaseLayer
from ..initializers import Initializer, GlorotUniform, NoneInitializer
from ..constraints import Constraint, NoneConstraint
from utils.variant import Variant


struct Dense[
    in_neurons: Int,
    out_neurons: Int,
    activation: String = "none",
    use_bias: Bool = True,
    weight_initializer: Initializer = NoneInitializer,
    weight_constraint: Constraint = NoneConstraint,
    bias_initializer: Initializer = NoneInitializer,
    bias_constraint: Constraint = NoneConstraint,
]():
    var W: Tensor[
        TensorShape(in_neurons, out_neurons), weight_initializer, weight_constraint
    ]
    var bias: Tensor[TensorShape(out_neurons), bias_initializer, bias_constraint]

    fn __init__(
        inout self,
    ) raises:
        self.W = Tensor[
            TensorShape(in_neurons, out_neurons), weight_initializer, weight_constraint
        ]().requires_grad()

        @parameter
        if self.use_bias:
            self.bias = Tensor[
                TensorShape(out_neurons), bias_initializer, bias_constraint
            ]().requires_grad()
        else:
            self.bias = Tensor[
                TensorShape(out_neurons), bias_initializer, bias_constraint
            ]()

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[x.shape, NoneInitializer, NoneConstraint, False, False]:
        var computed = x @ self.W

        @parameter
        if self.use_bias:
            computed = computed + self.bias

        @parameter
        if self.activation != "none":
            return computed.compute_activation[get_activation_code[activation]()]()

        return computed
