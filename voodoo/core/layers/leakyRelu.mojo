from tensor import TensorShape

from voodoo.core import Tensor, Initializer, Constraint, NoneInitializer, NoneConstraint
from voodoo.utils import get_activation_code, lrelu_code


struct LeakyReLu[
    in_neurons: Int,
    out_neurons: Int,
    use_bias: Bool = True,
    weight_initializer: Initializer = NoneInitializer,
    weight_constraint: Constraint = NoneConstraint,
    bias_initializer: Initializer = NoneInitializer,
    bias_constraint: Constraint = NoneConstraint,
    alpha: Float32 = 0.2,
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

        return computed.compute_activation[operator_id=lrelu_code, arg1=alpha]()
