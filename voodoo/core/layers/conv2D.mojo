from tensor import TensorShape

from voodoo.core import Tensor, Initializer, Constraint, NoneInitializer, NoneConstraint
from voodoo.utils import get_activation_code


struct Conv2D[
    in_channels: Int,
    kernel_width: Int,
    kernel_height: Int,
    stride: Int,
    padding: Int,
    activation: String = "none",
    use_bias: Bool = False,
    weight_initializer: Initializer = NoneInitializer,
    weight_constraint: Constraint = NoneConstraint,
    bias_initializer: Initializer = NoneInitializer,
    bias_constraint: Constraint = NoneConstraint,
]():
    var W: Tensor[
        TensorShape(in_channels, kernel_width, kernel_height),
        weight_initializer,
        weight_constraint,
    ]
    var bias: Tensor[TensorShape(in_channels, 1, 1), bias_initializer, bias_constraint]

    fn __init__(
        inout self,
    ) raises:
        self.W = Tensor[
            TensorShape(in_channels, kernel_width, kernel_height),
            weight_initializer,
            weight_constraint,
        ]().requires_grad()

        @parameter
        if self.use_bias:
            self.bias = Tensor[
                TensorShape(in_channels, 1, 1), bias_initializer, bias_constraint
            ]().requires_grad()
        else:
            self.bias = Tensor[
                TensorShape(in_channels, 1, 1), bias_initializer, bias_constraint
            ]()

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[
        TensorShape(
            x.shape[0],
            x.shape[1],
            (x.shape[2] - kernel_width + 2 * self.padding) // self.stride + 1,
            (x.shape[3] - kernel_height + 2 * self.padding) // self.stride + 1,
        ),
        NoneInitializer,
        NoneConstraint,
        False,
        False,
    ]:
        let res = x.conv_2d[
            TensorShape(
                x.shape[0],
                x.shape[1],
                (x.shape[2] - kernel_width + 2 * self.padding) // self.stride + 1,
                (x.shape[3] - kernel_height + 2 * self.padding) // self.stride + 1,
            )
        ](self.W, self.padding, self.stride).compute_activation[self.activation]()

        @parameter
        if self.use_bias:
            return res + self.bias

        return res
