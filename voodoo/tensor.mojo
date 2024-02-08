from tensor import TensorShape

from voodoo.node import Node
from voodoo.graph import Graph
from voodoo.utils import Vector
from voodoo.constants import MEMORY_POOL_SIZE
from voodoo.operator_codes import (
    add_code,
    sub_code,
    mul_code,
    div_code,
    pow_code,
)
from voodoo.constraints import Constraint, NoneConstraint
from voodoo.initializers import Initializer, Zeroes, NoneInitializer


struct Tensor[
    shape: TensorShape,
    initializer: Initializer = NoneInitializer,
    constraint: Constraint = NoneConstraint,
    is_static: Bool = True,
    is_single: Bool = False,
]:
    var graph: Graph
    var node: Node

    fn __init__(
        inout self,
    ) raises:
        self.graph = Graph()
        self.node = self.graph.node[False, shape, is_static, is_single, -1]()

        @parameter
        if initializer.key() != "NoneInitializer":
            initializer().initialize[shape](self.node.get_data())

            self.node.set_computed(True)

            @parameter
            if constraint.key() != "NoneConstraint":
                constraint().constrain[shape](self.node.get_data())

    fn __copyinit__(inout self, other: Self):
        self.graph = other.graph
        self.node = other.node

    fn load_tensor_for_binary_op[
        new_shape: TensorShape = shape
    ](self, other: Tensor) raises -> Tensor[
        new_shape, NoneInitializer, NoneConstraint, False, False
    ]:
        let self_static_or_single = self.node.get_is_static() or self.node.get_is_single()
        let other_static_or_single = other.node.get_is_static() or other.node.get_is_single()
        let first_greater = len(self.graph._nodes) < len(other.graph._nodes)
        let remove_other = not (self_static_or_single or other_static_or_single)

        var new_tensor = Tensor[
            new_shape, NoneInitializer, NoneConstraint, False, False
        ]()

        if self_static_or_single or (not other_static_or_single and first_greater):
            new_tensor.graph = other.graph
            new_tensor.graph.fuse_graphs(self.graph, remove_other)
        else:
            new_tensor.graph = self.graph
            new_tensor.graph.fuse_graphs(other.graph, remove_other)

        return new_tensor

    fn load_tensor_for_unary_op[
        new_shape: TensorShape = shape
    ](self) raises -> Tensor[new_shape, NoneInitializer, NoneConstraint, False, False]:
        if self.node.get_is_static() or self.node.get_is_single():
            var new_tensor = Tensor[
                new_shape, NoneInitializer, NoneConstraint, False, False
            ]()
            new_tensor.graph.fuse_graphs(self.graph)
            return new_tensor
        else:
            var new_tensor = Tensor[
                new_shape, NoneInitializer, NoneConstraint, False, False
            ]()
            new_tensor.graph = self.graph
            return new_tensor

    fn print(inout self, accuracy: Int = 6) raises:
        if not self.node.get_computed():
            _ = self.forward()
        self.node.print(accuracy)

    fn refresh(self) raises:
        initializer().initialize[shape](self.node.get_data())

    fn fill(owned self, val: Float32) -> Self:
        self.node.fill(val)
        return self ^

    fn fill_incr(owned self) raises -> Self:
        self.node.fill_incr()
        return self ^

    fn grad_fill_incr(owned self) raises -> Self:
        self.node.grad_fill_incr()
        return self ^

    fn requires_grad(owned self) raises -> Self:
        self.node.set_is_static(True)
        self.node.set_is_static(True)
        return self ^

    fn static(owned self) raises -> Self:
        _ = self.forward()
        self.node.set_is_static(True)
        return self ^

    fn store(self, idx: Int, val: Float32):
        self.node.get_data().store(idx, val)

    fn free(self) raises:
        self.graph.free()
        self.node.free()

    fn forward(inout self) raises -> Self:
        _ = self.graph.forward(self.node)
        return self

    fn forward_static(inout self) raises -> Self:
        _ = self.graph.forward_static(self.node)
        return self

    fn backward(inout self) raises:
        if not self.node.get_computed():
            _ = self.forward()
        self.graph.backward(self.node)

    fn optimize[type: String = "sgd", lr: Float32 = 0.001](self) raises:
        self.graph.optimizer_step[type, lr]()

    fn __getitem__(self, idx: Int) raises -> Float32:
        return self.node.get_data()[idx]

    fn __setitem__(self, idx: Int, val: Float32) raises:
        self.node.get_data().store(idx, val)

    fn copy(
        self,
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.copy(self.node)
        return new_tensor

    fn dropout[
        dropout_rate: Float32, noise_shape: TensorShape
    ](self) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.dropout(self.node, dropout_rate, noise_shape)
        return new_tensor

    fn _magic_arithmetic_generic[
        operation_code: Int
    ](self, other: Tensor) raises -> Tensor[
        shape, NoneInitializer, NoneConstraint, False, False
    ]:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.arithmetic_general[operation_code](
            self.node, other.node
        )
        return new_tensor

    fn __eq__(self, other: Tensor) raises -> Bool:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.arithmetic_general[add_code](
            self.node, other.node
        )
        return new_tensor.node.is_zero()

    fn __add__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self._magic_arithmetic_generic[add_code](other)

    fn __sub__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self._magic_arithmetic_generic[sub_code](other)

    fn __mul__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self._magic_arithmetic_generic[mul_code](other)

    fn __truediv__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self._magic_arithmetic_generic[div_code](other)

    fn __pow__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self._magic_arithmetic_generic[pow_code](other)

    fn __matmul__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.mmul(self.node, other.node)
        return new_tensor

    fn __radd__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__add__(other)

    fn __rsub__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__sub__(other)

    fn __rmul__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__mul__(other)

    fn __rtruediv__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__truediv__(other)

    fn __rpow__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__pow__(other)

    fn __rmatmul__(
        self, other: Tensor
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__matmul__(other)

    fn __iadd__(inout self, other: Tensor) raises:
        self.node = self.__add__(other).node

    fn __isub__(inout self, other: Tensor) raises:
        self.node = self.__sub__(other).node

    fn __imul__(inout self, other: Tensor) raises:
        self.node = self.__mul__(other).node

    fn __itruediv__(inout self, other: Tensor) raises:
        self.node = self.__truediv__(other).node

    fn __ipow__(inout self, other: Tensor) raises:
        self.node = self.__pow__(other).node

    fn __imatmul__(inout self, other: Tensor) raises:
        self.node = self.__matmul__(other).node

    fn _prep_scalar_tensor(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, True]:
        let new_tensor = Tensor[
            shape, NoneInitializer, NoneConstraint, False, True
        ]().fill(number)
        new_tensor.node.set_computed(True)
        return new_tensor

    fn __add__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__add__(self._prep_scalar_tensor(number))

    fn __sub__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__sub__(self._prep_scalar_tensor(number))

    fn __mul__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__mul__(self._prep_scalar_tensor(number))

    fn __truediv__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__truediv__(self._prep_scalar_tensor(number))

    fn __pow__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__pow__(self._prep_scalar_tensor(number))

    fn __radd__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__add__(number)

    fn __rsub__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__sub__(number)

    fn __rmul__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__mul__(number)

    fn __rtruediv__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        return self.__truediv__(number)

    fn __rpow__(
        self, number: Float32
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var other = Tensor[shape, NoneInitializer, NoneConstraint, False, False]().fill(
            number
        )
        other.node.set_is_single(True)
        other.node.set_computed(True)
        return other.__pow__(self)

    fn reshape[
        new_shape: TensorShape
    ](self) raises -> Tensor[new_shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op[new_shape]()
        new_tensor.node = new_tensor.graph.reshape(self.node, shape)
        return new_tensor

    fn flatten(
        self,
    ) raises -> Tensor[
        TensorShape(self.shape[0], self.shape.num_elements() // self.shape[0]),
        NoneInitializer,
        NoneConstraint,
        False,
        False,
    ]:
        var new_tensor = self.load_tensor_for_unary_op[
            TensorShape(self.shape[0], self.shape.num_elements() // self.shape[0])
        ]()
        new_tensor.node = new_tensor.graph.reshape(
            self.node,
            TensorShape(self.shape[0], self.shape.num_elements() // self.shape[0]),
        )
        return new_tensor

    fn transp(
        self,
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.transp(self.node)
        return new_tensor

    fn sum(self) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.sum(self.node)
        return new_tensor

    fn compute_function[
        operator_id: Int
    ](self) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.function_general[operator_id](self.node)
        return new_tensor

    fn compute_loss[
        operator_id: Int
    ](self, other: Tensor) raises -> Tensor[
        shape, NoneInitializer, NoneConstraint, False, False
    ]:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.loss_general[operator_id](
            self.node, other.node
        )
        return new_tensor

    fn compute_loss[
        operator_name: String
    ](self, other: Tensor) raises -> Tensor[
        shape, NoneInitializer, NoneConstraint, False, False
    ]:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.loss_general[get_loss_code[operator_name]()](
            self.node, other.node
        )
        return new_tensor

    fn compute_activation[
        operator_id: Int, arg1: Float32 = 0.0
    ](self) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.activation_general[operator_id, arg1](
            self.node
        )
        return new_tensor

    fn compute_activation[
        operator_name: String, arg1: Float32 = 0.0
    ](self) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.activation_general[
            get_activation_code[operator_name](), arg1
        ](self.node)
        return new_tensor

    fn conv_1d(
        self, other: Tensor, padding: Int, stride: Int
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.conv_1d(
            self.node, other.node, padding, stride
        )
        return new_tensor

    fn conv_2d(
        self, other: Tensor, padding: StaticIntTuple[2], stride: StaticIntTuple[2]
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node = new_tensor.graph.conv_2d(
            self.node, other.node, padding, stride
        )
        return new_tensor

    fn maxpool_1d(
        self, kernel_size: Int, stride: Int, padding: Int
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.maxpool_1d(
            self.node, kernel_size, stride, padding
        )
        return new_tensor

    fn maxpool_2d(
        self, kernel_size: StaticIntTuple[2], stride: Int, padding: Int
    ) raises -> Tensor[shape, NoneInitializer, NoneConstraint, False, False]:
        var new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node = new_tensor.graph.maxpool_2d(
            self.node, kernel_size, stride, padding
        )
        return new_tensor
