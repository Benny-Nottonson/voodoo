from memory import memset_zero
from math import ceil
from algorithm import vectorize

from .node import Node
from .graph import Graph
from .utils import Vector

alias VectorF32 = DTypePointer[DType.float32]
alias VectorInt = Vector[Int]
alias DTVector = Vector[VectorF32]
alias NodeVector = Vector[Pointer[Node]]
alias nelts = simdwidthof[DType.float32]()


struct Tensor:
    var graph_ptr: Pointer[Pointer[Graph]]
    var node_ptr: Pointer[Pointer[Node]]

    fn __init__(
        inout self,
        shape: DynamicVector[Int],
        is_static: Bool = True,
        is_single: Bool = False,
        init_graph: Bool = True,
        init_node: Bool = True,
    ) raises:
        let _shape = Vector[Int]()
        for i in range(len(shape)):
            _shape.push_back(shape[i])

        self.__init__(_shape, is_static, is_single, init_graph, init_node)

    fn __init__(
        inout self,
        shape: Vector[Int],
        is_static: Bool = True,
        is_single: Bool = False,
        init_graph: Bool = True,
        init_node: Bool = True,
    ) raises:
        let other_params = Vector[Int]()

        let graph_ptr = Pointer[Pointer[Graph]].alloc(1)
        let node_ptr = Pointer[Pointer[Node]].alloc(1)

        self.graph_ptr = graph_ptr
        self.node_ptr = node_ptr

        if is_static or init_graph:
            let graph = Graph()
            let graph_ptr = Pointer[Pointer[Graph]].alloc(1)
            let graph_p = Pointer[Graph].alloc(1)
            graph_p.store(graph)
            graph_ptr.store(graph_p)
            let node_ptr = Pointer[Pointer[Node]].alloc(1)

            self.graph_ptr = graph_ptr
            self.node_ptr = node_ptr

            if is_static:
                let node_p = graph.node(shape, True, is_single, False, -1, other_params)
                node_ptr.store(node_p)
            elif init_graph and init_node:
                let node_p = graph.node(
                    shape, False, is_single, False, -1, other_params
                )
                node_ptr.store(node_p)

    fn __copyinit__(inout self, other: Self):
        self.graph_ptr = other.graph_ptr
        self.node_ptr = other.node_ptr

    fn load_tensor_for_binary_op(self, other: Tensor) raises -> Tensor:
        if (
            (
                self.node_ptr.load().load().is_static_ptr.load()
                or self.node_ptr.load().load().is_single_ptr.load()
            )
            and other.node_ptr.load().load().is_static_ptr.load()
            or other.node_ptr.load().load().is_single_ptr.load()
        ):
            let new_tensor = Tensor(
                shape=self.node_ptr.load().load().shape_ptr.load().copy(),
                is_static=False,
                is_single=False,
                init_graph=True,
                init_node=False,
            )

            fuse_graphs(new_tensor.graph_ptr, self.graph_ptr)
            fuse_graphs(new_tensor.graph_ptr, other.graph_ptr)

            return new_tensor

        let new_tensor = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=False,
            init_graph=False,
            init_node=False,
        )

        if (
            self.node_ptr.load().load().is_static_ptr.load()
            or self.node_ptr.load().load().is_single_ptr.load()
        ):
            new_tensor.graph_ptr.store(other.graph_ptr.load())
            fuse_graphs(new_tensor.graph_ptr, self.graph_ptr)

        elif (
            other.node_ptr.load().load().is_static_ptr.load()
            or other.node_ptr.load().load().is_single_ptr.load()
        ):
            new_tensor.graph_ptr.store(self.graph_ptr.load())
            fuse_graphs(new_tensor.graph_ptr, other.graph_ptr)
        else:
            if (
                self.graph_ptr.load().load().nodes.load().len.load()
                >= other.graph_ptr.load().load().nodes.load().len.load()
            ):
                new_tensor.graph_ptr.store(self.graph_ptr.load())
                fuse_graphs(new_tensor.graph_ptr, other.graph_ptr, True)

                self.graph_ptr.free()
                other.graph_ptr.load().free()
                other.graph_ptr.free()
            else:
                new_tensor.graph_ptr.store(other.graph_ptr.load())
                fuse_graphs(new_tensor.graph_ptr, self.graph_ptr, True)
                self.graph_ptr.load().free()
                self.graph_ptr.free()
                other.graph_ptr.free()

        return new_tensor

    fn load_tensor_for_unary_op(self) raises -> Tensor:
        if (
            self.node_ptr.load().load().is_static_ptr.load()
            or self.node_ptr.load().load().is_single_ptr.load()
        ):
            let new_tensor = Tensor(
                shape=self.node_ptr.load().load().shape_ptr.load().copy(),
                is_static=False,
                is_single=False,
                init_graph=True,
                init_node=False,
            )
            fuse_graphs(new_tensor.graph_ptr, self.graph_ptr)
            return new_tensor

        let new_tensor = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=False,
            init_graph=False,
            init_node=False,
        )
        new_tensor.graph_ptr.store(self.graph_ptr.load())
        self.graph_ptr.free()

        return new_tensor

    fn print(self, accuracy: Int = 6) raises:
        if not self.node_ptr.load().load().computed_ptr.load():
            _ = self.forward()
        self.node_ptr.load().load().print(accuracy)

    fn print_memory_pool_manager(self) raises:
        self.graph_ptr.load().load().print_memory_pool_manager()

    fn print_graph(self) raises:
        if not self.node_ptr.load().load().computed_ptr.load():
            _ = self.forward()
        self.graph_ptr.load().load().print()

    fn glorot_normal(self) raises -> Self:
        self.node_ptr.load().load().glorot_normal()
        return self

    fn glorot_uniform(self) raises -> Self:
        self.node_ptr.load().load().glorot_uniform()
        return self

    fn he_normal(self) raises -> Self:
        self.node_ptr.load().load().he_normal()
        return self

    fn he_uniform(self) raises -> Self:
        self.node_ptr.load().load().he_uniform()
        return self

    fn identity(self) raises -> Self:
        self.node_ptr.load().load().identity()
        return self

    fn lecun_normal(self) raises -> Self:
        self.node_ptr.load().load().lecun_normal()
        return self

    fn lecun_uniform(self) raises -> Self:
        self.node_ptr.load().load().lecun_uniform()
        return self

    fn ones(self) raises -> Self:
        self.node_ptr.load().load().ones()
        return self

    fn random_normal(self) raises -> Self:
        self.node_ptr.load().load().random_normal()
        return self

    fn random_uniform(self, min: Float32, max: Float32) raises -> Self:
        self.node_ptr.load().load().random_uniform(min, max)
        return self

    fn truncated_normal(self) raises -> Self:
        self.node_ptr.load().load().truncated_normal()
        return self

    fn zeros(self) raises -> Self:
        self.node_ptr.load().load().zeros()
        return self

    fn fill(self, val: Float32) -> Self:
        self.node_ptr.load().load().fill(val)
        return self

    fn _custom_fill(self, vals: DynamicVector[Float32]) -> Self:
        self.node_ptr.load().load()._custom_fill(vals)
        return self

    fn fill_incr(self) raises -> Self:
        self.node_ptr.load().load().fill_incr()
        return self

    fn grad_fill_incr(self) raises -> Self:
        self.node_ptr.load().load().grad_fill_incr()
        return self

    fn requires_grad(self) raises -> Self:
        self.node_ptr.load().load().requires_grad_ptr.store(True)
        self.node_ptr.load().load().is_static_ptr.store(True)
        self.node_ptr.load().load().computed_ptr.store(True)
        return self

    fn static(self) raises -> Self:
        _ = self.forward()
        self.node_ptr.load().load().is_static_ptr.store(True)
        return self

    fn dynamic(self) raises -> Self:
        self.node_ptr.load().load().is_static_ptr.store(False)
        self.node_ptr.load().load().is_single_ptr.store(True)
        _ = self.forward()
        return self

    fn store(self, idx: Int, val: Float32):
        self.node_ptr.load().load().data.load().store(idx, val)

    fn free(self) raises:
        self.graph_ptr.load().load().nodes.load().free()
        self.graph_ptr.load().load().nodes.free()
        self.graph_ptr.load().load().memory_pool.load().free()
        self.graph_ptr.load().load().memory_pool.free()

        @unroll
        for i in range(30):
            self.graph_ptr.load().load().memory_pool_manager.load(i).free()
        self.graph_ptr.load().load().memory_pool_manager.free()
        self.graph_ptr.load().load().free_node_ids.load().free()
        self.graph_ptr.load().load().free_node_ids.free()
        self.graph_ptr.load().load().free_data_ids.load().free()
        self.graph_ptr.load().load().free_data_ids.free()
        self.graph_ptr.load().load().last_node_id.free()
        self.graph_ptr.load().load().kernels.free()
        self.graph_ptr.load().load().forward_order.load().free()
        self.graph_ptr.load().load().forward_order.free()
        self.graph_ptr.load().load().compiled.free()

        self.graph_ptr.load().free()
        self.graph_ptr.free()
        self.node_ptr.free()

    fn clear(self, reset_static_nodes: Bool = False) raises:
        self.graph_ptr.load().load().clear(reset_static_nodes)
        self.graph_ptr.load().free()
        self.graph_ptr.free()
        self.node_ptr.free()

    fn forward(self, keep_forward_order: Bool = False) raises -> Self:
        let graph = self.graph_ptr.load().load()
        _ = graph.forward(self.node_ptr.load(), keep_forward_order)
        return self

    fn forward_static(self) raises -> Self:
        let graph = self.graph_ptr.load().load()
        _ = graph.forward_static(self.node_ptr.load())
        return self

    fn backward(self) raises:
        if not self.node_ptr.load().load().computed_ptr.load():
            _ = self.forward()
        self.graph_ptr.load().load().backward(self.node_ptr.load())

    fn optimize[type: String = "sgd"](self, lr: Float32 = 0.001) raises:
        self.graph_ptr.load().load().optimizer_step[type](lr)

    fn __getitem__(self, idx: Int) raises -> Float32:
        if not self.node_ptr.load().load().computed_ptr.load():
            _ = self.forward()
        return self.node_ptr.load().load().data.load().load(idx)

    fn __setitem__(self, idx: Int, val: Float32) raises:
        self.node_ptr.load().load().data.load().store(idx, val)

    fn capacity(self) raises -> Int:
        return self.node_ptr.load().load().cap_ptr.load()

    fn sin(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().sin(self.node_ptr.load())
        )
        return new_tensor

    fn cos(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().cos(self.node_ptr.load())
        )
        return new_tensor

    fn tan(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().tan(self.node_ptr.load())
        )
        return new_tensor

    fn acos(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().acos(self.node_ptr.load())
        )
        return new_tensor

    fn asin(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().asin(self.node_ptr.load())
        )
        return new_tensor

    fn atan(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().atan(self.node_ptr.load())
        )
        return new_tensor

    fn cosh(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().cosh(self.node_ptr.load())
        )
        return new_tensor

    fn sinh(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().sinh(self.node_ptr.load())
        )
        return new_tensor

    fn log(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().log(self.node_ptr.load())
        )
        return new_tensor

    fn log2(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().log2(self.node_ptr.load())
        )
        return new_tensor

    fn exp2(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().exp2(self.node_ptr.load())
        )
        return new_tensor

    fn sqrt(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().sqrt(self.node_ptr.load())
        )
        return new_tensor

    fn abs(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().abs(self.node_ptr.load())
        )
        return new_tensor

    fn copy(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().copy(self.node_ptr.load())
        )
        return new_tensor

    fn max_pool_2d(
        self, kernel_width: Int, kernel_height: Int, stride: Int = 1, padding: Int = 0
    ) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .max_pool_2d(
                self.node_ptr.load(), kernel_width, kernel_height, stride, padding
            )
        )
        return new_tensor

    fn __add__(self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .add(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn __sub__(self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .sub(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn __mul__(self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .mul(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn __truediv__(self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .div(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn __pow__(self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .pow(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn __matmul__(self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .mmul(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn conv_2d(self, other: Tensor, padding: Int = 0, stride: Int = 1) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .conv_2d(self.node_ptr.load(), other.node_ptr.load(), padding, stride)
        )
        return new_tensor

    fn __radd__(self, other: Tensor) raises -> Tensor:
        return self.__add__(other)

    fn __rsub__(self, other: Tensor) raises -> Tensor:
        return self.__sub__(other)

    fn __rmul__(self, other: Tensor) raises -> Tensor:
        return self.__mul__(other)

    fn __rtruediv__(self, other: Tensor) raises -> Tensor:
        return self.__truediv__(other)

    fn __rpow__(self, other: Tensor) raises -> Tensor:
        return self.__pow__(other)

    fn __rmatmul__(self, other: Tensor) raises -> Tensor:
        return self.__matmul__(other)

    fn __iadd__(self, other: Tensor) raises:
        self.node_ptr.store(
            self.graph_ptr.load()
            .load()
            .add(self.node_ptr.load(), other.node_ptr.load())
        )

    fn __isub__(self, other: Tensor) raises:
        self.node_ptr.store(
            self.graph_ptr.load()
            .load()
            .sub(self.node_ptr.load(), other.node_ptr.load())
        )

    fn __imul__(self, other: Tensor) raises:
        self.node_ptr.store(
            self.graph_ptr.load()
            .load()
            .mul(self.node_ptr.load(), other.node_ptr.load())
        )

    fn __itruediv__(self, other: Tensor) raises:
        self.node_ptr.store(
            self.graph_ptr.load()
            .load()
            .div(self.node_ptr.load(), other.node_ptr.load())
        )

    fn __ipow__(self, other: Tensor) raises:
        self.node_ptr.store(
            self.graph_ptr.load()
            .load()
            .pow(self.node_ptr.load(), other.node_ptr.load())
        )

    fn __imatmul__(self, other: Tensor) raises:
        self.node_ptr.store(
            self.graph_ptr.load()
            .load()
            .mmul(self.node_ptr.load(), other.node_ptr.load())
        )

    fn __add__(self, number: Float32) raises -> Tensor:
        let other = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=True,
            init_graph=True,
            init_node=True,
        ).fill(number)
        other.node_ptr.load().load().computed_ptr.store(True)
        return self.__add__(other)

    fn __sub__(self, number: Float32) raises -> Tensor:
        let other = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=True,
            init_graph=True,
            init_node=True,
        ).fill(number)
        other.node_ptr.load().load().computed_ptr.store(True)
        return self.__sub__(other)

    fn __mul__(self, number: Float32) raises -> Tensor:
        let other = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=True,
            init_graph=True,
            init_node=True,
        ).fill(number)
        other.node_ptr.load().load().computed_ptr.store(True)
        return self.__mul__(other)

    fn __truediv__(self, number: Float32) raises -> Tensor:
        let other = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=True,
            init_graph=True,
            init_node=True,
        ).fill(number)
        other.node_ptr.load().load().computed_ptr.store(True)
        return self.__truediv__(other)

    fn __pow__(self, number: Float32) raises -> Tensor:
        let other = Tensor(
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
            is_static=False,
            is_single=True,
            init_graph=True,
            init_node=True,
        ).fill(number)
        other.node_ptr.load().load().computed_ptr.store(True)
        return self.__pow__(other)

    fn __radd__(self, number: Float32) raises -> Tensor:
        return self.__add__(number)

    fn __rsub__(self, number: Float32) raises -> Tensor:
        return self.__sub__(number)

    fn __rmul__(self, number: Float32) raises -> Tensor:
        return self.__mul__(number)

    fn __rtruediv__(self, number: Float32) raises -> Tensor:
        return self.__truediv__(number)

    fn __rpow__(self, number: Float32) raises -> Tensor:
        let other = Tensor(
            self.node_ptr.load().load().shape_ptr.load().copy(),
            False,
            False,
            True,
            True,
        ).fill(number)
        other.node_ptr.load().load().is_single_ptr.store(True)
        other.node_ptr.load().load().computed_ptr.store(True)
        return other.__pow__(self)

    fn __iadd__(inout self, number: Float32) raises:
        self = self.__add__(number)

    fn __isub__(inout self, number: Float32) raises:
        self = self.__sub__(number)

    fn __imul__(inout self, number: Float32) raises:
        self = self.__mul__(number)

    fn __itruediv__(inout self, number: Float32) raises:
        self = self.__truediv__(number)

    fn __ipow__(inout self, number: Float32) raises:
        self = self.__pow__(number)

    fn __len__(self) raises -> Int:
        return self.capacity()

    fn reshape(self, shape: Vector[Int]) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().reshape(self.node_ptr.load(), shape)
        )
        return new_tensor

    fn transpose(self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().transpose(self.node_ptr.load())
        )
        return new_tensor

    fn sum(self, axis: Int = -1) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().sum(self.node_ptr.load(), axis)
        )
        return new_tensor

    fn compute_loss[operator_id: Int](self, other: Tensor) raises -> Tensor:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .loss_general[operator_id=operator_id](
                self.node_ptr.load(), other.node_ptr.load()
            )
        )
        return new_tensor

    fn compute_activation[operator_id: Int](self) raises -> Tensor:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .activation_general[operator_id=operator_id](self.node_ptr.load())
        )
        return new_tensor


fn fuse_graphs(
    graph_ptr: Pointer[Pointer[Graph]],
    other_graph_ptr: Pointer[Pointer[Graph]],
    remove_other: Bool = False,
) raises:
    let num_nodes = graph_ptr.load().load().nodes.load().len.load()
    let memory_pool_len = graph_ptr.load().load().memory_pool.load().len.load()

    for i in range(other_graph_ptr.load().load().nodes.load().len.load()):
        let node_ptr = other_graph_ptr.load().load().nodes.load().load(i)
        node_ptr.load().id_ptr.store(node_ptr.load().id_ptr.load() + num_nodes)
        for j in range(node_ptr.load().children_ptr.load().len.load()):
            node_ptr.load().children_ptr.load().store(
                j, node_ptr.load().children_ptr.load().load(j) + num_nodes
            )
        for j in range(node_ptr.load().parents_ptr.load().len.load()):
            node_ptr.load().parents_ptr.load().store(
                j, node_ptr.load().parents_ptr.load().load(j) + num_nodes
            )
        node_ptr.load().data_id.store(node_ptr.load().data_id.load() + memory_pool_len)
        graph_ptr.load().load().nodes.load().push_back(node_ptr)

    for i in range(other_graph_ptr.load().load().memory_pool.load().len.load()):
        graph_ptr.load().load().memory_pool.load().push_back(
            other_graph_ptr.load().load().memory_pool.load().load(i)
        )

    for i in range(30):
        for j in range(
            other_graph_ptr.load().load().memory_pool_manager.load(i).len.load()
        ):
            graph_ptr.load().load().memory_pool_manager.load(i).push_back(
                other_graph_ptr.load().load().memory_pool_manager.load(i).load(j)
                + memory_pool_len
            )

    for i in range(graph_ptr.load().load().free_node_ids.load().len.load()):
        graph_ptr.load().load().free_node_ids.load().push_back(
            other_graph_ptr.load().load().free_node_ids.load().load(i) + num_nodes
        )

    for i in range(graph_ptr.load().load().free_data_ids.load().len.load()):
        graph_ptr.load().load().free_data_ids.load().push_back(
            other_graph_ptr.load().load().free_data_ids.load().load(i) + memory_pool_len
        )

    if remove_other:
        other_graph_ptr.load().load().nodes.load().free()
        other_graph_ptr.load().load().nodes.free()
        other_graph_ptr.load().load().memory_pool.load().free()
        other_graph_ptr.load().load().memory_pool.free()
        for i in range(30):
            other_graph_ptr.load().load().memory_pool_manager.load(i).free()
        other_graph_ptr.load().load().memory_pool_manager.free()
        other_graph_ptr.load().load().free_node_ids.load().free()
        other_graph_ptr.load().load().free_node_ids.free()
        other_graph_ptr.load().load().free_data_ids.load().free()
        other_graph_ptr.load().load().free_data_ids.free()
        other_graph_ptr.load().load().last_node_id.free()
        other_graph_ptr.load().load().kernels.free()
        other_graph_ptr.load().load().forward_order.load().free()
        other_graph_ptr.load().load().forward_order.free()
        other_graph_ptr.load().load().compiled.free()


fn add(a: Tensor, b: Tensor) raises -> Tensor:
    return a + b


fn sub(a: Tensor, b: Tensor) raises -> Tensor:
    return a - b


fn mul(a: Tensor, b: Tensor) raises -> Tensor:
    return a * b


fn div(a: Tensor, b: Tensor) raises -> Tensor:
    return a / b


fn pow(a: Tensor, b: Tensor) raises -> Tensor:
    return a**b


fn mmul(a: Tensor, b: Tensor) raises -> Tensor:
    return a @ b


fn conv_2d(a: Tensor, b: Tensor, stride: Int = 1, padding: Int = 0) raises -> Tensor:
    return a.conv_2d(b, padding, stride)


fn sin(tensor: Tensor) raises -> Tensor:
    return tensor.sin()


fn cos(tensor: Tensor) raises -> Tensor:
    return tensor.cos()


fn tan(tensor: Tensor) raises -> Tensor:
    return tensor.tan()


fn acos(tensor: Tensor) raises -> Tensor:
    return tensor.acos()


fn asin(tensor: Tensor) raises -> Tensor:
    return tensor.asin()


fn atan(tensor: Tensor) raises -> Tensor:
    return tensor.atan()


fn cosh(tensor: Tensor) raises -> Tensor:
    return tensor.cosh()


fn sinh(tensor: Tensor) raises -> Tensor:
    return tensor.sinh()


fn log(tensor: Tensor) raises -> Tensor:
    return tensor.log()


fn log2(tensor: Tensor) raises -> Tensor:
    return tensor.log2()


fn exp2(tensor: Tensor) raises -> Tensor:
    return tensor.exp2()


fn sqrt(tensor: Tensor) raises -> Tensor:
    return tensor.sqrt()


fn abs(tensor: Tensor) raises -> Tensor:
    return tensor.abs()


fn deep_copy(tensor: Tensor) raises -> Tensor:
    return tensor.copy()


fn reshape(tensor: Tensor, shape: DynamicVector[Int]) raises -> Tensor:
    let _shape = Vector[Int]()
    for i in range(len(shape)):
        _shape.push_back(shape[i])
    return tensor.reshape(_shape)


fn transpose(tensor: Tensor) raises -> Tensor:
    return tensor.transpose()


fn sum(tensor: Tensor, axis: Int = -1) raises -> Tensor:
    return tensor.sum(axis)


fn max_pool_2d(
    tensor: Tensor,
    kernel_width: Int,
    kernel_height: Int,
    stride: Int = 1,
    padding: Int = 0,
) raises -> Tensor:
    return tensor.max_pool_2d(kernel_width, kernel_height, stride, padding)


fn fill(tensor: Tensor, val: Float32) raises -> Tensor:
    return tensor.fill(val)


fn fill_incr(tensor: Tensor) raises -> Tensor:
    return tensor.fill_incr()


fn zero_grad(tensor: Tensor) raises -> Tensor:
    tensor.node_ptr.load().load().fill_grad(0.0)
    return tensor
