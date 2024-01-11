from .node import Node
from .graph import Graph, memory_pool_size
from .utils import Vector

struct Tensor[is_static: Bool = True, is_single: Bool = False]:
    var graph_ptr: Pointer[Pointer[Graph]]
    var node_ptr: Pointer[Pointer[Node]]

    fn __init__(
        inout self,
        shape: DynamicVector[Int],
    ):
        let _shape = Vector[Int]()
        for i in range(len(shape)):
            _shape.push_back(shape[i])

        self.__init__(_shape)

    fn __init__(
        inout self,
        shape: Vector[Int],
    ):
        let other_params = Vector[Int]()

        self.graph_ptr = Pointer[Pointer[Graph]].alloc(1)
        self.node_ptr = Pointer[Pointer[Node]].alloc(1)

        let graph = Graph()
        let graph_p = Pointer[Graph].alloc(1)
        graph_p.store(graph)
        self.graph_ptr.store(graph_p)
        try:
            let node_p = graph.node(
                shape, is_static, is_single, False, -1, other_params
            )
            self.node_ptr.store(node_p)
        except:
            self.graph_ptr.free()
            self.node_ptr.free()
            graph_p.free()
            print("Error: Tensor initialization failed")

    fn __copyinit__(inout self, other: Self):
        self.graph_ptr = other.graph_ptr
        self.node_ptr = other.node_ptr

    fn load_tensor_for_binary_op(self, other: Tensor) raises -> Tensor[False, False]:
        let self_static_or_single = (
            self.node_ptr.load().load().is_static_ptr.load()
            or self.node_ptr.load().load().is_single_ptr.load()
        )
        let other_static_or_single = (
            other.node_ptr.load().load().is_static_ptr.load()
            or other.node_ptr.load().load().is_single_ptr.load()
        )

        let new_tensor = Tensor[False, False](
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
        )

        if self_static_or_single:
            new_tensor.graph_ptr.store(other.graph_ptr.load())
            fuse_graphs(new_tensor.graph_ptr, self.graph_ptr)
        elif other_static_or_single:
            new_tensor.graph_ptr.store(self.graph_ptr.load())
            fuse_graphs(new_tensor.graph_ptr, other.graph_ptr)
        else:
            let self_nodes_len = self.graph_ptr.load().load().nodes.load().len.load()
            let other_nodes_len = other.graph_ptr.load().load().nodes.load().len.load()

            if self_nodes_len >= other_nodes_len:
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

    fn load_tensor_for_unary_op(self) raises -> Tensor[False, False]:
        let self_static_or_single = (
            self.node_ptr.load().load().is_static_ptr.load()
            or self.node_ptr.load().load().is_single_ptr.load()
        )

        let new_tensor = Tensor[False, False](
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
        )

        if self_static_or_single:
            fuse_graphs(new_tensor.graph_ptr, self.graph_ptr)

        elif not self_static_or_single:
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

    fn initialize[
        initialization_function: String, val: Float32 = 0, val2: Float32 = 0
    ](self) raises -> Tensor[is_static, is_single]:
        self.node_ptr.load().load().initialize[initialization_function, val, val2]()
        return self

    fn fill(self, val: Float32) -> Self:
        self.node_ptr.load().load().fill(val)
        return self

    fn fill_incr(self) raises -> Self:
        self.node_ptr.load().load().fill_incr()
        return self

    fn grad_fill_incr(self) raises -> Self:
        self.node_ptr.load().load().grad_fill_incr()
        return self

    fn requires_grad(self) raises -> Self:
        let node_ptr = self.node_ptr.load().load()
        node_ptr.requires_grad_ptr.store(True)
        node_ptr.is_static_ptr.store(True)
        node_ptr.computed_ptr.store(True)
        return self

    fn static(self) raises -> Self:
        _ = self.forward()
        self.node_ptr.load().load().is_static_ptr.store(True)
        return self

    fn dynamic(self) raises -> Self:
        let node_ptr = self.node_ptr.load().load()
        node_ptr.is_static_ptr.store(False)
        node_ptr.is_single_ptr.store(True)
        _ = self.forward()
        return self

    fn store(self, idx: Int, val: Float32):
        self.node_ptr.load().load().data.load().store(idx, val)

    fn free(self) raises:
        let graph = self.graph_ptr.load().load()
        graph.nodes.load().free()
        graph.nodes.free()
        graph.memory_pool.load().free()
        graph.memory_pool.free()

        @unroll
        for i in range(memory_pool_size):
            graph.memory_pool_manager.load(i).free()

        graph.memory_pool_manager.free()
        graph.free_node_ids.load().free()
        graph.free_node_ids.free()
        graph.free_data_ids.load().free()
        graph.free_data_ids.free()
        graph.last_node_id.free()
        graph.kernels.free()
        graph.forward_order.load().free()
        graph.forward_order.free()
        graph.compiled.free()

        self.graph_ptr.load().free()
        self.graph_ptr.free()
        self.node_ptr.free()

    fn clear(self, reset_static_nodes: Bool = False) raises:
        self.graph_ptr.load().load().clear(reset_static_nodes)
        self.graph_ptr.load().free()
        self.graph_ptr.free()
        self.node_ptr.free()

    fn forward(self, keep_forward_order: Bool = False) raises -> Self:
        _ = (
            self.graph_ptr.load()
            .load()
            .forward(self.node_ptr.load(), keep_forward_order)
        )
        return self

    fn forward_static(self) raises -> Self:
        _ = self.graph_ptr.load().load().forward_static(self.node_ptr.load())
        return self

    fn backward(self) raises:
        if not self.node_ptr.load().load().computed_ptr.load():
            _ = self.forward()
        self.graph_ptr.load().load().backward(self.node_ptr.load())

    fn optimize[type: String = "sgd", lr: Float32 = 0.001](self) raises:
        self.graph_ptr.load().load().optimizer_step[type, lr]()

    fn __getitem__(self, idx: Int) raises -> Float32:
        if not self.node_ptr.load().load().computed_ptr.load():
            _ = self.forward()
        return self.node_ptr.load().load().data.load().load(idx)

    fn __setitem__(self, idx: Int, val: Float32) raises:
        self.node_ptr.load().load().data.load().store(idx, val)

    fn capacity(self) raises -> Int:
        return self.node_ptr.load().load().cap_ptr.load()

    fn copy(self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().copy(self.node_ptr.load())
        )
        return new_tensor

    fn max_pool_2d(
        self, kernel_width: Int, kernel_height: Int, stride: Int = 1, padding: Int = 0
    ) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .max_pool_2d(
                self.node_ptr.load(), kernel_width, kernel_height, stride, padding
            )
        )
        return new_tensor

    fn dropout[
        dropout_rate: Float32, noise_shape: DynamicVector[Int]
    ](self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .dropout(self.node_ptr.load(), dropout_rate, noise_shape)
        )
        return new_tensor

    fn _magic_arithmetic_generic[operation_code: Int](self, other: Tensor) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .arithmetic_general[operation_code](self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn __add__(self, other: Tensor) raises -> Tensor[False, False]:
        return self._magic_arithmetic_generic[add_code](other)

    fn __sub__(self, other: Tensor) raises -> Tensor[False, False]:
        return self._magic_arithmetic_generic[sub_code](other)

    fn __mul__(self, other: Tensor) raises -> Tensor[False, False]:
        return self._magic_arithmetic_generic[mul_code](other)

    fn __truediv__(self, other: Tensor) raises -> Tensor[False, False]:
        return self._magic_arithmetic_generic[div_code](other)

    fn __pow__(self, other: Tensor) raises -> Tensor[False, False]:
        return self._magic_arithmetic_generic[pow_code](other)

    fn __matmul__(self, other: Tensor) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .mmul(self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn conv_2d(self, other: Tensor, stride: Int, padding: Int) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .conv_2d(self.node_ptr.load(), other.node_ptr.load(), padding, stride)
        )
        return new_tensor

    fn __radd__(self, other: Tensor) raises -> Tensor[False, False]:
        return self.__add__(other)

    fn __rsub__(self, other: Tensor) raises -> Tensor[False, False]:
        return self.__sub__(other)

    fn __rmul__(self, other: Tensor) raises -> Tensor[False, False]:
        return self.__mul__(other)

    fn __rtruediv__(self, other: Tensor) raises -> Tensor[False, False]:
        return self.__truediv__(other)

    fn __rpow__(self, other: Tensor) raises -> Tensor[False, False]:
        return self.__pow__(other)

    fn __rmatmul__(self, other: Tensor) raises -> Tensor[False, False]:
        return self.__matmul__(other)

    fn __iadd__(self, other: Tensor) raises:
        self.node_ptr.store((self + other).node_ptr.load())

    fn __isub__(self, other: Tensor) raises:
        self.node_ptr.store((self - other).node_ptr.load())

    fn __imul__(self, other: Tensor) raises:
        self.node_ptr.store((self * other).node_ptr.load())

    fn __itruediv__(self, other: Tensor) raises:
        self.node_ptr.store((self / other).node_ptr.load())

    fn __ipow__(self, other: Tensor) raises:
        self.node_ptr.store((self**other).node_ptr.load())

    fn __imatmul__(self, other: Tensor) raises:
        self.node_ptr.store((self @ other).node_ptr.load())

    fn _prep_scalar_tensor(self, number: Float32) raises -> Tensor[False, True]:
        let new_tensor = Tensor[False, True](
            shape=self.node_ptr.load().load().shape_ptr.load().copy(),
        ).fill(number)
        new_tensor.node_ptr.load().load().computed_ptr.store(True)
        return new_tensor

    fn __add__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__add__(self._prep_scalar_tensor(number))

    fn __sub__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__sub__(self._prep_scalar_tensor(number))

    fn __mul__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__mul__(self._prep_scalar_tensor(number))

    fn __truediv__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__truediv__(self._prep_scalar_tensor(number))

    fn __pow__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__pow__(self._prep_scalar_tensor(number))

    fn __radd__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__add__(number)

    fn __rsub__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__sub__(number)

    fn __rmul__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__mul__(number)

    fn __rtruediv__(self, number: Float32) raises -> Tensor[False, False]:
        return self.__truediv__(number)

    fn __rpow__(self, number: Float32) raises -> Tensor[False, False]:
        let other = Tensor[False, False](
            self.node_ptr.load().load().shape_ptr.load().copy(),
        ).fill(number)
        other.node_ptr.load().load().is_single_ptr.store(True)
        other.node_ptr.load().load().computed_ptr.store(True)
        return other.__pow__(self)

    fn __len__(self) raises -> Int:
        return self.capacity()

    fn reshape(self, shape: Vector[Int]) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().reshape(self.node_ptr.load(), shape)
        )
        return new_tensor

    fn flatten(self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        let shape = Vector[Int]()
        shape.push_back(self.node_ptr.load().load().shape_ptr.load().load(0))
        shape.push_back(
            self.node_ptr.load().load().shape_ptr.load().load(1)
            * self.node_ptr.load().load().shape_ptr.load().load(2)
            * self.node_ptr.load().load().shape_ptr.load().load(3)
        )
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().reshape(self.node_ptr.load(), shape)
        )
        return new_tensor

    fn transp(self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().transp(self.node_ptr.load())
        )
        return new_tensor

    fn sum(self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load().load().sum(self.node_ptr.load())
        )
        return new_tensor

    fn compute_function[operator_id: Int](self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .function_general[operator_id](self.node_ptr.load())
        )
        return new_tensor

    fn compute_loss[operator_id: Int](self, other: Tensor) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .loss_general[operator_id](self.node_ptr.load(), other.node_ptr.load())
        )
        return new_tensor

    fn compute_loss[operator_name: String](self, other: Tensor) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_binary_op(other)
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .loss_general[get_loss_code[operator_name]()](
                self.node_ptr.load(), other.node_ptr.load()
            )
        )
        return new_tensor

    fn compute_activation[operator_id: Int, arg1: Float32 = 0.0](self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .activation_general[operator_id, arg1](self.node_ptr.load())
        )
        return new_tensor

    fn compute_activation[
        operator_name: String, arg1: Float32 = 0.0
    ](self) raises -> Tensor[False, False]:
        let new_tensor = self.load_tensor_for_unary_op()
        new_tensor.node_ptr.store(
            new_tensor.graph_ptr.load()
            .load()
            .activation_general[get_activation_code[operator_name](), arg1](
                self.node_ptr.load()
            )
        )
        return new_tensor


fn fuse_graphs(
    graph_ptr: Pointer[Pointer[Graph]],
    other_graph_ptr: Pointer[Pointer[Graph]],
    remove_other: Bool = False,
) raises:
    let num_nodes = graph_ptr.load().load().nodes.load().len.load()
    let memory_pool_len = graph_ptr.load().load().memory_pool.load().len.load()
    let other_graph = other_graph_ptr.load().load()

    for i in range(other_graph.nodes.load().len.load()):
        let node_ptr = other_graph.nodes.load().load(i)
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

    for i in range(other_graph.memory_pool.load().len.load()):
        graph_ptr.load().load().memory_pool.load().push_back(
            other_graph.memory_pool.load().load(i)
        )

    for i in range(memory_pool_size):
        for j in range(other_graph.memory_pool_manager.load(i).len.load()):
            graph_ptr.load().load().memory_pool_manager.load(i).push_back(
                other_graph.memory_pool_manager.load(i).load(j) + memory_pool_len
            )

    for i in range(graph_ptr.load().load().free_node_ids.load().len.load()):
        graph_ptr.load().load().free_node_ids.load().push_back(
            other_graph.free_node_ids.load().load(i) + num_nodes
        )

    for i in range(graph_ptr.load().load().free_data_ids.load().len.load()):
        graph_ptr.load().load().free_data_ids.load().push_back(
            other_graph.free_data_ids.load().load(i) + memory_pool_len
        )

    if remove_other:
        other_graph.nodes.load().free()
        other_graph.nodes.free()
        other_graph.memory_pool.load().free()
        other_graph.memory_pool.free()
        for i in range(memory_pool_size):
            other_graph.memory_pool_manager.load(i).free()
        other_graph.memory_pool_manager.free()
        other_graph.free_node_ids.load().free()
        other_graph.free_node_ids.free()
        other_graph.free_data_ids.load().free()
        other_graph.free_data_ids.free()
        other_graph.last_node_id.free()
        other_graph.kernels.free()
        other_graph.forward_order.load().free()
        other_graph.forward_order.free()
        other_graph.compiled.free()
