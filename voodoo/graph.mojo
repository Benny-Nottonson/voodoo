from memory import memset_zero
from math import log2, exp2, ceil, round
from voodoo.kernels import load_kernels
from .node import Node
from .utils import Vector, get_broadcasted_shape_for_ew_op, warn
from .kernels.optimizers import SGD
from .constants import MEMORY_POOL_SIZE, OP_TUPLE, BINARY_OP, UNARY_OP
from .operator_codes import (
    copy_code,
    mmul_code,
    conv1d_code,
    conv2d_code,
    maxpool1d_code,
    maxpool2d_code,
    dropout_code,
    reshape_code,
    transp_code,
    sum_code,
)
from tensor import TensorShape


@register_passable("trivial")
struct Graph:
    var nodes: Vector[Node]
    var memory_pool: Vector[DTypePointer[DType.float32]]
    var memory_pool_manager: Pointer[Vector[Int]]
    var free_node_ids: Vector[Int]
    var free_data_ids: Vector[Int]
    var last_node_id: Pointer[Int]
    var kernels: Pointer[OP_TUPLE]
    var forward_order: Vector[Int]
    var grad_nodes_order: Vector[Int]

    fn __init__() -> Self:
        let memory_pool_manager = Pointer[Vector[Int]].alloc(MEMORY_POOL_SIZE)

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            memory_pool_manager.store(i, Vector[Int]())

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        return Graph {
            nodes: Vector[Node](),
            memory_pool: Vector[DTypePointer[DType.float32]](),
            memory_pool_manager: memory_pool_manager,
            free_node_ids: Vector[Int](),
            free_data_ids: Vector[Int](),
            last_node_id: last_node_id,
            kernels: load_kernels(),
            forward_order: Vector[Int](),
            grad_nodes_order: Vector[Int](),
        }

    @always_inline("nodebug")
    fn get_nodes(self) -> Vector[Node]:
        return self.nodes

    @always_inline("nodebug")
    fn push_back_nodes(inout self, node: Node) raises:
        self.nodes.push_back(node)

    @always_inline("nodebug")
    fn get_memory_pool(self) -> Vector[DTypePointer[DType.float32]]:
        return self.memory_pool

    @always_inline("nodebug")
    fn push_back_memory_pool(inout self, data: DTypePointer[DType.float32]) raises:
        self.memory_pool.push_back(data)

    @always_inline("nodebug")
    fn get_memory_pool_manager(self) -> Pointer[Vector[Int]]:
        return self.memory_pool_manager

    @always_inline("nodebug")
    fn get_free_node_ids(self) -> Vector[Int]:
        return self.free_node_ids

    @always_inline("nodebug")
    fn push_back_free_node_ids(inout self, node_id: Int) raises:
        self.free_node_ids.push_back(node_id)

    @always_inline("nodebug")
    fn pop_back_free_node_ids(inout self) raises -> Int:
        return self.free_node_ids.pop_back()

    @always_inline("nodebug")
    fn get_free_data_ids(self) -> Vector[Int]:
        return self.free_data_ids

    @always_inline("nodebug")
    fn push_back_free_data_ids(inout self, data_id: Int) raises:
        self.free_data_ids.push_back(data_id)

    @always_inline("nodebug")
    fn pop_back_free_data_ids(inout self) raises -> Int:
        return self.free_data_ids.pop_back()

    @always_inline("nodebug")
    fn get_last_node_id(self) -> Int:
        return self.last_node_id.load()

    @always_inline("nodebug")
    fn set_last_node_id(inout self, node_id: Int) raises:
        self.last_node_id.store(node_id)

    @always_inline("nodebug")
    fn free_last_node_id(self):
        self.last_node_id.free()

    @always_inline("nodebug")
    fn get_kernel(self, idx: Int) -> OP_TUPLE:
        return self.kernels.load(idx)

    @always_inline("nodebug")
    fn free_kernels(self):
        self.kernels.free()

    @always_inline("nodebug")
    fn get_free_node_id(inout self) raises -> Int:
        if self.get_free_node_ids().get_len() > 0:
            return self.pop_back_free_node_ids()
        else:
            return self.get_nodes().get_len()

    @always_inline("nodebug")
    fn get_free_data_id(inout self) raises -> Int:
        if self.get_free_data_ids().get_len() > 0:
            return self.pop_back_free_data_ids()
        return self.get_memory_pool().get_len()

    @always_inline("nodebug")
    fn load_ceiled_cap(self, cap: Int) raises -> Int:
        return exp2(ceil(log2(Float32(cap)))).to_int()

    @always_inline("nodebug")
    fn get_index(self, cap: Int) raises -> Int:
        return ceil(log2(Float32(cap))).to_int()

    fn node[
        checkpoint: Bool
    ](
        inout self,
        shape: Vector[Int],
        is_static: Bool,
        is_single: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parents: Node,
    ) raises -> Node:
        var node = Node(self.get_free_node_id(), shape, is_static, other_params.copy())
        node.checkpoint_ptr.store(checkpoint)
        node.operator_id_ptr.store(operator_id)
        node.is_single_ptr.store(is_single)
        node.grad_operator_id_ptr.store(operator_id + 1)

        for i in range(len(parents)):
            node.parents.push_back(parents[i].id_ptr.load())
            var parent = parents[i]
            parent.children.push_back(node.id_ptr.load())
            parent.dependencies_ptr.store(parents[i].dependencies_ptr.load() + 1)

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].dependencies_ptr.load() == 0:
                _ = self.forward_recursive(parents[i])

        let node_id = node.id_ptr.load()
        if node_id < self.get_nodes().get_len():
            self.get_nodes().store(node_id, node)
        else:
            self.push_back_nodes(node)

        return node

    fn node[
        checkpoint: Bool
    ](
        inout self,
        shape: TensorShape,
        is_static: Bool,
        is_single: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parents: Node,
    ) raises -> Node:
        var _shape = Vector[Int]()
        for i in range(shape.rank()):
            _shape.push_back(shape[i])
        var node = Node(self.get_free_node_id(), _shape, is_static, other_params.copy())
        node.checkpoint_ptr.store(checkpoint)
        node.is_single_ptr.store(is_single)
        node.operator_id_ptr.store(operator_id)
        node.grad_operator_id_ptr.store(operator_id + 1)

        for i in range(len(parents)):
            node.parents.push_back(parents[i].id_ptr.load())
            var parent = parents[i]
            parent.children.push_back(node.id_ptr.load())
            parent.dependencies_ptr.store(parents[i].dependencies_ptr.load() + 1)

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].dependencies_ptr.load() == 0:
                _ = self.forward_recursive(parents[i])

        let node_id = node.id_ptr.load()
        if node_id < self.get_nodes().get_len():
            self.get_nodes().store(node_id, node)
        else:
            self.push_back_nodes(node)

        return node

    fn get_free_data(inout self, node: Node, unique: Bool = False) raises:
        if node.data_id_ptr.load() != -1:
            return

        var idx = -1
        for i in range(node.parents.get_len()):
            let ind = node.parents.load(i)
            let parent = self.get_nodes().load(node.parents.load(i))
            if (
                self.load_ceiled_cap(parent.cap_ptr.load())
                == self.load_ceiled_cap(node.cap_ptr.load())
                and parent.dependencies_ptr.load() == 1
                and not parent.is_static_ptr.load()
                and not node.is_static_ptr.load()
                and not parent.checkpoint_ptr.load()
                and not node.checkpoint_ptr.load()
                and not unique
                and not parent.is_single_ptr.load()
                and not node.is_single_ptr.load()
            ):
                node.data_id_ptr.store(parent.data_id_ptr.load())
                node.data_ptr.store(
                    0, self.get_memory_pool().load(node.data_id_ptr.load())
                )
                idx = i
                break

        for i in range(node.parents.get_len()):
            if i == idx:
                continue
            else:
                let parent = self.get_nodes().load(node.parents.load(i))
                parent.dependencies_ptr.store(parent.dependencies_ptr.load() - 1)

        if idx == -1:
            let index = self.get_index(node.cap_ptr.load())
            var mem_pool = self.get_memory_pool_manager().load(index)
            if mem_pool.get_len() > 0:
                let data_id = mem_pool.pop_back()
                node.data_id_ptr.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.cap_ptr.load())

                node.data_ptr.store(
                    0, self.get_memory_pool().load(node.data_id_ptr.load())
                )
                memset_zero(node.data_ptr.load(0), ceiled_cap)
            else:
                let data_id = self.get_free_data_id()
                node.data_id_ptr.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.cap_ptr.load() + 1)
                let new_data_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
                if data_id == self.get_memory_pool().get_len():
                    self.push_back_memory_pool(new_data_ptr)
                else:
                    self.get_memory_pool().data.store(data_id, new_data_ptr)

                node.data_ptr.store(
                    0, self.get_memory_pool().load(node.data_id_ptr.load())
                )
                memset_zero(node.data_ptr.load(0), ceiled_cap)

    fn get_free_grad(inout self, node: Node) raises:
        if node.grad_id_ptr.load() != -1:
            return

        let index = self.get_index(node.cap_ptr.load())
        var mem_pool = self.get_memory_pool_manager().load(index)
        if mem_pool.get_len() > 0:
            let grad_id = mem_pool.pop_back()
            node.grad_id_ptr.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.cap_ptr.load())

            node.data_ptr.store(1, self.get_memory_pool().load(node.grad_id_ptr.load()))
            memset_zero(node.data_ptr.load(1), ceiled_cap)
        else:
            let grad_id = self.get_free_data_id()
            node.grad_id_ptr.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.cap_ptr.load())
            let new_grad_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
            if grad_id == self.get_memory_pool().get_len():
                self.push_back_memory_pool(new_grad_ptr)
            else:
                self.get_memory_pool().data.store(grad_id, new_grad_ptr)

            node.data_ptr.store(1, self.get_memory_pool().load(node.grad_id_ptr.load()))
            memset_zero(node.data_ptr.load(1), ceiled_cap)

    fn release_data(self, node: Node) raises:
        if (
            node.is_static_ptr.load()
            or node.checkpoint_ptr.load()
            or node.is_single_ptr.load()
            or node.data_id_ptr.load() == -1
        ):
            return

        if node.dependencies_ptr.load() == 0:
            let index = self.get_index(node.cap_ptr.load())
            let data_id = node.data_id_ptr.load()
            var mem_pool = self.get_memory_pool_manager().load(index)
            mem_pool.push_back(data_id)
            node.data_id_ptr.store(-1)
            node.dependencies_ptr.store(node.children.get_len())
            node.computed_ptr.store(False)

    fn release_data_forced(self, node: Node) raises:
        if node.is_static_ptr.load() or node.data_id_ptr.load() == -1:
            return
        let index = self.get_index(node.cap_ptr.load())
        let data_id = node.data_id_ptr.load()
        var mem_pool = self.get_memory_pool_manager().load(index)
        mem_pool.push_back(data_id)
        node.data_id_ptr.store(-1)
        node.computed_ptr.store(False)
        node.dependencies_ptr.store(node.children.get_len())

    fn release_grad_forced(self, node: Node) raises:
        if node.is_static_ptr.load() or node.grad_id_ptr.load() == -1:
            return
        let index = self.get_index(node.cap_ptr.load())
        let grad_id = node.grad_id_ptr.load()
        var mem_pool = self.get_memory_pool_manager().load(index)
        mem_pool.push_back(grad_id)
        node.grad_id_ptr.store(-1)
        node.grad_computed_ptr.store(False)

    fn clear_cache(inout self, reset_static_nodes: Bool = False) raises:
        let memory_pool = self.get_memory_pool()
        if self.get_last_node_id() != -1:
            let node = self.get_nodes().load(self.get_last_node_id())
            self.release_data_forced(node)

        for i in range(self.get_nodes().get_len() - 1):
            if self.get_nodes().load(i).data_id_ptr.load() == -1:
                continue
            for j in range(i + 1, self.get_nodes().get_len()):
                if (
                    self.get_nodes().load(i).id_ptr.load()
                    == self.get_nodes().load(j).id_ptr.load()
                ):
                    self.get_nodes().store(i, Node(-1, -1))
                    break

        for i in range(memory_pool.get_len()):
            let array = memory_pool.load(i)
            for j in range(i + 1, memory_pool.get_len()):
                let other = memory_pool.load(j)
                if array == other:
                    memory_pool.store(i, DTypePointer[DType.float32].get_null())

        let deletable_data = Vector[Bool](memory_pool.get_len())
        for i in range(memory_pool.get_len()):
            deletable_data.store(i, True)
        for i in range(self.get_nodes().get_len()):
            let node = self.get_nodes().load(i)
            if node.data_id_ptr.load() == -1:
                continue

            if node.is_static_ptr.load():
                if node.data_id_ptr.load() != -1:
                    deletable_data.store(node.data_id_ptr.load(), False)
                if node.grad_id_ptr.load() != -1:
                    deletable_data.store(node.grad_id_ptr.load(), False)

        for i in range(deletable_data.get_len()):
            if (
                deletable_data.load(i)
                and not memory_pool.load(i) == DTypePointer[DType.float32].get_null()
            ):
                memory_pool.load(i).free()
        deletable_data.free()

        for i in range(self.get_nodes().get_len() - 1, -1, -1):
            var node = self.get_nodes().load(i)
            if node.data_id_ptr.load() == -1:
                continue

            if not node.is_static_ptr:
                self.push_back_free_node_ids(node.id_ptr.load())
                node.free()
            else:
                node.children.clear()
                node.parents.clear()
                node.dependencies_ptr.store(0)
                node.id_ptr.store(0)
                node.data_id_ptr.store(0)
                node.grad_id_ptr.store(0)

    fn free(self):
        self.get_nodes().free()

        for i in range(self.get_memory_pool().get_len()):
            self.get_memory_pool().load(i).free()

        self.get_memory_pool().free()

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            self.get_memory_pool_manager().load(i).free()

        self.get_memory_pool_manager().free()
        self.get_free_node_ids().free()
        self.get_free_data_ids().free()
        self.free_last_node_id()
        self.free_kernels()
        self.forward_order.free()
        self.grad_nodes_order.free()

    fn forward_recursive(
        inout self, node: Node, keep_forward_order: Bool = False
    ) raises -> Node:
        if node.computed_ptr.load():
            return node

        let operator_id = node.operator_id_ptr.load()
        if node.parents.get_len() == 1:
            let parent1 = self.forward_recursive(
                self.get_nodes().load(node.parents.load(0)),
                keep_forward_order,
            )
            self.get_free_data(node)
            self.get_kernel(operator_id).get[0, UNARY_OP]()(node, parent1)
            self.release_data(parent1)
        else:
            let parent1 = self.forward_recursive(
                self.get_nodes().load(node.parents.load(0)),
                keep_forward_order,
            )
            let parent2 = self.forward_recursive(
                self.get_nodes().load(node.parents.load(1)),
                keep_forward_order,
            )
            self.get_free_data(node)
            self.get_kernel(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)

            self.release_data(parent1)
            self.release_data(parent2)

        if keep_forward_order:
            self.forward_order.push_back(node.id_ptr.load())

        node.computed_ptr.store(True)

        return node

    fn forward(inout self, node: Node, keep_forward_order: Bool = False) raises -> Node:
        self.set_last_node_id(node.id_ptr.load())
        let res = self.forward_recursive(node, keep_forward_order)
        return res

    fn forward_static(inout self, node: Node) raises -> Node:
        self.release_data_forced(node)

        for i in range(self.get_nodes().get_len()):
            let node = self.get_nodes().load(i)
            if node.is_single_ptr.load():
                continue

            if not node.is_static_ptr.load():
                node.computed_ptr.store(False)
                node.grad_id_ptr.store(-1)
                node.data_id_ptr.store(-1)
            node.dependencies_ptr.store(node.children.get_len())

        _ = self.forward_recursive(node)

        return self.get_nodes().load(self.get_last_node_id())

    fn forward_recursive_graph_slice(inout self, node: Node) raises -> Node:
        if node.computed_ptr.load():
            return node

        let operator_id = node.operator_id_ptr.load()
        if node.parents.get_len() == 1:
            let parent1 = self.forward_recursive_graph_slice(
                self.get_nodes().load(node.parents.load(0))
            )
            self.get_free_data(node, True)

            self.get_kernel(operator_id).get[0, UNARY_OP]()(node, parent1)
        else:
            let parent1 = self.forward_recursive_graph_slice(
                self.get_nodes().load(node.parents.load(0))
            )
            let parent2 = self.forward_recursive_graph_slice(
                self.get_nodes().load(node.parents.load(1))
            )

            self.get_free_data(node, True)
            self.get_kernel(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)

        node.computed_ptr.store(True)

        return node

    fn backward_recursive(inout self, node: Node) raises -> Node:
        if node.grad_computed_ptr.load():
            return node

        for i in range(node.children.get_len()):
            let child_id = node.children.load(i)
            let child = self.get_nodes().load(child_id)
            _ = self.backward_recursive(child)

            let grad_operator_id = child.grad_operator_id_ptr.load()
            if child.parents.get_len() == 1:
                let parent1 = self.get_nodes().load(child.parents.load(0))
                _ = self.forward_recursive_graph_slice(parent1)

                if parent1.grad_id_ptr.load() == -1:
                    self.get_free_grad(parent1)

                parent1.grad_computed_ptr.store(True)

                self.get_kernel(grad_operator_id).get[0, UNARY_OP]()(child, parent1)

            else:
                let parent1 = self.get_nodes().load(child.parents.load(0))
                let parent2 = self.get_nodes().load(child.parents.load(1))

                _ = self.forward_recursive_graph_slice(parent1)
                _ = self.forward_recursive_graph_slice(parent2)

                if parent1.grad_id_ptr.load() == -1:
                    self.get_free_grad(parent1)
                if parent2.grad_id_ptr.load() == -1:
                    self.get_free_grad(parent2)

                parent1.grad_computed_ptr.store(True)
                parent2.grad_computed_ptr.store(True)

                self.get_kernel(grad_operator_id).get[1, BINARY_OP]()(
                    child, parent1, parent2
                )

            if child.id_ptr.load() != self.get_last_node_id():
                self.release_data_forced(child)
            self.release_grad_forced(child)

        return node

    fn find_grad_nodes_order(inout self, node: Node) raises:
        self.grad_nodes_order.clear()
        for i in range(self.get_nodes().get_len()):
            let node = self.get_nodes().load(i)
            node.tmp_visited_ptr.store(False)
        self.grad_nodes_order.clear()

        var backward = DynamicVector[Int]()
        backward.push_back(node.id_ptr.load())
        var it = 0
        while it < len(backward):
            let currId = backward[it]
            let curr = self.get_nodes().load(currId)
            for i in range(curr.parents.get_len()):
                let parId = curr.parents.load(i)
                let par = self.get_nodes().load(parId)
                if not par.tmp_visited_ptr.load():
                    backward.push_back(parId)
            if curr.is_static_ptr.load() or curr.checkpoint_ptr.load():
                self.grad_nodes_order.push_back(currId)
            let node = self.get_nodes().load(currId)
            node.tmp_visited_ptr.store(True)
            it += 1

    fn backward(inout self, node: Node) raises:
        self.find_grad_nodes_order(node)

        self.set_last_node_id(node.id_ptr.load())

        for i in range(self.get_nodes().get_len()):
            let node = self.get_nodes().load(i)
            node.grad_computed_ptr.store(False)

            if (
                node.is_single_ptr.load()
                or node.id_ptr.load() == self.get_last_node_id()
            ):
                continue

            if not node.is_static_ptr.load():
                node.grad_id_ptr.store(-1)
                if not node.checkpoint_ptr.load():
                    node.computed_ptr.store(False)
                    node.data_id_ptr.store(-1)
            else:
                if node.grad_id_ptr.load() != -1:
                    memset_zero(
                        node.data_ptr.load(1),
                        self.load_ceiled_cap(node.cap_ptr.load()),
                    )

        self.get_free_grad(node)
        node.fill_grad(1.0)
        node.grad_computed_ptr.store(True)
        for i in range(self.grad_nodes_order.get_len()):
            let curr_node = self.get_nodes().load(self.grad_nodes_order.load(i))
            _ = self.backward_recursive(curr_node)

    fn optimizer_step[type: String, learning_rate: Float32](self) raises:
        if type == "sgd":
            SGD[learning_rate].step(self.get_nodes())
        else:
            warn("Invalid optimizer: " + type + " using sgd\n")
            SGD[learning_rate].step(self.get_nodes())

    @always_inline("nodebug")
    fn copy(inout self, parent1: Node) raises -> Node:
        return self.node[False](
            parent1.shape.copy(),
            True,
            False,
            copy_code,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn mmul(inout self, a: Node, b: Node) raises -> Node:
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        let a_dims = a.num_dims_ptr.load()
        let b_dims = b.num_dims_ptr.load()
        shape[len(shape) - 2] = a.shape.copy().load(a_dims - 2)
        shape[len(shape) - 1] = b.shape.copy().load(b_dims - 1)
        if a.shape.load(a_dims - 1) != b.shape.load(b_dims - 2):
            raise "Shapes don't fit for matrix multiplication. Got shapes: " + str(
                a.shape.load(a_dims - 1)
            ) + " " + str(b.shape.load(b_dims - 2))

        let other_params = Vector[Int]()

        return self.node[True](shape, False, False, mmul_code, other_params, a, b)

    @always_inline("nodebug")
    fn conv_1d(
        inout self,
        a: Node,
        b: Node,
        padding: Int,
        stride: Int,
    ) raises -> Node:
        let batch_size = a.shape.load(0)
        let channels = a.shape.load(1)
        let input_width = a.shape.load(2)
        let kernel_width = b.shape.load(1)

        let shape = TensorShape(
            batch_size,
            channels,
            (input_width - kernel_width + 2 * padding) // stride + 1,
        )

        var other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)

        return self.node[True](shape, False, False, conv1d_code, other_params, a, b)

    @always_inline("nodebug")
    fn conv_2d(
        inout self,
        a: Node,
        b: Node,
        padding: StaticIntTuple[2],
        stride: StaticIntTuple[2],
    ) raises -> Node:
        let batch_size = a.shape.load(0)
        let channels = a.shape.load(1)
        let input_width = a.shape.load(2)
        let input_height = a.shape.load(3)
        let kernel_width = b.shape.load(1)
        let kernel_height = b.shape.load(2)

        let shape = TensorShape(
            batch_size,
            channels,
            (input_width - kernel_width + 2 * padding[0]) // stride[0] + 1,
            (input_height - kernel_height + 2 * padding[1]) // stride[1] + 1,
        )

        var other_params = Vector[Int]()
        other_params.push_back(padding[0])
        other_params.push_back(padding[1])
        other_params.push_back(stride[0])
        other_params.push_back(stride[1])

        return self.node[True](shape, False, False, conv2d_code, other_params, a, b)

    @always_inline("nodebug")
    fn maxpool_1d(
        inout self,
        a: Node,
        kernel_size: Int,
        stride: Int,
        padding: Int,
    ) raises -> Node:
        var other_params = Vector[Int]()
        other_params.push_back(kernel_size)
        other_params.push_back(stride)
        other_params.push_back(padding)

        let shape = TensorShape(
            a.shape.load(0),
            a.shape.load(1),
            (a.shape.load(2) - kernel_size + 2 * padding) // stride + 1,
        )

        return self.node[True](shape, False, False, maxpool1d_code, other_params, a)

    @always_inline("nodebug")
    fn maxpool_2d(
        inout self,
        a: Node,
        kernel_size: StaticIntTuple[2],
        stride: Int,
        padding: Int,
    ) raises -> Node:
        var other_params = Vector[Int]()
        other_params.push_back(kernel_size[0])
        other_params.push_back(kernel_size[1])
        other_params.push_back(stride)
        other_params.push_back(padding)

        let shape = TensorShape(
            a.shape.load(0),
            a.shape.load(1),
            (a.shape.load(2) - kernel_size[0] + 2 * padding) // stride + 1,
            (a.shape.load(3) - kernel_size[1] + 2 * padding) // stride + 1,
        )

        return self.node[True](shape, False, False, maxpool2d_code, other_params, a)

    @always_inline("nodebug")
    fn dropout(
        inout self, a: Node, dropout_rate: Float32, noise_shape: DynamicVector[Int]
    ) raises -> Node:
        return self.node[False](
            a.shape.copy(),
            False,
            False,
            dropout_code,
            Vector[Int](),
            a,
        )

    @always_inline("nodebug")
    fn reshape(inout self, parent1: Node, shape: Vector[Int]) raises -> Node:
        return self.node[False](
            shape, False, False, reshape_code, Vector[Int](), parent1
        )

    @always_inline("nodebug")
    fn transp(inout self, parent1: Node) raises -> Node:
        let old_shape = parent1.shape.copy()

        return self.node[False](
            TensorShape(
                old_shape.load(old_shape.get_len() - 1),
                old_shape.load(old_shape.get_len() - 2),
            ),
            False,
            False,
            transp_code,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn sum(inout self, parent1: Node) raises -> Node:
        return self.node[False](
            TensorShape(1), False, False, sum_code, Vector[Int](), parent1
        )

    @always_inline("nodebug")
    fn function_general[operator_id: Int](inout self, parent1: Node) raises -> Node:
        return self.node[False](
            parent1.shape.copy(),
            False,
            False,
            operator_id,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn arithmetic_general[
        operator_id: Int
    ](inout self, a: Node, b: Node) raises -> Node:
        return self.node[False](
            get_broadcasted_shape_for_ew_op(a, b),
            False,
            False,
            operator_id,
            Vector[Int](),
            a,
            b,
        )

    @always_inline("nodebug")
    fn activation_general[
        operator_id: Int,
        arg1: Float32 = 0.0,
    ](inout self, parent1: Node) raises -> Node:
        var other_params = Vector[Int]()
        other_params.push_back(round(arg1 * 1000000.0).to_int())
        return self.node[False](
            parent1.shape.copy(),
            False,
            False,
            operator_id,
            other_params,
            parent1,
        )

    @always_inline("nodebug")
    fn loss_general[
        operator_id: Int
    ](inout self, parent1: Node, parent2: Node) raises -> Node:
        return self.node[False](
            TensorShape(1),
            False,
            False,
            operator_id,
            Vector[Int](),
            parent1,
            parent2,
        )
