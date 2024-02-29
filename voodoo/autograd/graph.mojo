from memory import memset_zero, memset
from math import log2, exp2, ceil, round
from tensor import TensorShape

from voodoo.constants import MEMORY_POOL_SIZE, OP_TUPLE, BINARY_OP, UNARY_OP
from voodoo.autograd.kernels import KERNELS
from voodoo.core import Optimizer
from voodoo.utils import (
    Vector,
    get_broadcasted_shape_for_ew_op,
    warn,
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


@register_passable("trivial")
struct MemoryPool(Sized):
    var _memory_pool: Vector[DTypePointer[DType.float32]]

    fn __init__() -> Self:
        return MemoryPool {_memory_pool: Vector[DTypePointer[DType.float32]]()}

    fn __getitem__(self, index: Int) -> DTypePointer[DType.float32]:
        return self._memory_pool[index]

    fn __setitem__(inout self, index: Int, value: DTypePointer[DType.float32]):
        self._memory_pool[index] = value

    fn __len__(self) -> Int:
        return len(self._memory_pool)

    fn free(self):
        for i in range(len(self._memory_pool)):
            self._memory_pool[i].free()
        self._memory_pool.free()

    fn push_back(inout self, value: DTypePointer[DType.float32]):
        self._memory_pool.push_back(value)


@register_passable("trivial")
struct MemoryPoolManager:
    var _memory_pool_manager: StaticTuple[MEMORY_POOL_SIZE, Vector[Int]]

    fn __init__() -> Self:
        var memory_pool_manager = StaticTuple[MEMORY_POOL_SIZE, Vector[Int]]()

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            memory_pool_manager[i] = Vector[Int]()

        return MemoryPoolManager {_memory_pool_manager: memory_pool_manager}

    fn __getitem__(self, index: Int) -> Vector[Int]:
        return self._memory_pool_manager[index]

    fn __setitem__(inout self, index: Int, value: Vector[Int]):
        self._memory_pool_manager[index] = value

    fn __len__(self) -> Int:
        return MEMORY_POOL_SIZE

    fn free(self):
        @unroll
        for i in range(MEMORY_POOL_SIZE):
            self._memory_pool_manager[i].free()


@register_passable("trivial")
struct Graph:
    var _nodes: Vector[Node]
    var _memory_pool: MemoryPool
    var _memory_pool_manager: MemoryPoolManager
    var _free_node_ids: Vector[Int]
    var _free_data_ids: Vector[Int]
    var _last_node_id: Pointer[Int]
    var _grad_nodes_order: Vector[Int]

    fn __init__() -> Self:
        var last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        return Graph {
            _nodes: Vector[Node](),
            _memory_pool: MemoryPool(),
            _memory_pool_manager: MemoryPoolManager(),
            _free_node_ids: Vector[Int](),
            _free_data_ids: Vector[Int](),
            _last_node_id: last_node_id,
            _grad_nodes_order: Vector[Int](),
        }

    fn get_free_node_id(inout self) raises -> Int:
        if len(self._free_node_ids) > 0:
            return self._free_node_ids.pop_back()
        else:
            return len(self._nodes)

    fn get_free_data_id(inout self) raises -> Int:
        if len(self._free_data_ids) > 0:
            return self._free_data_ids.pop_back()
        return len(self._memory_pool)

    fn load_ceiled_cap(self, cap: Int) raises -> Int:
        return exp2(ceil(log2(Float32(cap)))).to_int()

    fn get_index(self, cap: Int) raises -> Int:
        return ceil(log2(Float32(cap))).to_int()

    fn node[
        checkpoint: Bool,
        is_static: Bool,
        is_single: Bool,
        operator_id: Int,
    ](
        inout self,
        shape: Vector[Int],
        other_params: Vector[Int],
        *parents: Node,
    ) raises -> Node:
        var node = Node(
            self.get_free_node_id(),
            shape,
            is_static,
            other_params.copy(),
            checkpoint,
            operator_id,
            is_single,
        )

        for i in range(len(parents)):
            var parent = parents[i]
            node.push_back_parent(parent.get_id())
            parent.push_back_child(node.get_id())
            parent.set_dependencies(parent.get_dependencies() + 1)

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].get_dependencies() == 0:
                _ = self.forward_recursive(parents[i])

        var node_id = node.get_id()
        if node_id < len(self._nodes):
            self._nodes[node_id] = node
        else:
            self._nodes.push_back(node)

        return node

    fn get_free_data[unique: Bool = False](inout self, node: Node) raises:
        if node.get_data_id() != -1:
            return

        var idx = -1
        var node_parents = node.get_parents()
        var node_cap = node.get_cap()
        var node_is_static = node.get_is_static()
        var node_checkpoint = node.get_checkpoint()
        var node_is_single = node.get_is_single()
        var node_ceiled_cap = self.load_ceiled_cap(node_cap)

        if (
            not unique
            and not node_is_static
            and not node_checkpoint
            and not node_is_single
        ):
            for i in range(len(node_parents)):
                var parent = self._nodes[node_parents[i]]
                if (
                    self.load_ceiled_cap(parent.get_cap()) == node_ceiled_cap
                    and parent.get_dependencies() == 1
                    and not parent.get_is_static()
                    and not parent.get_checkpoint()
                    and not parent.get_is_single()
                ):
                    node.set_data_id(parent.get_data_id())
                    node.set_data(self._memory_pool[node.get_data_id()])
                    idx = i
                    break

        for i in range(len(node_parents)):
            if i == idx:
                continue
            else:
                var parent = self._nodes[node_parents[i]]
                parent.set_dependencies(parent.get_dependencies() - 1)

        if idx == -1:
            var mem_pool = self._memory_pool_manager[self.get_index(node_cap)]
            if len(mem_pool) > 0:
                var data_id = mem_pool.pop_back()
                node.set_data_id(data_id)
                var ceiled_cap = self.load_ceiled_cap(node_cap)

                node.set_data(self._memory_pool[node.get_data_id()])
                memset_zero(node.get_data(), ceiled_cap)
            else:
                var data_id = self.get_free_data_id()
                node.set_data_id(data_id)
                var ceiled_cap = self.load_ceiled_cap(node_cap + 1)
                var new_data_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
                if data_id == len(self._memory_pool):
                    self._memory_pool.push_back(new_data_ptr)
                else:
                    self._memory_pool[data_id] = new_data_ptr

                node.set_data(self._memory_pool[node.get_data_id()])
                memset_zero(node.get_data(), ceiled_cap)

    fn get_free_grad(inout self, node: Node) raises:
        if node.get_grad_id() != -1:
            return

        var index = self.get_index(node.get_cap())
        var mem_pool = self._memory_pool_manager[index]
        if len(mem_pool) > 0:
            var grad_id = mem_pool.pop_back()
            node.set_grad_id(grad_id)
            var ceiled_cap = self.load_ceiled_cap(node.get_cap())

            node.set_grad(self._memory_pool[node.get_grad_id()])
            memset_zero(node.get_grad(), ceiled_cap)
        else:
            var grad_id = self.get_free_data_id()
            node.set_grad_id(grad_id)
            var ceiled_cap = self.load_ceiled_cap(node.get_cap())
            var new_grad_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
            if grad_id == len(self._memory_pool):
                self._memory_pool.push_back(new_grad_ptr)
            else:
                self._memory_pool[grad_id] = new_grad_ptr

            node.set_grad(self._memory_pool[node.get_grad_id()])
            memset_zero(node.get_grad(), ceiled_cap)

    fn release_data[forced: Bool = False](self, node: Node) raises:
        if node.get_is_static() or node.get_data_id() == -1:
            return

        @parameter
        if not forced:
            if node.get_checkpoint() or node.get_is_single():
                return

        @parameter
        if forced:
            var index = self.get_index(node.get_cap())
            var data_id = node.get_data_id()
            var mem_pool = self._memory_pool_manager[index]
            mem_pool.push_back(data_id)
            node.set_data_id(-1)
            node.set_dependencies(len(node.get_children()))
            node.set_computed(False)
            return

        if node.get_dependencies() == 0:
            var index = self.get_index(node.get_cap())
            var data_id = node.get_data_id()
            var mem_pool = self._memory_pool_manager[index]
            mem_pool.push_back(data_id)
            node.set_data_id(-1)
            node.set_dependencies(len(node.get_children()))
            node.set_computed(False)

    fn release_grad_forced(self, node: Node) raises:
        if node.get_is_static() or node.get_grad_id() == -1:
            return
        var index = self.get_index(node.get_cap())
        var grad_id = node.get_grad_id()
        var mem_pool = self._memory_pool_manager[index]
        mem_pool.push_back(grad_id)
        node.set_grad_id(-1)
        node.set_grad_computed(False)

    fn clear_cache(inout self, reset_static_nodes: Bool = False) raises:
        var dt_null = DTypePointer[DType.float32].get_null()

        if self._last_node_id.load() != -1:
            self.release_data[True](self._nodes[self._last_node_id.load()])

        for i in range(len(self._nodes) - 1):
            if self._nodes[i].get_data_id() == -1:
                continue

            for j in range(i + 1, len(self._nodes)):
                if self._nodes[i].get_id() == self._nodes[j].get_id():
                    self._nodes[i].set_data_id(-1)
                    self._nodes[i].set_grad_id(-1)
                    break

        for i in range(len(self._memory_pool) - 1):
            for j in range(i + 1, len(self._memory_pool)):
                if self._memory_pool[i] == self._memory_pool[j]:
                    self._memory_pool[i] = dt_null

        var deletable_data = Vector[Bool](len(self._memory_pool))
        memset(deletable_data._data, True, len(deletable_data))

        for i in range(len(self._nodes)):
            var node = self._nodes[i]
            var data_id = node.get_data_id()

            if node.get_is_static() and data_id != -1:
                deletable_data[data_id] = False
                if node.get_grad_id() != -1:
                    deletable_data[node.get_grad_id()] = False

        for i in range(len(deletable_data)):
            if deletable_data[i] and not self._memory_pool[i] == dt_null:
                self._memory_pool[i].free()
        deletable_data.free()

        for i in range(len(self._nodes) - 1, -1, -1):
            var node = self._nodes[i]
            if node.get_data_id() == -1:
                continue

            if not node.get_is_static():
                self._free_node_ids.push_back(node.get_id())
                node.free()
            else:
                node.clear_children()
                node.clear_parents()
                node.set_dependencies(0)
                node.set_id(0)
                node.set_data_id(0)
                node.set_grad_id(0)

    fn free(self):
        self._nodes.free()
        self._memory_pool.free()
        self._memory_pool_manager.free()
        self._free_node_ids.free()
        self._free_data_ids.free()
        self._last_node_id.free()
        self._grad_nodes_order.free()

    fn forward_recursive(inout self, node: Node) raises -> Node:
        if node.get_computed():
            return node

        var operator_id = node.get_operator_id()
        var parents = node.get_parents()
        var num_parents = len(parents)

        if num_parents == 1:
            var parent_node = self.forward_recursive(self._nodes[parents[0]])
            self.get_free_data(node)
            KERNELS.get(operator_id).get[0, UNARY_OP]()(node, parent_node)
            self.release_data(parent_node)
        else:
            var parent1 = self.forward_recursive(self._nodes[parents[0]])
            var parent2 = self.forward_recursive(self._nodes[parents[1]])
            self.get_free_data(node)
            KERNELS.get(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)
            self.release_data(parent1)
            self.release_data(parent2)

        node.set_computed(True)

        return node

    fn forward(inout self, node: Node) raises -> Node:
        self._last_node_id.store(node.get_id())
        return self.forward_recursive(node)

    fn forward_static(inout self, node: Node) raises -> Node:
        self.release_data[True](node)

        for i in range(len(self._nodes)):
            var node = self._nodes[i]
            if node.get_is_single():
                continue

            if not node.get_is_static():
                node.set_computed(False)
                node.set_grad_id(-1)
                node.set_data_id(-1)
            node.set_dependencies(len(node.get_children()))

        _ = self.forward_recursive(node)

        return self._nodes[self._last_node_id.load()]

    fn forward_recursive_graph_slice(inout self, node: Node) raises -> Node:
        if node.get_computed():
            return node

        var operator_id = node.get_operator_id()
        var parents = node.get_parents()
        var num_parents = len(parents)

        if num_parents == 1:
            var parent1 = self.forward_recursive_graph_slice(self._nodes[parents[0]])
            self.get_free_data[True](node)
            KERNELS.get(operator_id).get[0, UNARY_OP]()(node, parent1)
        else:
            var parent1 = self.forward_recursive_graph_slice(self._nodes[parents[0]])
            var parent2 = self.forward_recursive_graph_slice(self._nodes[parents[1]])
            self.get_free_data[True](node)
            KERNELS.get(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)

        node.set_computed(True)

        return node

    fn backward_recursive(inout self, node: Node) raises -> Node:
        if node.get_grad_computed():
            return node

        var children = node.get_children()

        for i in range(len(children)):
            var child = self._nodes[children[i]]
            var grad_operator_id = child.get_grad_operator_id()
            var child_parents = child.get_parents()

            _ = self.backward_recursive(child)

            if len(child_parents) == 1:
                var parent1 = self._nodes[child_parents[0]]
                _ = self.forward_recursive_graph_slice(parent1)

                if parent1.get_grad_id() == -1:
                    self.get_free_grad(parent1)

                parent1.set_grad_computed(True)

                KERNELS.get(grad_operator_id).get[0, UNARY_OP]()(child, parent1)
            else:
                var parent1 = self._nodes[child_parents[0]]
                var parent2 = self._nodes[child_parents[1]]

                _ = self.forward_recursive_graph_slice(parent1)
                _ = self.forward_recursive_graph_slice(parent2)

                if parent1.get_grad_id() == -1:
                    self.get_free_grad(parent1)
                if parent2.get_grad_id() == -1:
                    self.get_free_grad(parent2)

                parent1.set_grad_computed(True)
                parent2.set_grad_computed(True)

                KERNELS.get(grad_operator_id).get[1, BINARY_OP]()(
                    child, parent1, parent2
                )

            if child.get_id() != self._last_node_id.load():
                self.release_data[True](child)
            self.release_grad_forced(child)

        return node

    fn find_grad_nodes_order(inout self, node: Node) raises:
        self._grad_nodes_order.clear()
        for i in range(len(self._nodes)):
            self._nodes[i].set_tmp_visited(False)

        var backward = Vector[Int]()
        backward.push_back(node.get_id())
        var it = 0
        while it < len(backward):
            var currId = backward[it]
            var curr = self._nodes[currId]
            for i in range(len(curr.get_parents())):
                var parId = curr.get_parents()[i]
                if not self._nodes[parId].get_tmp_visited():
                    backward.push_back(parId)
            if curr.get_is_static() or curr.get_checkpoint():
                self._grad_nodes_order.push_back(currId)
            self._nodes[currId].set_tmp_visited(True)
            it += 1

    fn backward(inout self, node: Node) raises:
        var new_last_node_id = node.get_id()

        self.find_grad_nodes_order(node)
        self._last_node_id.store(new_last_node_id)

        for i in range(len(self._nodes)):
            var node = self._nodes[i]
            node.set_grad_computed(False)

            if node.get_is_single() or node.get_id() == new_last_node_id:
                continue

            if not node.get_is_static():
                node.set_grad_id(-1)
                if not node.get_checkpoint():
                    node.set_computed(False)
                    node.set_data_id(-1)
            else:
                if node.get_grad_id() != -1:
                    memset_zero(
                        node.get_grad(),
                        self.load_ceiled_cap(node.get_cap()),
                    )

        self.get_free_grad(node)
        node.fill_grad(1.0)
        node.set_grad_computed(True)
        for i in range(len(self._grad_nodes_order)):
            _ = self.backward_recursive(self._nodes[self._grad_nodes_order[i]])

    fn optimizer_step[optimizer: Optimizer](self):
        optimizer.step(self._nodes)

    fn copy(inout self, parent1: Node) raises -> Node:
        return self.node[False, True, False, copy_code](
            parent1.get_shape().copy(),
            Vector[Int](),
            parent1,
        )

    fn mmul(inout self, a: Node, b: Node) raises -> Node:
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        var a_dims = a.get_num_dims()
        var b_dims = b.get_num_dims()
        shape[len(shape) - 2] = a.get_shape().copy()[a_dims - 2]
        shape[len(shape) - 1] = b.get_shape().copy()[b_dims - 1]
        if a.get_shape()[a_dims - 1] != b.get_shape()[b_dims - 2]:
            raise "Shapes don't fit for matrix multiplication. Got shapes: " + str(
                a.get_shape()[a_dims - 1]
            ) + " " + str(b.get_shape()[b_dims - 2])

        var other_params = Vector[Int]()

        return self.node[True, False, False, mmul_code](shape, other_params, a, b)

    fn conv_1d(
        inout self,
        a: Node,
        b: Node,
        padding: Int,
        stride: Int,
    ) raises -> Node:
        var batch_size = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]
        var kernel_width = b.get_shape()[1]

        var shape = TensorShape(
            batch_size,
            channels,
            (input_width - kernel_width + 2 * padding) // stride + 1,
        )

        var other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)

        return self.node[True, False, False, conv1d_code](shape, other_params, a, b)

    fn conv_2d(
        inout self,
        a: Node,
        b: Node,
        padding: StaticIntTuple[2],
        stride: StaticIntTuple[2],
    ) raises -> Node:
        var batch_size = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]
        var input_height = a.get_shape()[3]
        var kernel_width = b.get_shape()[1]
        var kernel_height = b.get_shape()[2]

        var shape = TensorShape(
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

        return self.node[True, False, False, conv2d_code](shape, other_params, a, b)

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

        var shape = TensorShape(
            a.get_shape()[0],
            a.get_shape()[1],
            (a.get_shape()[2] - kernel_size + 2 * padding) // stride + 1,
        )

        return self.node[True, False, False, maxpool1d_code](shape, other_params, a)

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

        var shape = TensorShape(
            a.get_shape()[0],
            a.get_shape()[1],
            (a.get_shape()[2] - kernel_size[0] + 2 * padding) // stride + 1,
            (a.get_shape()[3] - kernel_size[1] + 2 * padding) // stride + 1,
        )

        return self.node[True, False, False, maxpool2d_code](shape, other_params, a)

    fn dropout(
        inout self, a: Node, dropout_rate: Float32, noise_shape: TensorShape
    ) raises -> Node:
        return self.node[False, False, False, dropout_code](
            a.get_shape().copy(),
            Vector[Int](),
            a,
        )

    fn reshape(inout self, parent1: Node, shape: Vector[Int]) raises -> Node:
        return self.node[False, False, False, reshape_code](
            shape, Vector[Int](), parent1
        )

    fn transp(inout self, parent1: Node) raises -> Node:
        var old_shape = parent1.get_shape().copy()

        return self.node[False, False, False, transp_code](
            TensorShape(old_shape[len(old_shape) - 1], old_shape[len(old_shape) - 2]),
            Vector[Int](),
            parent1,
        )

    fn sum(inout self, parent1: Node) raises -> Node:
        return self.node[False, False, False, sum_code](
            TensorShape(1), Vector[Int](), parent1
        )

    fn function_general[operator_id: Int](inout self, parent1: Node) raises -> Node:
        return self.node[False, False, False, operator_id](
            parent1.get_shape().copy(),
            Vector[Int](),
            parent1,
        )

    fn arithmetic_general[
        operator_id: Int
    ](inout self, a: Node, b: Node) raises -> Node:
        return self.node[False, False, False, operator_id](
            get_broadcasted_shape_for_ew_op(a, b),
            Vector[Int](),
            a,
            b,
        )

    fn activation_general[
        operator_id: Int,
        arg1: Float32 = 0.0,
    ](inout self, parent1: Node) raises -> Node:
        var other_params = Vector[Int]()
        other_params.push_back(round(arg1 * 1000000.0).to_int())
        return self.node[False, False, False, operator_id](
            parent1.get_shape().copy(),
            other_params,
            parent1,
        )

    fn loss_general[
        operator_id: Int
    ](inout self, parent1: Node, parent2: Node) raises -> Node:
        return self.node[False, False, False, operator_id](
            TensorShape(1),
            Vector[Int](),
            parent1,
            parent2,
        )

    fn fuse_graphs(
        inout self: Graph,
        other_graph: Graph,
        remove_other: Bool = False,
    ) raises:
        var num_nodes = len(self._nodes)
        var memory_pool_len = len(self._memory_pool)

        for i in range(len(other_graph._nodes)):
            var node = other_graph._nodes[i]
            node.set_id(node.get_id() + num_nodes)
            for j in range(len(node.get_children())):
                node.get_children()[j] = node.get_children()[j] + num_nodes
            for j in range(len(node.get_parents())):
                node.get_parents()[j] = node.get_parents()[j] + num_nodes
            node.set_data_id(node.get_data_id() + memory_pool_len)
            self._nodes.push_back(node)

        for i in range(len(other_graph._memory_pool)):
            self._memory_pool.push_back(other_graph._memory_pool[i])

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            var mem_pool_len = len(other_graph._memory_pool_manager[i])
            for j in range(mem_pool_len):
                self._memory_pool_manager[i].push_back(
                    other_graph._memory_pool_manager[i][j] + memory_pool_len
                )

        var free_node_ids_len = len(self._free_node_ids)
        for i in range(free_node_ids_len):
            self._free_node_ids.push_back(other_graph._free_node_ids[i] + num_nodes)

        var free_data_ids_len = len(self._free_data_ids)
        for i in range(free_data_ids_len):
            self._free_data_ids.push_back(
                other_graph._free_data_ids[i] + memory_pool_len
            )

        if remove_other:
            other_graph.free()
