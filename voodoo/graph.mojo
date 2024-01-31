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
    var _nodes: Vector[Node]
    var _memory_pool: Vector[DTypePointer[DType.float32]]
    var _memory_pool_manager: Pointer[Vector[Int]]
    var _free_node_ids: Vector[Int]
    var _free_data_ids: Vector[Int]
    var _last_node_id: Pointer[Int]
    var _kernels: Pointer[OP_TUPLE]
    var _grad_nodes_order: Vector[Int]

    fn __init__() -> Self:
        let memory_pool_manager = Pointer[Vector[Int]].alloc(MEMORY_POOL_SIZE)

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            memory_pool_manager.store(i, Vector[Int]())

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        return Graph {
            _nodes: Vector[Node](),
            _memory_pool: Vector[DTypePointer[DType.float32]](),
            _memory_pool_manager: memory_pool_manager,
            _free_node_ids: Vector[Int](),
            _free_data_ids: Vector[Int](),
            _last_node_id: last_node_id,
            _kernels: load_kernels(),
            _grad_nodes_order: Vector[Int](),
        }

    @always_inline("nodebug")
    fn get_free_node_id(inout self) raises -> Int:
        if len(self._free_data_ids) > 0:
            return self._free_node_ids.pop_back()
        else:
            return len(self._nodes)

    @always_inline("nodebug")
    fn get_free_data_id(inout self) raises -> Int:
        if len(self._free_data_ids) > 0:
            return self._free_data_ids.pop_back()
        return len(self._memory_pool)

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
        node.set_checkpoint(checkpoint)
        node.set_operator_id(operator_id)
        node.set_is_single(is_single)
        node.set_grad_operator_id(operator_id + 1)

        for i in range(len(parents)):
            node.push_back_parent(parents[i].get_id())
            var parent = parents[i]
            parent.push_back_child(node.get_id())
            parent.set_dependencies(parents[i].get_dependencies() + 1)

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].get_dependencies() == 0:
                _ = self.forward_recursive(parents[i])

        let node_id = node.get_id()
        if node_id < len(self._nodes):
            self._nodes.store(node_id, node)
        else:
            self._nodes.push_back(node)

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
        node.set_checkpoint(checkpoint)
        node.set_is_single(is_single)
        node.set_operator_id(operator_id)
        node.set_grad_operator_id(operator_id + 1)

        for i in range(len(parents)):
            node.push_back_parent(parents[i].get_id())
            var parent = parents[i]
            parent.push_back_child(node.get_id())
            parent.set_dependencies(parents[i].get_dependencies() + 1)

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].get_dependencies() == 0:
                _ = self.forward_recursive(parents[i])

        let node_id = node.get_id()
        if node_id < len(self._nodes):
            self._nodes.store(node_id, node)
        else:
            self._nodes.push_back(node)

        return node

    fn get_free_data(inout self, node: Node, unique: Bool = False) raises:
        if node.get_data_id() != -1:
            return

        var idx = -1
        for i in range(len(node.get_parents())):
            let ind = node.get_parents().load(i)
            let parent = self._nodes.load(node.get_parents().load(i))
            if (
                self.load_ceiled_cap(parent.get_cap())
                == self.load_ceiled_cap(node.get_cap())
                and parent.get_dependencies() == 1
                and not parent.get_is_static()
                and not node.get_is_static()
                and not parent.get_checkpoint()
                and not node.get_checkpoint()
                and not unique
                and not parent.get_is_single()
                and not node.get_is_single()
            ):
                node.set_data_id(parent.get_data_id())
                node.set_data(self._memory_pool.load(node.get_data_id()))
                idx = i
                break

        for i in range(len(node.get_parents())):
            if i == idx:
                continue
            else:
                let parent = self._nodes.load(node.get_parents().load(i))
                parent.set_dependencies(parent.get_dependencies() - 1)

        if idx == -1:
            let index = self.get_index(node.get_cap())
            var mem_pool = self._memory_pool_manager.load(index)
            if len(mem_pool) > 0:
                let data_id = mem_pool.pop_back()
                node.set_data_id(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.get_cap())

                node.set_data(self._memory_pool.load(node.get_data_id()))
                memset_zero(node.get_data(), ceiled_cap)
            else:
                let data_id = self.get_free_data_id()
                node.set_data_id(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.get_cap() + 1)
                let new_data_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
                if data_id == len(self._memory_pool):
                    self._memory_pool.push_back(new_data_ptr)
                else:
                    self._memory_pool.store(data_id, new_data_ptr)

                node.set_data(self._memory_pool.load(node.get_data_id()))
                memset_zero(node.get_data(), ceiled_cap)

    fn get_free_grad(inout self, node: Node) raises:
        if node.get_grad_id() != -1:
            return

        let index = self.get_index(node.get_cap())
        var mem_pool = self._memory_pool_manager.load(index)
        if len(mem_pool) > 0:
            let grad_id = mem_pool.pop_back()
            node.set_grad_id(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.get_cap())

            node.set_grad(self._memory_pool.load(node.get_grad_id()))
            memset_zero(node.get_grad(), ceiled_cap)
        else:
            let grad_id = self.get_free_data_id()
            node.set_grad_id(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.get_cap())
            let new_grad_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
            if grad_id == len(self._memory_pool):
                self._memory_pool.push_back(new_grad_ptr)
            else:
                self._memory_pool.store(grad_id, new_grad_ptr)

            node.set_grad(self._memory_pool.load(node.get_grad_id()))
            memset_zero(node.get_grad(), ceiled_cap)

    fn release_data(self, node: Node) raises:
        if (
            node.get_is_static()
            or node.get_checkpoint()
            or node.get_is_single()
            or node.get_data_id() == -1
        ):
            return

        if node.get_dependencies() == 0:
            let index = self.get_index(node.get_cap())
            let data_id = node.get_data_id()
            var mem_pool = self._memory_pool_manager.load(index)
            mem_pool.push_back(data_id)
            node.set_data_id(-1)
            node.set_dependencies(len(node.get_children()))
            node.set_computed(False)

    fn release_data_forced(self, node: Node) raises:
        if node.get_is_static() or node.get_data_id() == -1:
            return
        let index = self.get_index(node.get_cap())
        let data_id = node.get_data_id()
        var mem_pool = self._memory_pool_manager.load(index)
        mem_pool.push_back(data_id)
        node.set_data_id(-1)
        node.set_computed(False)
        node.set_dependencies(len(node.get_children()))

    fn release_grad_forced(self, node: Node) raises:
        if node.get_is_static() or node.get_grad_id() == -1:
            return
        let index = self.get_index(node.get_cap())
        let grad_id = node.get_grad_id()
        var mem_pool = self._memory_pool_manager.load(index)
        mem_pool.push_back(grad_id)
        node.set_grad_id(-1)
        node.set_grad_computed(False)

    fn clear_cache(inout self, reset_static_nodes: Bool = False) raises:
        if self._last_node_id.load() != -1:
            let node = self._nodes.load(self._last_node_id.load())
            self.release_data_forced(node)

        for i in range(len(self._nodes) - 1):
            if self._nodes.load(i).get_data_id() == -1:
                continue
            for j in range(i + 1, len(self._nodes)):
                if self._nodes.load(i).get_id() == self._nodes.load(j).get_id():
                    self._nodes.store(i, Node(-1, -1))
                    break

        for i in range(len(self._memory_pool)):
            let array = self._memory_pool.load(i)
            for j in range(i + 1, len(self._memory_pool)):
                let other = self._memory_pool.load(j)
                if array == other:
                    self._memory_pool.store(i, DTypePointer[DType.float32].get_null())

        let deletable_data = Vector[Bool](len(self._memory_pool))
        for i in range(len(self._memory_pool)):
            deletable_data.store(i, True)
        for i in range(len(self._nodes)):
            let node = self._nodes.load(i)
            if node.get_data_id() == -1:
                continue

            if node.get_is_static():
                if node.get_data_id() != -1:
                    deletable_data.store(node.get_data_id(), False)
                if node.get_grad_id() != -1:
                    deletable_data.store(node.get_grad_id(), False)

        for i in range(len(deletable_data)):
            if (
                deletable_data.load(i)
                and not self._memory_pool.load(i)
                == DTypePointer[DType.float32].get_null()
            ):
                self._memory_pool.load(i).free()
        deletable_data.free()

        for i in range(len(self._nodes) - 1, -1, -1):
            var node = self._nodes.load(i)
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

        for i in range(len(self._memory_pool)):
            self._memory_pool.load(i).free()

        self._memory_pool.free()

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            self._memory_pool_manager.load(i).free()

        self._memory_pool_manager.free()
        self._free_node_ids.free()
        self._free_data_ids.free()
        self._last_node_id.free()
        self._kernels.free()
        self._grad_nodes_order.free()

    fn forward_recursive(inout self, node: Node) raises -> Node:
        if node.get_computed():
            return node

        let operator_id = node.get_operator_id()
        if len(node.get_parents()) == 1:
            let parent1 = self.forward_recursive(
                self._nodes.load(node.get_parents().load(0))
            )
            self.get_free_data(node)
            self._kernels.load(operator_id).get[0, UNARY_OP]()(node, parent1)
            self.release_data(parent1)
        else:
            let parent1 = self.forward_recursive(
                self._nodes.load(node.get_parents().load(0))
            )
            let parent2 = self.forward_recursive(
                self._nodes.load(node.get_parents().load(1))
            )
            self.get_free_data(node)
            self._kernels.load(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)
            self.release_data(parent1)
            self.release_data(parent2)

        node.set_computed(True)

        return node

    fn forward(inout self, node: Node) raises -> Node:
        self._last_node_id.store(node.get_id())
        let res = self.forward_recursive(node)
        return res

    fn forward_static(inout self, node: Node) raises -> Node:
        self.release_data_forced(node)

        for i in range(len(self._nodes)):
            let node = self._nodes.load(i)
            if node.get_is_single():
                continue

            if not node.get_is_static():
                node.set_computed(False)
                node.set_grad_id(-1)
                node.set_data_id(-1)
            node.set_dependencies(len(node.get_children()))

        _ = self.forward_recursive(node)

        return self._nodes.load(self._last_node_id.load())

    fn forward_recursive_graph_slice(inout self, node: Node) raises -> Node:
        if node.get_computed():
            return node

        let operator_id = node.get_operator_id()
        if len(node.get_parents()) == 1:
            let parent1 = self.forward_recursive_graph_slice(
                self._nodes.load(node.get_parents().load(0))
            )
            self.get_free_data(node, True)

            self._kernels.load(operator_id).get[0, UNARY_OP]()(node, parent1)
        else:
            let parent1 = self.forward_recursive_graph_slice(
                self._nodes.load(node.get_parents().load(0))
            )
            let parent2 = self.forward_recursive_graph_slice(
                self._nodes.load(node.get_parents().load(1))
            )

            self.get_free_data(node, True)
            self._kernels.load(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)

        node.set_computed(True)

        return node

    fn backward_recursive(inout self, node: Node) raises -> Node:
        if node.get_grad_computed():
            return node

        for i in range(len(node.get_children())):
            let child_id = node.get_children().load(i)
            let child = self._nodes.load(child_id)
            _ = self.backward_recursive(child)

            let grad_operator_id = child.get_grad_operator_id()
            if len(child.get_parents()) == 1:
                let parent1 = self._nodes.load(child.get_parents().load(0))
                _ = self.forward_recursive_graph_slice(parent1)

                if parent1.get_grad_id() == -1:
                    self.get_free_grad(parent1)

                parent1.set_grad_computed(True)

                self._kernels.load(grad_operator_id).get[0, UNARY_OP]()(child, parent1)

            else:
                let parent1 = self._nodes.load(child.get_parents().load(0))
                let parent2 = self._nodes.load(child.get_parents().load(1))

                _ = self.forward_recursive_graph_slice(parent1)
                _ = self.forward_recursive_graph_slice(parent2)

                if parent1.get_grad_id() == -1:
                    self.get_free_grad(parent1)
                if parent2.get_grad_id() == -1:
                    self.get_free_grad(parent2)

                parent1.set_grad_computed(True)
                parent2.set_grad_computed(True)

                self._kernels.load(grad_operator_id).get[1, BINARY_OP]()(
                    child, parent1, parent2
                )

            if child.get_id() != self._last_node_id.load():
                self.release_data_forced(child)
            self.release_grad_forced(child)

        return node

    fn find_grad_nodes_order(inout self, node: Node) raises:
        self._grad_nodes_order.clear()
        for i in range(len(self._nodes)):
            var node = self._nodes.load(i)
            node.set_tmp_visited(False)
        self._grad_nodes_order.clear()

        var backward = DynamicVector[Int]()
        backward.push_back(node.get_id())
        var it = 0
        while it < len(backward):
            let currId = backward[it]
            let curr = self._nodes.load(currId)
            for i in range(len(curr.get_parents())):
                let parId = curr.get_parents().load(i)
                let par = self._nodes.load(parId)
                if not par.get_tmp_visited():
                    backward.push_back(parId)
            if curr.get_is_static() or curr.get_checkpoint():
                self._grad_nodes_order.push_back(currId)
            var node = self._nodes.load(currId)
            node.set_tmp_visited(True)
            it += 1

    fn backward(inout self, node: Node) raises:
        self.find_grad_nodes_order(node)

        self._last_node_id.store(node.get_id())

        for i in range(len(self._nodes)):
            let node = self._nodes.load(i)
            node.set_grad_computed(False)

            if node.get_is_single() or node.get_id() == self._last_node_id.load():
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
            let curr_node = self._nodes.load(self._grad_nodes_order.load(i))
            _ = self.backward_recursive(curr_node)

    fn optimizer_step[type: String, learning_rate: Float32](self) raises:
        if type == "sgd":
            SGD[learning_rate].step(self._nodes)
        else:
            warn("Invalid optimizer: " + type + " using sgd\n")
            SGD[learning_rate].step(self._nodes)

    @always_inline("nodebug")
    fn copy(inout self, parent1: Node) raises -> Node:
        return self.node[False](
            parent1.get_shape().copy(),
            True,
            False,
            copy_code,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn mmul(inout self, a: Node, b: Node) raises -> Node:
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        let a_dims = a.get_num_dims()
        let b_dims = b.get_num_dims()
        shape[len(shape) - 2] = a.get_shape().copy().load(a_dims - 2)
        shape[len(shape) - 1] = b.get_shape().copy().load(b_dims - 1)
        if a.get_shape().load(a_dims - 1) != b.get_shape().load(b_dims - 2):
            raise "Shapes don't fit for matrix multiplication. Got shapes: " + str(
                a.get_shape().load(a_dims - 1)
            ) + " " + str(b.get_shape().load(b_dims - 2))

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
        let batch_size = a.get_shape().load(0)
        let channels = a.get_shape().load(1)
        let input_width = a.get_shape().load(2)
        let kernel_width = b.get_shape().load(1)

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
        let batch_size = a.get_shape().load(0)
        let channels = a.get_shape().load(1)
        let input_width = a.get_shape().load(2)
        let input_height = a.get_shape().load(3)
        let kernel_width = b.get_shape().load(1)
        let kernel_height = b.get_shape().load(2)

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
            a.get_shape().load(0),
            a.get_shape().load(1),
            (a.get_shape().load(2) - kernel_size + 2 * padding) // stride + 1,
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
            a.get_shape().load(0),
            a.get_shape().load(1),
            (a.get_shape().load(2) - kernel_size[0] + 2 * padding) // stride + 1,
            (a.get_shape().load(3) - kernel_size[1] + 2 * padding) // stride + 1,
        )

        return self.node[True](shape, False, False, maxpool2d_code, other_params, a)

    @always_inline("nodebug")
    fn dropout(
        inout self, a: Node, dropout_rate: Float32, noise_shape: DynamicVector[Int]
    ) raises -> Node:
        return self.node[False](
            a.get_shape().copy(),
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
        let old_shape = parent1.get_shape().copy()

        return self.node[False](
            TensorShape(
                old_shape.load(len(old_shape) - 1),
                old_shape.load(len(old_shape) - 2),
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
            parent1.get_shape().copy(),
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
            parent1.get_shape().copy(),
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

    fn fuse_graphs(
        inout self: Graph,
        other_graph: Graph,
        remove_other: Bool = False,
    ) raises:
        let num_nodes = len(self._nodes)
        let memory_pool_len = len(self._memory_pool)

        for i in range(len(other_graph._nodes)):
            var node = other_graph._nodes.load(i)
            node.set_id(node.get_id() + num_nodes)
            for j in range(len(node.get_children())):
                node.get_children().store(j, node.get_children().load(j) + num_nodes)
            for j in range(len(node.get_parents())):
                node.get_parents().store(j, node.get_parents().load(j) + num_nodes)
            node.set_data_id(node.get_data_id() + memory_pool_len)
            self._nodes.push_back(node)

        for i in range(len(other_graph._memory_pool)):
            self._memory_pool.push_back(other_graph._memory_pool.load(i))

        for i in range(MEMORY_POOL_SIZE):
            for j in range(len(other_graph._memory_pool_manager.load(i))):
                var mem_pool = self._memory_pool_manager.load(i)
                mem_pool.push_back(
                    other_graph._memory_pool_manager.load(i).load(j) + memory_pool_len
                )

        for i in range(len(self._free_node_ids)):
            self._free_node_ids.push_back(
                other_graph._free_node_ids.load(i) + num_nodes
            )

        for i in range(len(self._free_data_ids)):
            self._free_data_ids.push_back(
                other_graph._free_data_ids.load(i) + memory_pool_len
            )

        if remove_other:
            other_graph.free()
