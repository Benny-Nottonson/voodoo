from memory import memset_zero
from math import log2, exp2, ceil, round
from voodoo.kernels import load_kernels
from .node import Node
from .utils import Vector, get_broadcasted_shape_for_ew_op, warn
from .utils.shape import shape
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


@register_passable("trivial")
struct Graph:
    var nodes: Vector[Node]
    var memory_pool: Vector[DTypePointer[DType.float32]]
    var memory_pool_manager: Pointer[Vector[Int]]  # Needs to be a pointer
    var free_node_ids: Vector[Int]
    var free_data_ids: Vector[Int]
    var last_node_id: Pointer[Int]  # Needs to be a pointer
    var kernels: Pointer[OP_TUPLE]  # Needs to be a pointer
    var forward_order: Vector[Int]
    var grad_nodes_order: Vector[Int]

    fn __init__() -> Self:
        let nodes = Vector[Node]()

        let memory_pool = Vector[DTypePointer[DType.float32]]()

        let memory_pool_manager = Pointer[Vector[Int]].alloc(MEMORY_POOL_SIZE)

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            memory_pool_manager.store(i, Vector[Int]())

        let free_node_ids = Vector[Int]()

        let free_data_ids = Vector[Int]()

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        let forward_order = Vector[Int]()

        let grad_nodes_order = Vector[Int]()

        return Graph {
            nodes: nodes,
            memory_pool: memory_pool,
            memory_pool_manager: memory_pool_manager,
            free_node_ids: free_node_ids,
            free_data_ids: free_data_ids,
            last_node_id: last_node_id,
            kernels: load_kernels(),
            forward_order: forward_order,
            grad_nodes_order: grad_nodes_order,
        }

    fn print_memory_pool_manager(self) raises:
        @unroll
        for i in range(MEMORY_POOL_SIZE):
            let ceiled_cap = exp2(Float32(i)).to_int()
            print_no_newline("    cap:", ceiled_cap)
            print_no_newline(" - data_ids: [")
            for j in range(self.memory_pool_manager.load(i).len.load()):
                print_no_newline(self.memory_pool_manager.load(i).load(j))
                if j < self.memory_pool_manager.load(i).len.load() - 1:
                    print_no_newline(", ")
            print("]")

    fn print(self, accuracy: Int = 6) raises:
        print("\nGraph (Nodes):")
        for i in range(self.nodes.len.load()):
            let node = self.nodes.load(i)
            if node.data_id.load() == -1:
                continue
            node.print(accuracy)

    fn get_free_node_id(self) raises -> Int:
        var fid: Int = 0
        if self.free_node_ids.len.load() > 0:
            fid = self.free_node_ids.pop_back()
        else:
            fid = self.nodes.len.load()
        return fid

    fn get_free_node_id_no_pop(self) raises -> Int:
        var fid: Int = 0
        if self.free_node_ids.len.load() > 0:
            fid = self.free_node_ids.load(self.free_node_ids.len.load() - 1)
        else:
            fid = self.nodes.len.load()
        return fid

    @always_inline("nodebug")
    fn get_free_data_id(self) raises -> Int:
        if self.free_data_ids.len.load() > 0:
            return self.free_data_ids.pop_back()
        return self.memory_pool.len.load()

    @always_inline("nodebug")
    fn load_ceiled_cap(self, cap: Int) raises -> Int:
        return exp2(ceil(log2(Float32(cap)))).to_int()

    @always_inline("nodebug")
    fn get_index(self, cap: Int) raises -> Int:
        return ceil(log2(Float32(cap))).to_int()

    fn node[
        checkpoint: Bool
    ](
        self,
        shape: Vector[Int],
        is_static: Bool,
        is_single: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parents: Node,
    ) raises -> Node:
        var node = Node(self.get_free_node_id(), shape, is_static, other_params.copy())
        node.checkpoint = checkpoint
        node.operator_id = operator_id
        node.is_single_ptr.store(is_single)
        node.grad_operator_id = operator_id + 1

        for i in range(len(parents)):
            node.add_parent(parents[i].load_id())
            var mutable_parent = parents[i]
            mutable_parent.add_child(node.load_id())
            mutable_parent.incr_dependencies()

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].dependencies == 0:
                _ = self.forward_recursive(parents[i])

        let node_id = node.load_id()
        if node_id < self.nodes.len.load():
            self.nodes.store(node_id, node)
        else:
            self.nodes.push_back(node)

        return node

    fn node[
        checkpoint: Bool
    ](
        self,
        shape: DynamicVector[Int],
        is_static: Bool,
        is_single: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parents: Node,
    ) raises -> Node:
        let _shape = Vector[Int]()
        for i in range(len(shape)):
            _shape.push_back(shape[i])
        var node = Node(self.get_free_node_id(), _shape, is_static, other_params.copy())
        node.checkpoint = checkpoint
        node.is_single_ptr.store(is_single)
        node.operator_id = operator_id
        node.grad_operator_id = operator_id + 1

        for i in range(len(parents)):
            node.add_parent(parents[i].load_id())
            var mutable_parent = parents[i]
            mutable_parent.add_child(node.load_id())
            mutable_parent.incr_dependencies()

        self.get_free_data(node)

        for i in range(len(parents)):
            if parents[i].dependencies == 0:
                _ = self.forward_recursive(parents[i])

        let node_id = node.load_id()
        if node_id < self.nodes.len.load():
            self.nodes.store(node_id, node)
        else:
            self.nodes.push_back(node)

        return node

    fn get_free_data(self, node: Node, unique: Bool = False) raises:
        if node.data_id.load() != -1:
            return

        var idx = -1
        for i in range(node.parents.len.load()):
            let ind = node.parents.load(i)
            let parent = self.nodes.load(node.load_parent_id(i))
            if (
                self.load_ceiled_cap(parent.cap) == self.load_ceiled_cap(node.cap)
                and parent.dependencies == 1
                and not parent.is_static
                and not node.is_static
                and not parent.checkpoint
                and not node.checkpoint
                and not unique
                and not parent.is_single_ptr.load()
                and not node.is_single_ptr.load()
            ):
                node.data_id.store(parent.data_id.load())
                node.data.store(0, self.memory_pool.load(node.data_id.load()))
                idx = i
                break

        for i in range(node.parents.len.load()):
            if i == idx:
                continue
            else:
                var parent = self.nodes.load(node.load_parent_id(i))
                parent.decr_dependencies()

        if idx == -1:
            let index = self.get_index(node.cap)
            if self.memory_pool_manager.load(index).len.load() > 0:
                let data_id = self.memory_pool_manager.load(index).pop_back()
                node.data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.cap)

                node.data.store(0, self.memory_pool.load(node.data_id.load()))
                memset_zero(node.data.load(0), ceiled_cap)
            else:
                let data_id = self.get_free_data_id()
                node.data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.cap + 1)
                let new_data_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
                if data_id == self.memory_pool.len.load():
                    self.memory_pool.push_back(new_data_ptr)
                else:
                    self.memory_pool.data.store(data_id, new_data_ptr)

                node.data.store(0, self.memory_pool.load(node.data_id.load()))
                memset_zero(node.data.load(0), ceiled_cap)

    fn get_free_grad(self, node: Node) raises:
        if node.grad_id.load() != -1:
            return

        let index = self.get_index(node.cap)
        if self.memory_pool_manager.load(index).len.load() > 0:
            let grad_id = self.memory_pool_manager.load(index).pop_back()
            node.grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.cap)

            node.data.store(1, self.memory_pool.load(node.grad_id.load()))
            memset_zero(node.data.load(1), ceiled_cap)
        else:
            let grad_id = self.get_free_data_id()
            node.grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.cap)
            let new_grad_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
            if grad_id == self.memory_pool.len.load():
                self.memory_pool.push_back(new_grad_ptr)
            else:
                self.memory_pool.data.store(grad_id, new_grad_ptr)

            node.data.store(1, self.memory_pool.load(node.grad_id.load()))
            memset_zero(node.data.load(1), ceiled_cap)

    fn release_data(self, inout node: Node) raises:
        if (
            node.is_static
            or node.checkpoint
            or node.is_single_ptr.load()
            or node.data_id.load() == -1
        ):
            return

        if node.dependencies == 0:
            let index = self.get_index(node.cap)
            let data_id = node.data_id.load()
            self.memory_pool_manager.load(index).push_back(data_id)
            node.data_id.store(-1)
            node.dependencies = node.children.len.load()
            node.computed_ptr.store(False)

    fn release_data_forced(self, inout node: Node) raises:
        if node.is_static or node.data_id.load() == -1:
            return
        let index = self.get_index(node.cap)
        let data_id = node.data_id.load()
        self.memory_pool_manager.load(index).push_back(data_id)
        node.data_id.store(-1)
        node.computed_ptr.store(False)
        node.dependencies = node.children.len.load()

    fn release_grad_forced(self, node: Node) raises:
        if node.is_static or node.grad_id.load() == -1:
            return
        let index = self.get_index(node.cap)
        let grad_id = node.grad_id.load()
        self.memory_pool_manager.load(index).push_back(grad_id)
        node.grad_id.store(-1)
        node.grad_computed_ptr.store(False)

    fn clear_cache(self, reset_static_nodes: Bool = False) raises:
        let memory_pool = self.memory_pool
        if self.last_node_id.load() != -1:
            var node = self.nodes.load(self.last_node_id.load())
            self.release_data_forced(node)

        for i in range(self.nodes.len.load() - 1):
            if self.nodes.load(i).data_id.load() == -1:
                continue
            for j in range(i + 1, self.nodes.len.load()):
                if self.nodes.load(i).load_id() == self.nodes.load(j).load_id():
                    self.nodes.store(i, Node(-1, -1))
                    break

        for i in range(memory_pool.len.load()):
            let array = memory_pool.load(i)
            for j in range(i + 1, memory_pool.len.load()):
                let other = memory_pool.load(j)
                if array == other:
                    memory_pool.store(i, DTypePointer[DType.float32].get_null())

        let deletable_data = Vector[Bool](memory_pool.len.load())
        for i in range(memory_pool.len.load()):
            deletable_data.store(i, True)
        for i in range(self.nodes.len.load()):
            let node = self.nodes.load(i)
            if node.data_id.load() == -1:
                continue

            if node.is_static:
                if node.data_id.load() != -1:
                    deletable_data.store(node.data_id.load(), False)
                if node.grad_id.load() != -1:
                    deletable_data.store(node.grad_id.load(), False)

        for i in range(deletable_data.len.load()):
            if (
                deletable_data.load(i)
                and not memory_pool.load(i) == DTypePointer[DType.float32].get_null()
            ):
                memory_pool.load(i).free()
        deletable_data.free()

        for i in range(self.nodes.len.load() - 1, -1, -1):
            var node = self.nodes.load(i)
            if node.data_id.load() == -1:
                continue

            if not node.load_is_static():
                self.free_node_ids.push_back(node.load_id())

                node.id_ptr.free()
                node.data_id.free()
                node.grad_id.free()
                node.data.free()
                node.parents.free()
                node.children.free()
                node.computed_ptr.free()
                node.grad_computed_ptr.free()
                node.shape.free()
                node.strides.free()
                node.other_params.free()
            else:
                node.children.clear()
                node.parents.clear()
                node.dependencies = 0
                node.id_ptr.store(0)
                node.data_id.store(0)
                node.grad_id.store(0)

    fn clear(self, reset_static_nodes: Bool = False) raises:
        self.clear_cache(reset_static_nodes)
        self.nodes.free()
        self.memory_pool.free()

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            self.memory_pool_manager.load(i).free()

        self.memory_pool_manager.free()
        self.free_node_ids.free()
        self.free_data_ids.free()
        self.last_node_id.free()
        self.kernels.free()
        self.forward_order.free()

    fn free(inout self):
        for i in range(self.nodes.len.load()):
            var node = self.nodes.load(i)
            node.free()
        self.nodes.free()

        for i in range(self.memory_pool.len.load()):
            self.memory_pool.load(i).free()
        self.memory_pool.free()

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            self.memory_pool_manager.load(i).free()

        self.memory_pool_manager.free()
        self.free_node_ids.free()
        self.free_data_ids.free()
        self.last_node_id.free()
        self.kernels.free()
        self.forward_order.free()
        self.grad_nodes_order.free()

    fn forward_recursive(
        self, node: Node, keep_forward_order: Bool = False
    ) raises -> Node:
        if node.load_computed():
            return node

        let operator_id = node.operator_id
        if node.load_num_parents() == 1:
            var parent1 = self.forward_recursive(
                self.nodes.load(node.load_parent_id(0)),
                keep_forward_order,
            )
            self.get_free_data(node)
            self.kernels.load(operator_id).get[0, UNARY_OP]()(node, parent1)
            self.release_data(parent1)
        else:
            var parent1 = self.forward_recursive(
                self.nodes.load(node.load_parent_id(0)),
                keep_forward_order,
            )
            var parent2 = self.forward_recursive(
                self.nodes.load(node.load_parent_id(1)),
                keep_forward_order,
            )
            self.get_free_data(node)
            self.kernels.load(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)

            self.release_data(parent1)
            self.release_data(parent2)

        if keep_forward_order:
            self.forward_order.push_back(node.load_id())

        node.computed_ptr.store(True)

        return node

    fn forward(self, node: Node, keep_forward_order: Bool = False) raises -> Node:
        self.last_node_id.store(node.load_id())
        let res = self.forward_recursive(node, keep_forward_order)
        return res

    fn forward_static(self, inout node: Node) raises -> Node:
        self.release_data_forced(node)

        for i in range(self.nodes.len.load()):
            var node = self.nodes.load(i)
            if node.is_single_ptr.load():
                continue

            if not node.is_static:
                node.computed_ptr.store(False)
                node.grad_id.store(-1)
                node.data_id.store(-1)
            node.dependencies = node.children.len.load()

        _ = self.forward_recursive(node)

        return self.nodes.load(self.last_node_id.load())

    fn forward_recursive_graph_slice(self, node: Node) raises -> Node:
        if node.computed_ptr.load():
            return node

        let operator_id = node.operator_id
        if node.load_num_parents() == 1:
            var parent1 = self.forward_recursive_graph_slice(
                self.nodes.load(node.parents.load(0))
            )
            self.get_free_data(node, True)

            self.kernels.load(operator_id).get[0, UNARY_OP]()(node, parent1)
        else:
            let parent1 = self.forward_recursive_graph_slice(
                self.nodes.load(node.parents.load(0))
            )
            let parent2 = self.forward_recursive_graph_slice(
                self.nodes.load(node.parents.load(1))
            )

            self.get_free_data(node, True)
            self.kernels.load(operator_id).get[1, BINARY_OP]()(node, parent1, parent2)

        node.computed_ptr.store(True)

        return node

    fn backward_recursive(self, node: Node) raises -> Node:
        if node.grad_computed_ptr.load():
            return node

        for i in range(node.children.len.load()):
            let child_id = node.children.load(i)
            var child = self.nodes.load(child_id)
            _ = self.backward_recursive(child)

            let grad_operator_id = child.grad_operator_id
            if child.parents.len.load() == 1:
                var parent1 = self.nodes.load(child.load_parent_id(0))
                _ = self.forward_recursive_graph_slice(parent1)

                if parent1.grad_id.load() == -1:
                    self.get_free_grad(parent1)

                parent1.grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[0, UNARY_OP]()(child, parent1)

            else:
                var parent1 = self.nodes.load(child.load_parent_id(0))
                var parent2 = self.nodes.load(child.load_parent_id(1))

                _ = self.forward_recursive_graph_slice(parent1)
                _ = self.forward_recursive_graph_slice(parent2)

                if parent1.grad_id.load() == -1:
                    self.get_free_grad(parent1)
                if parent2.grad_id.load() == -1:
                    self.get_free_grad(parent2)

                parent1.grad_computed_ptr.store(True)
                parent2.grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[1, BINARY_OP]()(
                    child, parent1, parent2
                )

            if child.load_id() != self.last_node_id.load():
                self.release_data_forced(child)
            self.release_grad_forced(child)

        return node

    fn find_grad_nodes_order(self, node: Node) raises:
        self.grad_nodes_order.clear()
        for i in range(self.nodes.len.load()):
            var node = self.nodes.load(i)
            node.tmp_visited = False
        self.grad_nodes_order.clear()

        var backward = DynamicVector[Int]()
        backward.push_back(node.load_id())
        var it = 0
        while it < len(backward):
            let currId = backward[it]
            let curr = self.nodes.load(currId)
            for i in range(curr.parents.len.load()):
                let parId = curr.parents.load(i)
                let par = self.nodes.load(parId)
                if not par.tmp_visited:
                    backward.push_back(parId)
            if curr.requires_grad or curr.checkpoint:
                self.grad_nodes_order.push_back(currId)
            var node = self.nodes.load(currId)
            node.tmp_visited = True
            it += 1

    fn backward(self, node: Node) raises:
        self.find_grad_nodes_order(node)

        self.last_node_id.store(node.load_id())

        for i in range(self.nodes.len.load()):
            let node = self.nodes.load(i)
            node.grad_computed_ptr.store(False)

            if node.is_single_ptr.load() or node.load_id() == self.last_node_id.load():
                continue

            if not node.is_static:
                node.grad_id.store(-1)
                if not node.checkpoint:
                    node.computed_ptr.store(False)
                    node.data_id.store(-1)
            else:
                if node.grad_id.load() != -1:
                    memset_zero(
                        node.data.load(1),
                        self.load_ceiled_cap(node.cap),
                    )

        self.get_free_grad(node)
        node.fill_grad(1.0)
        node.grad_computed_ptr.store(True)
        for i in range(self.grad_nodes_order.len.load()):
            let curr_node = self.nodes.load(self.grad_nodes_order.load(i))
            _ = self.backward_recursive(curr_node)

    fn optimizer_step[type: String, learning_rate: Float32](self) raises:
        if type == "sgd":
            SGD[learning_rate].step(self.nodes)
        else:
            warn("Invalid optimizer: " + type + " using sgd\n")
            SGD[learning_rate].step(self.nodes)

    @always_inline("nodebug")
    fn copy(self, parent1: Node) raises -> Node:
        return self.node[False](
            parent1.shape.copy(),
            True,
            False,
            copy_code,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn mmul(self, a: Node, b: Node) raises -> Node:
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        let a_dims = a.num_dims
        let b_dims = b.num_dims
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
        self,
        a: Node,
        b: Node,
        padding: Int,
        stride: Int,
    ) raises -> Node:
        let batch_size = a.shape.load(0)
        let channels = a.shape.load(1)
        let input_width = a.shape.load(2)
        let kernel_width = b.shape.load(1)

        let shape = shape(
            batch_size,
            channels,
            (input_width - kernel_width + 2 * padding) // stride + 1,
        )

        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)

        return self.node[True](shape, False, False, conv1d_code, other_params, a, b)

    @always_inline("nodebug")
    fn conv_2d(
        self,
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

        let shape = shape(
            batch_size,
            channels,
            (input_width - kernel_width + 2 * padding[0]) // stride[0] + 1,
            (input_height - kernel_height + 2 * padding[1]) // stride[1] + 1,
        )

        let other_params = Vector[Int]()
        other_params.push_back(padding[0])
        other_params.push_back(padding[1])
        other_params.push_back(stride[0])
        other_params.push_back(stride[1])

        return self.node[True](shape, False, False, conv2d_code, other_params, a, b)

    @always_inline("nodebug")
    fn maxpool_1d(
        self,
        a: Node,
        kernel_size: Int,
        stride: Int,
        padding: Int,
    ) raises -> Node:
        let other_params = Vector[Int]()
        other_params.push_back(kernel_size)
        other_params.push_back(stride)
        other_params.push_back(padding)

        let shape = Vector[Int]()
        shape.push_back(a.shape.load(0))
        shape.push_back(a.shape.load(1))
        shape.push_back((a.shape.load(2) - kernel_size + 2 * padding) // stride + 1)

        return self.node[True](shape, False, False, maxpool1d_code, other_params, a)

    @always_inline("nodebug")
    fn maxpool_2d(
        self,
        a: Node,
        kernel_size: StaticIntTuple[2],
        stride: Int,
        padding: Int,
    ) raises -> Node:
        let other_params = Vector[Int]()
        other_params.push_back(kernel_size[0])
        other_params.push_back(kernel_size[1])
        other_params.push_back(stride)
        other_params.push_back(padding)

        let shape = Vector[Int]()
        shape.push_back(a.shape.load(0))
        shape.push_back(a.shape.load(1))
        shape.push_back((a.shape.load(2) - kernel_size[0] + 2 * padding) // stride + 1)
        shape.push_back((a.shape.load(3) - kernel_size[1] + 2 * padding) // stride + 1)

        return self.node[True](shape, False, False, maxpool2d_code, other_params, a)

    @always_inline("nodebug")
    fn dropout(
        self, a: Node, dropout_rate: Float32, noise_shape: DynamicVector[Int]
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
    fn reshape(self, parent1: Node, shape: Vector[Int]) raises -> Node:
        return self.node[False](
            shape, False, False, reshape_code, Vector[Int](), parent1
        )

    @always_inline("nodebug")
    fn transp(self, parent1: Node) raises -> Node:
        return self.node[False](
            parent1.shape.copy().get_transposed(),
            False,
            False,
            transp_code,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn sum(self, parent1: Node) raises -> Node:
        return self.node[False](
            shape(1), False, False, sum_code, Vector[Int](), parent1
        )

    @always_inline("nodebug")
    fn function_general[operator_id: Int](self, parent1: Node) raises -> Node:
        return self.node[False](
            parent1.shape.copy(),
            False,
            False,
            operator_id,
            Vector[Int](),
            parent1,
        )

    @always_inline("nodebug")
    fn arithmetic_general[operator_id: Int](self, a: Node, b: Node) raises -> Node:
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
    ](self, parent1: Node) raises -> Node:
        let other_params = Vector[Int]()
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
    ](self, parent1: Node, parent2: Node) raises -> Node:
        return self.node[False](
            shape(1),
            False,
            False,
            operator_id,
            Vector[Int](),
            parent1,
            parent2,
        )
