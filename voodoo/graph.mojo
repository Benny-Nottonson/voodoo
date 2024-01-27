from memory import memset_zero
from math import log, log2, exp, exp2, ceil, round
from algorithm import vectorize, unswitch

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
    var nodes: Pointer[Vector[Pointer[Node]]]
    var memory_pool: Pointer[Vector[DTypePointer[DType.float32]]]
    var memory_pool_manager: Pointer[Vector[Int]]
    var free_node_ids: Pointer[Vector[Int]]
    var free_data_ids: Pointer[Vector[Int]]
    var last_node_id: Pointer[Int]
    var kernels: Pointer[OP_TUPLE]
    var forward_order: Pointer[Vector[Int]]
    var grad_nodes_order: Pointer[Vector[Int]]

    fn __init__() -> Self:
        let nodes = Pointer[Vector[Pointer[Node]]].alloc(1)
        nodes.store(Vector[Pointer[Node]]())

        let memory_pool = Pointer[Vector[DTypePointer[DType.float32]]].alloc(1)
        memory_pool.store(Vector[DTypePointer[DType.float32]]())

        let memory_pool_manager = Pointer[Vector[Int]].alloc(MEMORY_POOL_SIZE)

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            memory_pool_manager.store(i, Vector[Int]())

        let free_node_ids = Pointer[Vector[Int]].alloc(1)
        free_node_ids.store(Vector[Int]())

        let free_data_ids = Pointer[Vector[Int]].alloc(1)
        free_data_ids.store(Vector[Int]())

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        let forward_order = Pointer[Vector[Int]].alloc(1)
        forward_order.store(Vector[Int]())

        let grad_nodes_order = Pointer[Vector[Int]].alloc(1)
        grad_nodes_order.store(Vector[Int]())

        let backward_order = Pointer[Vector[Int]].alloc(1)
        backward_order.store(Vector[Int]())

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
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node == Pointer[Node].get_null():
                continue
            node.load().print(accuracy)

    fn get_free_node_id(self) raises -> Int:
        var fid: Int = 0
        if self.free_node_ids.load().len.load() > 0:
            fid = self.free_node_ids.load().pop_back()
        else:
            fid = self.nodes.load().len.load()
        return fid

    fn get_free_node_id_no_pop(self) raises -> Int:
        var fid: Int = 0
        if self.free_node_ids.load().len.load() > 0:
            fid = self.free_node_ids.load().load(
                self.free_node_ids.load().len.load() - 1
            )
        else:
            fid = self.nodes.load().len.load()
        return fid

    fn get_free_data_id(self) raises -> Int:
        if self.free_data_ids.load().len.load() > 0:
            return self.free_data_ids.load().pop_back()
        return self.memory_pool.load().len.load()

    fn load_ceiled_cap(self, cap: Int) raises -> Int:
        return exp2(ceil(log2(Float32(cap)))).to_int()

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
        *parent_ptrs: Pointer[Node],
    ) raises -> Pointer[Node]:
        var node = Node(self.get_free_node_id(), shape, is_static, other_params.copy())
        node.checkpoint = checkpoint
        node.operator_id = operator_id
        node.is_single_ptr.store(is_single)
        node.grad_operator_id = operator_id + 1
        let node_ptr = Pointer[Node].alloc(1)
        node_ptr.store(node)

        for i in range(len(parent_ptrs)):
            node.add_parent(parent_ptrs[i].load().load_id())
            parent_ptrs[i].load().add_child(node.load_id())
            var mutable_parent = parent_ptrs[i].load()
            mutable_parent.incr_dependencies()

        self.get_free_data_ptr(node_ptr)

        for i in range(len(parent_ptrs)):
            if parent_ptrs[i].load().dependencies == 0:
                _ = self.forward_recursive(parent_ptrs[i])

        let node_id = node_ptr.load().load_id()
        if node_id < self.nodes.load().len.load():
            self.nodes.load().store(node_id, node_ptr)
        else:
            self.nodes.load().push_back(node_ptr)

        return node_ptr

    fn node[
        checkpoint: Bool
    ](
        self,
        shape: DynamicVector[Int],
        is_static: Bool,
        is_single: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parent_ptrs: Pointer[Node],
    ) raises -> Pointer[Node]:
        let _shape = Vector[Int]()
        for i in range(len(shape)):
            _shape.push_back(shape[i])
        var node = Node(self.get_free_node_id(), _shape, is_static, other_params.copy())
        node.checkpoint = checkpoint
        node.is_single_ptr.store(is_single)
        node.operator_id = operator_id
        node.grad_operator_id = operator_id + 1
        let node_ptr = Pointer[Node].alloc(1)
        node_ptr.store(node)

        for i in range(len(parent_ptrs)):
            node.add_parent(parent_ptrs[i].load().load_id())
            parent_ptrs[i].load().add_child(node.load_id())
            var mutable_parent = parent_ptrs[i].load()
            mutable_parent.incr_dependencies()

        self.get_free_data_ptr(node_ptr)

        for i in range(len(parent_ptrs)):
            if parent_ptrs[i].load().dependencies == 0:
                _ = self.forward_recursive(parent_ptrs[i])

        let node_id = node_ptr.load().load_id()
        if node_id < self.nodes.load().len.load():
            self.nodes.load().store(node_id, node_ptr)
        else:
            self.nodes.load().push_back(node_ptr)

        return node_ptr

    fn get_free_data_ptr(self, node: Pointer[Node], unique: Bool = False) raises:
        let loaded_node = node.load()
        if loaded_node.data_id.load() != -1:
            return

        var idx = -1
        for i in range(loaded_node.parents.len.load()):
            let ind = loaded_node.parents.load(i)
            let parent = self.nodes.load().load(loaded_node.load_parent_id(i))
            if (
                self.load_ceiled_cap(parent.load().cap)
                == self.load_ceiled_cap(loaded_node.cap)
                and parent.load().dependencies == 1
                and not parent.load().is_static
                and not loaded_node.is_static
                and not parent.load().checkpoint
                and not loaded_node.checkpoint
                and not unique
                and not parent.load().is_single_ptr.load()
                and not loaded_node.is_single_ptr.load()
            ):
                loaded_node.data_id.store(parent.load().data_id.load())
                loaded_node.data.store(
                    0, self.memory_pool.load().load(loaded_node.data_id.load())
                )
                idx = i
                break

        for i in range(loaded_node.parents.len.load()):
            if i == idx:
                continue
            else:
                let parent = self.nodes.load().load(loaded_node.load_parent_id(i))
                var mutable_parent = parent.load()
                mutable_parent.decr_dependencies()

        if idx == -1:
            let index = self.get_index(loaded_node.cap)
            if self.memory_pool_manager.load(index).len.load() > 0:
                let data_id = self.memory_pool_manager.load(index).pop_back()
                loaded_node.data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(loaded_node.cap)

                loaded_node.data.store(
                    0, self.memory_pool.load().load(loaded_node.data_id.load())
                )
                memset_zero(loaded_node.data.load(0), ceiled_cap)
            else:
                let data_id = self.get_free_data_id()
                loaded_node.data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(loaded_node.cap + 1)
                let new_data_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
                if data_id == self.memory_pool.load().len.load():
                    self.memory_pool.load().push_back(new_data_ptr)
                else:
                    self.memory_pool.load().data.load().store(data_id, new_data_ptr)

                loaded_node.data.store(
                    0, self.memory_pool.load().load(loaded_node.data_id.load())
                )
                memset_zero(loaded_node.data.load(0), ceiled_cap)

    fn get_free_grad_ptr(self, node: Pointer[Node]) raises:
        let loaded_node = node.load()
        if loaded_node.grad_id.load() != -1:
            return

        let index = self.get_index(loaded_node.cap)
        if self.memory_pool_manager.load(index).len.load() > 0:
            let grad_id = self.memory_pool_manager.load(index).pop_back()
            loaded_node.grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(loaded_node.cap)

            loaded_node.data.store(
                1, self.memory_pool.load().load(loaded_node.grad_id.load())
            )
            memset_zero(loaded_node.data.load(1), ceiled_cap)
        else:
            let grad_id = self.get_free_data_id()
            loaded_node.grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(loaded_node.cap)
            let new_grad_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
            if grad_id == self.memory_pool.load().len.load():
                self.memory_pool.load().push_back(new_grad_ptr)
            else:
                self.memory_pool.load().data.load().store(grad_id, new_grad_ptr)

            loaded_node.data.store(
                1, self.memory_pool.load().load(loaded_node.grad_id.load())
            )
            memset_zero(loaded_node.data.load(1), ceiled_cap)

    fn release_data(self, node_ptr: Pointer[Node]) raises:
        var node = node_ptr.load()
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

    fn release_data_forced(self, node_ptr: Pointer[Node]) raises:
        var node = node_ptr.load()
        if node.is_static or node.data_id.load() == -1:
            return
        let index = self.get_index(node.cap)
        let data_id = node.data_id.load()
        self.memory_pool_manager.load(index).push_back(data_id)
        node.data_id.store(-1)
        node.computed_ptr.store(False)
        node.dependencies = node.children.len.load()

    fn release_grad_forced(self, node_ptr: Pointer[Node]) raises:
        let node = node_ptr.load()
        if node.is_static or node.grad_id.load() == -1:
            return
        let index = self.get_index(node.cap)
        let grad_id = node.grad_id.load()
        self.memory_pool_manager.load(index).push_back(grad_id)
        node.grad_id.store(-1)
        node.grad_computed_ptr.store(False)

    fn clear_cache(self, reset_static_nodes: Bool = False) raises:
        let memory_pool = self.memory_pool.load()
        if self.last_node_id.load() != -1:
            let node_ptr = self.nodes.load().load(self.last_node_id.load())
            self.release_data_forced(node_ptr)

        for i in range(self.nodes.load().len.load() - 1):
            if self.nodes.load().load(i) == Pointer[Node].get_null():
                continue
            for j in range(i + 1, self.nodes.load().len.load()):
                if (
                    self.nodes.load().load(i).load().load_id()
                    == self.nodes.load().load(j).load().load_id()
                ):
                    self.nodes.load().store(i, Pointer[Node].get_null())
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
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node == Pointer[Node].get_null():
                continue
            let loaded_node = node.load()

            if loaded_node.is_static:
                if loaded_node.data_id.load() != -1:
                    deletable_data.store(loaded_node.data_id.load(), False)
                if loaded_node.grad_id.load() != -1:
                    deletable_data.store(loaded_node.grad_id.load(), False)

        for i in range(deletable_data.len.load()):
            if (
                deletable_data.load(i)
                and not memory_pool.load(i) == DTypePointer[DType.float32].get_null()
            ):
                memory_pool.load(i).free()
        deletable_data.free()

        for i in range(self.nodes.load().len.load() - 1, -1, -1):
            let node_ptr = self.nodes.load().load(i)
            if node_ptr == Pointer[Node].get_null():
                continue

            var node = node_ptr.load()

            if not node.load_is_static():
                self.free_node_ids.load().push_back(node.load_id())

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
                node_ptr.free()
            else:
                node.children.clear()
                node.parents.clear()
                node.dependencies = 0
                node.id_ptr.store(0)
                node.data_id.store(0)
                node.grad_id.store(0)

    fn clear(self, reset_static_nodes: Bool = False) raises:
        self.clear_cache(reset_static_nodes)

        self.nodes.load().free()
        self.nodes.free()
        self.memory_pool.load().free()
        self.memory_pool.free()

        @unroll
        for i in range(MEMORY_POOL_SIZE):
            self.memory_pool_manager.load(i).free()
        self.memory_pool_manager.free()
        self.free_node_ids.load().free()
        self.free_node_ids.free()
        self.free_data_ids.load().free()
        self.free_data_ids.free()
        self.last_node_id.free()
        self.kernels.free()
        self.forward_order.load().free()
        self.forward_order.free()

    fn forward_recursive(
        self, node_ptr: Pointer[Node], keep_forward_order: Bool = False
    ) raises -> Pointer[Node]:
        let node = node_ptr.load()
        if node.load_computed():
            return node_ptr

        let operator_id = node.operator_id
        if node.load_num_parents() == 1:
            let parent1_ptr = self.forward_recursive(
                self.nodes.load().load(node.load_parent_id(0)),
                keep_forward_order,
            )
            self.get_free_data_ptr(node_ptr)
            self.kernels.load(operator_id).get[0, UNARY_OP]()(node, parent1_ptr.load())
            self.release_data(parent1_ptr)
        else:
            let parent1_ptr = self.forward_recursive(
                self.nodes.load().load(node.load_parent_id(0)),
                keep_forward_order,
            )
            let parent2_ptr = self.forward_recursive(
                self.nodes.load().load(node.load_parent_id(1)),
                keep_forward_order,
            )
            self.get_free_data_ptr(node_ptr)
            self.kernels.load(operator_id).get[1, BINARY_OP]()(
                node, parent1_ptr.load(), parent2_ptr.load()
            )

            self.release_data(parent1_ptr)
            self.release_data(parent2_ptr)

        if keep_forward_order:
            self.forward_order.load().push_back(node.load_id())

        node.computed_ptr.store(True)

        return node_ptr

    fn forward(
        self, node_ptr: Pointer[Node], keep_forward_order: Bool = False
    ) raises -> Pointer[Node]:
        self.last_node_id.store(node_ptr.load().load_id())
        let res = self.forward_recursive(node_ptr, keep_forward_order)
        return res

    fn forward_static(self, node_ptr: Pointer[Node]) raises -> Pointer[Node]:
        self.release_data_forced(node_ptr)

        for i in range(self.nodes.load().len.load()):
            var node = self.nodes.load().load(i).load()
            if node.is_single_ptr.load():
                continue

            if not node.is_static:
                node.computed_ptr.store(False)
                node.grad_id.store(-1)
                node.data_id.store(-1)
            node.dependencies = node.children.len.load()

        _ = self.forward_recursive(node_ptr)

        return self.nodes.load().load(self.last_node_id.load())

    fn forward_recursive_graph_slice(
        self, node_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let node = node_ptr.load()
        if node.computed_ptr.load():
            return node_ptr

        let operator_id = node.operator_id
        if node.load_num_parents() == 1:
            let parent1_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node.parents.load(0))
            )
            self.get_free_data_ptr(node_ptr, True)

            self.kernels.load(operator_id).get[0, UNARY_OP]()(node, parent1_ptr.load())
        else:
            let parent1_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node.parents.load(0))
            )
            let parent2_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node.parents.load(1))
            )

            self.get_free_data_ptr(node_ptr, True)
            self.kernels.load(operator_id).get[1, BINARY_OP]()(
                node, parent1_ptr.load(), parent2_ptr.load()
            )

        node.computed_ptr.store(True)

        return node_ptr

    fn backward_recursive(self, node_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let node = node_ptr.load()
        if node.grad_computed_ptr.load():
            return node_ptr

        for i in range(node.children.len.load()):
            let child_id = node.children.load(i)
            let child_ptr = self.nodes.load().load(child_id)
            _ = self.backward_recursive(child_ptr)

            let grad_operator_id = child_ptr.load().grad_operator_id
            if child_ptr.load().parents.len.load() == 1:
                let parent1_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(0)
                )
                _ = self.forward_recursive_graph_slice(parent1_ptr)

                if parent1_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent1_ptr)

                parent1_ptr.load().grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[0, UNARY_OP]()(
                    child_ptr.load(), parent1_ptr.load()
                )

            else:
                let parent1_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(0)
                )
                let parent2_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(1)
                )

                _ = self.forward_recursive_graph_slice(parent1_ptr)
                _ = self.forward_recursive_graph_slice(parent2_ptr)

                if parent1_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent1_ptr)
                if parent2_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent2_ptr)

                parent1_ptr.load().grad_computed_ptr.store(True)
                parent2_ptr.load().grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[1, BINARY_OP]()(
                    child_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
                )

            if child_ptr.load().load_id() != self.last_node_id.load():
                self.release_data_forced(child_ptr)
            self.release_grad_forced(child_ptr)

        return node_ptr

    fn find_grad_nodes_order(self, node_ptr: Pointer[Node]) raises:
        self.grad_nodes_order.store(Vector[Int]())
        for i in range(self.nodes.load().len.load()):
            var node = self.nodes.load().load(i).load()
            node.tmp_visited = False
        self.grad_nodes_order.load().clear()

        var backward = DynamicVector[Int]()
        backward.push_back(node_ptr.load().load_id())
        var it = 0
        while it < len(backward):
            let currId = backward[it]
            let curr = self.nodes.load().load(currId)
            for i in range(curr.load().parents.len.load()):
                let parId = curr.load().parents.load(i)
                let par = self.nodes.load().load(parId)
                if not par.load().tmp_visited:
                    backward.push_back(parId)
            if curr.load().requires_grad or curr.load().checkpoint:
                self.grad_nodes_order.load().push_back(currId)
            var node = self.nodes.load().load(currId).load()
            node.tmp_visited = True
            it += 1

    fn backward(self, node_ptr: Pointer[Node]) raises:
        self.find_grad_nodes_order(node_ptr)

        self.last_node_id.store(node_ptr.load().load_id())

        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i).load()
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

        self.get_free_grad_ptr(node_ptr)
        let node = node_ptr.load()
        node.fill_grad(1.0)
        node.grad_computed_ptr.store(True)
        for i in range(self.grad_nodes_order.load().len.load()):
            let curr_node_ptr = self.nodes.load().load(
                self.grad_nodes_order.load().load(i)
            )
            _ = self.backward_recursive(curr_node_ptr)

    fn optimizer_step[type: String, learning_rate: Float32](self) raises:
        if type == "sgd":
            SGD[learning_rate].step(self.nodes)
        else:
            warn("Invalid optimizer: " + type + " using sgd\n")
            SGD[learning_rate].step(self.nodes)

    fn copy(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        return self.node[False](
            parent1_ptr.load().shape.copy(),
            True,
            False,
            copy_code,
            Vector[Int](),
            parent1_ptr,
        )

    fn mmul(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        let a_loaded = a.load()
        let b_loaded = b.load()
        let a_dims = a_loaded.num_dims
        let b_dims = b_loaded.num_dims
        shape[len(shape) - 2] = a_loaded.shape.copy().load(a_dims - 2)
        shape[len(shape) - 1] = b_loaded.shape.copy().load(b_dims - 1)
        if a_loaded.shape.load(a_dims - 1) != b_loaded.shape.load(b_dims - 2):
            raise "Shapes don't fit for matrix multiplication. Got shapes: " + str(
                a_loaded.shape.load(a_dims - 1)
            ) + " " + str(b_loaded.shape.load(b_dims - 2))

        let other_params = Vector[Int]()

        return self.node[True](shape, False, False, mmul_code, other_params, a, b)

    fn conv_1d(
        self,
        a: Pointer[Node],
        b: Pointer[Node],
        padding: Int,
        stride: Int,
    ) raises -> Pointer[Node]:
        let batch_size = a.load().shape.load(0)
        let channels = a.load().shape.load(1)
        let input_width = a.load().shape.load(2)
        let kernel_width = b.load().shape.load(1)

        let shape = shape(
            batch_size,
            channels,
            (input_width - kernel_width + 2 * padding) // stride + 1,
        )

        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)

        return self.node[True](shape, False, False, conv1d_code, other_params, a, b)

    fn conv_2d(
        self,
        a: Pointer[Node],
        b: Pointer[Node],
        padding: StaticIntTuple[2],
        stride: StaticIntTuple[2],
    ) raises -> Pointer[Node]:
        let batch_size = a.load().shape.load(0)
        let channels = a.load().shape.load(1)
        let input_width = a.load().shape.load(2)
        let input_height = a.load().shape.load(3)
        let kernel_width = b.load().shape.load(1)
        let kernel_height = b.load().shape.load(2)

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

    fn maxpool_1d(
        self,
        a: Pointer[Node],
        kernel_size: Int,
        stride: Int,
        padding: Int,
    ) raises -> Pointer[Node]:
        let other_params = Vector[Int]()
        other_params.push_back(kernel_size)
        other_params.push_back(stride)
        other_params.push_back(padding)

        let shape = Vector[Int]()
        shape.push_back(a.load().shape.load(0))
        shape.push_back(a.load().shape.load(1))
        shape.push_back(
            (a.load().shape.load(2) - kernel_size + 2 * padding) // stride + 1
        )

        return self.node[True](shape, False, False, maxpool1d_code, other_params, a)

    fn maxpool_2d(
        self,
        a: Pointer[Node],
        kernel_size: StaticIntTuple[2],
        stride: Int,
        padding: Int,
    ) raises -> Pointer[Node]:
        let other_params = Vector[Int]()
        other_params.push_back(kernel_size[0])
        other_params.push_back(kernel_size[1])
        other_params.push_back(stride)
        other_params.push_back(padding)

        let shape = Vector[Int]()
        shape.push_back(a.load().shape.load(0))
        shape.push_back(a.load().shape.load(1))
        shape.push_back(
            (a.load().shape.load(2) - kernel_size[0] + 2 * padding) // stride + 1
        )
        shape.push_back(
            (a.load().shape.load(3) - kernel_size[1] + 2 * padding) // stride + 1
        )

        return self.node[True](shape, False, False, maxpool2d_code, other_params, a)

    fn dropout(
        self, a: Pointer[Node], dropout_rate: Float32, noise_shape: DynamicVector[Int]
    ) raises -> Pointer[Node]:
        return self.node[False](
            a.load().shape.copy(),
            False,
            False,
            dropout_code,
            Vector[Int](),
            a,
        )

    fn reshape(
        self, parent1_ptr: Pointer[Node], shape: Vector[Int]
    ) raises -> Pointer[Node]:
        return self.node[False](
            shape, False, False, reshape_code, Vector[Int](), parent1_ptr
        )

    fn transp(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        return self.node[False](
            parent1_ptr.load().shape.copy().get_transposed(),
            False,
            False,
            transp_code,
            Vector[Int](),
            parent1_ptr,
        )

    fn sum(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        return self.node[False](
            shape(1), False, False, sum_code, Vector[Int](), parent1_ptr
        )

    fn function_general[
        operator_id: Int
    ](self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        return self.node[False](
            parent1_ptr.load().shape.copy(),
            False,
            False,
            operator_id,
            Vector[Int](),
            parent1_ptr,
        )

    fn arithmetic_general[
        operator_id: Int
    ](self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        return self.node[False](
            get_broadcasted_shape_for_ew_op(a, b),
            False,
            False,
            operator_id,
            Vector[Int](),
            a,
            b,
        )

    fn activation_general[
        operator_id: Int,
        arg1: Float32 = 0.0,
    ](self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let other_params = Vector[Int]()
        other_params.push_back(round(arg1 * 1000000.0).to_int())
        return self.node[False](
            parent1_ptr.load().shape.copy(),
            False,
            False,
            operator_id,
            other_params,
            parent1_ptr,
        )

    fn loss_general[
        operator_id: Int
    ](self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]) raises -> Pointer[
        Node
    ]:
        return self.node[False](
            shape(1),
            False,
            False,
            operator_id,
            Vector[Int](),
            parent1_ptr,
            parent2_ptr,
        )
