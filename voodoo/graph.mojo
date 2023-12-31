from memory import memset_zero
from math import log, log2, exp, exp2, ceil, round
from algorithm import vectorize, unswitch

from .kernels import op_tuple, unary_op, binary_op, Kernels
from .node import Node
from .utils import Vector, get_broadcasted_shape_for_ew_op
from .utils.shape import shape
from .cpu_kernels.optimizers import *

alias nelts = simdwidthof[DType.float32]()
alias memory_pool_size = 30


# TODO: Could combine more unary / binary functions into one main caller (See activations / losses)
@register_passable("trivial")
struct Graph:
    var nodes: Pointer[Vector[Pointer[Node]]]
    var memory_pool: Pointer[Vector[DTypePointer[DType.float32]]]
    var memory_pool_manager: Pointer[Vector[Int]]
    var free_node_ids: Pointer[Vector[Int]]
    var free_data_ids: Pointer[Vector[Int]]
    var last_node_id: Pointer[Int]
    var kernels: Pointer[op_tuple]
    var forward_order: Pointer[Vector[Int]]
    var grad_nodes_order: Pointer[Vector[Int]]
    var compiled: Pointer[Bool]

    # TODO: Figure out how to make Kernels compile time constant, may need to rewrite the struct to not be a pointer
    fn __init__() -> Self:
        let nodes = Pointer[Vector[Pointer[Node]]].alloc(1)
        nodes.store(Vector[Pointer[Node]]())

        let memory_pool = Pointer[Vector[DTypePointer[DType.float32]]].alloc(1)
        memory_pool.store(Vector[DTypePointer[DType.float32]]())

        let memory_pool_manager = Pointer[Vector[Int]].alloc(memory_pool_size)

        @unroll
        for i in range(memory_pool_size):
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

        let compiled = Pointer[Bool].alloc(1)
        compiled.store(False)

        return Graph {
            nodes: nodes,
            memory_pool: memory_pool,
            memory_pool_manager: memory_pool_manager,
            free_node_ids: free_node_ids,
            free_data_ids: free_data_ids,
            last_node_id: last_node_id,
            kernels: Kernels().kernels,
            forward_order: forward_order,
            grad_nodes_order: grad_nodes_order,
            compiled: compiled,
        }

    fn print_memory_pool_manager(self) raises:
        @unroll
        for i in range(memory_pool_size):
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

    fn node(
        self,
        shape: Vector[Int],
        is_static: Bool,
        is_single: Bool,
        checkpoint: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parent_ptrs: Pointer[Node],
    ) raises -> Pointer[Node]:
        let node = Node(self.get_free_node_id(), shape, is_static)
        node.operator_id_ptr.store(operator_id)
        node.checkpoint_ptr.store(checkpoint)
        node.is_single_ptr.store(is_single)
        node.grad_operator_id_ptr.store(operator_id + 1)
        node.other_params_ptr.store(other_params.copy())
        let node_ptr = Pointer[Node].alloc(1)
        node_ptr.store(node)

        for i in range(len(parent_ptrs)):
            node.add_parent(parent_ptrs[i].load().load_id())
            parent_ptrs[i].load().add_child(node.load_id())
            parent_ptrs[i].load().incr_dependencies()

        self.get_free_data_ptr(node_ptr)

        for i in range(len(parent_ptrs)):
            if parent_ptrs[i].load().dependencies_ptr.load() == 0:
                _ = self.forward_recursive(parent_ptrs[i])

        let node_id = node_ptr.load().load_id()
        if node_id < self.nodes.load().len.load():
            self.nodes.load().store(node_id, node_ptr)
        else:
            self.nodes.load().push_back(node_ptr)

        return node_ptr

    fn node(
        self,
        shape: DynamicVector[Int],
        is_static: Bool,
        is_single: Bool,
        checkpoint: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parent_ptrs: Pointer[Node],
    ) raises -> Pointer[Node]:
        let _shape = Vector[Int]()
        for i in range(len(shape)):
            _shape.push_back(shape[i])
        let node = Node(self.get_free_node_id(), _shape, is_static)
        node.checkpoint_ptr.store(checkpoint)
        node.is_single_ptr.store(is_single)
        node.operator_id_ptr.store(operator_id)
        node.grad_operator_id_ptr.store(operator_id + 1)
        node.other_params_ptr.store(other_params.copy())
        let node_ptr = Pointer[Node].alloc(1)
        node_ptr.store(node)

        for i in range(len(parent_ptrs)):
            node.add_parent(parent_ptrs[i].load().load_id())
            parent_ptrs[i].load().add_child(node.load_id())
            parent_ptrs[i].load().incr_dependencies()

        self.get_free_data_ptr(node_ptr)

        for i in range(len(parent_ptrs)):
            if parent_ptrs[i].load().dependencies_ptr.load() == 0:
                _ = self.forward_recursive(parent_ptrs[i])

        let node_id = node_ptr.load().load_id()
        if node_id < self.nodes.load().len.load():
            self.nodes.load().store(node_id, node_ptr)
        else:
            self.nodes.load().push_back(node_ptr)

        return node_ptr

    fn get_free_data_ptr(self, node: Pointer[Node], unique: Bool = False) raises:
        if node.load().data_id.load() != -1:
            return

        var idx = -1
        for i in range(node.load().parents_ptr.load().len.load()):
            let ind = node.load().parents_ptr.load().load(i)
            let parent = self.nodes.load().load(node.load().load_parent_id(i))
            if (
                self.load_ceiled_cap(parent.load().cap_ptr.load())
                == self.load_ceiled_cap(node.load().cap_ptr.load())
                and parent.load().dependencies_ptr.load() == 1
                and not parent.load().is_static_ptr.load()
                and not node.load().is_static_ptr.load()
                and not parent.load().checkpoint_ptr.load()
                and not node.load().checkpoint_ptr.load()
                and not unique
                and not parent.load().is_single_ptr.load()
                and not node.load().is_single_ptr.load()
            ):
                node.load().data_id.store(parent.load().data_id.load())
                node.load().data.store(
                    0, self.memory_pool.load().load(node.load().data_id.load())
                )
                idx = i
                break

        for i in range(node.load().parents_ptr.load().len.load()):
            if i == idx:
                continue
            else:
                let parent = self.nodes.load().load(node.load().load_parent_id(i))
                parent.load().decr_dependencies()

        if idx == -1:
            let index = self.get_index(node.load().cap_ptr.load())
            if self.memory_pool_manager.load(index).len.load() > 0:
                let data_id = self.memory_pool_manager.load(index).pop_back()
                node.load().data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load())

                node.load().data.store(
                    0, self.memory_pool.load().load(node.load().data_id.load())
                )
                memset_zero(node.load().data.load(0), ceiled_cap)
            else:
                let data_id = self.get_free_data_id()
                node.load().data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load() + 1)
                let new_data_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
                if data_id == self.memory_pool.load().len.load():
                    self.memory_pool.load().push_back(new_data_ptr)
                else:
                    self.memory_pool.load().data.load().store(data_id, new_data_ptr)

                node.load().data.store(
                    0, self.memory_pool.load().load(node.load().data_id.load())
                )
                memset_zero(node.load().data.load(0), ceiled_cap)

    fn get_free_grad_ptr(self, node: Pointer[Node]) raises:
        if node.load().grad_id.load() != -1:
            return

        let index = self.get_index(node.load().cap_ptr.load())
        if self.memory_pool_manager.load(index).len.load() > 0:
            let grad_id = self.memory_pool_manager.load(index).pop_back()
            node.load().grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load())

            node.load().data.store(
                1, self.memory_pool.load().load(node.load().grad_id.load())
            )
            memset_zero(node.load().data.load(1), ceiled_cap)
        else:
            let grad_id = self.get_free_data_id()
            node.load().grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load())
            let new_grad_ptr = DTypePointer[DType.float32].alloc(ceiled_cap)
            if grad_id == self.memory_pool.load().len.load():
                self.memory_pool.load().push_back(new_grad_ptr)
            else:
                self.memory_pool.load().data.load().store(grad_id, new_grad_ptr)

            node.load().data.store(
                1, self.memory_pool.load().load(node.load().grad_id.load())
            )
            memset_zero(node.load().data.load(1), ceiled_cap)

    fn release_data(self, node_ptr: Pointer[Node]) raises:
        if (
            node_ptr.load().is_static_ptr.load()
            or node_ptr.load().checkpoint_ptr.load()
            or node_ptr.load().is_single_ptr.load()
            or node_ptr.load().data_id.load() == -1
        ):
            return

        if node_ptr.load().dependencies_ptr.load() == 0:
            let index = self.get_index(node_ptr.load().cap_ptr.load())
            let data_id = node_ptr.load().data_id.load()
            self.memory_pool_manager.load(index).push_back(data_id)
            node_ptr.load().data_id.store(-1)
            node_ptr.load().dependencies_ptr.store(
                node_ptr.load().children_ptr.load().len.load()
            )
            node_ptr.load().computed_ptr.store(False)

    fn release_data_forced(self, node_ptr: Pointer[Node]) raises:
        if node_ptr.load().is_static_ptr.load() or node_ptr.load().data_id.load() == -1:
            return
        let index = self.get_index(node_ptr.load().cap_ptr.load())
        let data_id = node_ptr.load().data_id.load()
        self.memory_pool_manager.load(index).push_back(data_id)
        node_ptr.load().data_id.store(-1)
        node_ptr.load().computed_ptr.store(False)
        node_ptr.load().dependencies_ptr.store(
            node_ptr.load().children_ptr.load().len.load()
        )

    fn release_grad_forced(self, node_ptr: Pointer[Node]) raises:
        if node_ptr.load().is_static_ptr.load() or node_ptr.load().grad_id.load() == -1:
            return
        let index = self.get_index(node_ptr.load().cap_ptr.load())
        let grad_id = node_ptr.load().grad_id.load()
        self.memory_pool_manager.load(index).push_back(grad_id)
        node_ptr.load().grad_id.store(-1)
        node_ptr.load().grad_computed_ptr.store(False)

    fn clear_cache(self, reset_static_nodes: Bool = False) raises:
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

        for i in range(self.memory_pool.load().len.load()):
            let array = self.memory_pool.load().load(i)
            for j in range(i + 1, self.memory_pool.load().len.load()):
                let other = self.memory_pool.load().load(j)
                if array == other:
                    self.memory_pool.load().store(
                        i, DTypePointer[DType.float32].get_null()
                    )

        let deletable_data = Vector[Bool](self.memory_pool.load().len.load())
        for i in range(self.memory_pool.load().len.load()):
            deletable_data.store(i, True)
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node == Pointer[Node].get_null():
                continue

            if node.load().is_static_ptr.load():
                if node.load().data_id.load() != -1:
                    deletable_data.store(node.load().data_id.load(), False)
                if node.load().grad_id.load() != -1:
                    deletable_data.store(node.load().grad_id.load(), False)

        for i in range(deletable_data.len.load()):
            if (
                deletable_data.load(i)
                and not self.memory_pool.load().load(i)
                == DTypePointer[DType.float32].get_null()
            ):
                self.memory_pool.load().load(i).free()
        deletable_data.free()

        for i in range(self.nodes.load().len.load() - 1, -1, -1):
            let node_ptr = self.nodes.load().load(i)
            if node_ptr == Pointer[Node].get_null():
                continue

            if not node_ptr.load().load_is_static():
                self.free_node_ids.load().push_back(node_ptr.load().load_id())

                node_ptr.load().id_ptr.free()
                node_ptr.load().data_id.free()
                node_ptr.load().grad_id.free()
                node_ptr.load().data.free()
                node_ptr.load().parents_ptr.load().free()
                node_ptr.load().parents_ptr.free()
                node_ptr.load().children_ptr.load().free()
                node_ptr.load().children_ptr.free()
                node_ptr.load().dependencies_ptr.free()
                node_ptr.load().is_static_ptr.free()
                node_ptr.load().computed_ptr.free()
                node_ptr.load().grad_computed_ptr.free()
                node_ptr.load().operator_id_ptr.free()
                node_ptr.load().grad_operator_id_ptr.free()
                node_ptr.load().requires_grad_ptr.free()
                node_ptr.load().tmp_visited_ptr.free()
                node_ptr.load().checkpoint_ptr.free()

                node_ptr.load().cap_ptr.free()
                node_ptr.load().num_dims_ptr.free()
                node_ptr.load().shape_ptr.load().free()
                node_ptr.load().shape_ptr.free()
                node_ptr.load().strides_ptr.load().free()
                node_ptr.load().strides_ptr.free()
                node_ptr.load().other_params_ptr.load().free()
                node_ptr.load().other_params_ptr.free()

                node_ptr.free()
            else:
                node_ptr.load().children_ptr.load().clear()
                node_ptr.load().parents_ptr.load().clear()
                node_ptr.load().dependencies_ptr.store(0)
                node_ptr.load().id_ptr.store(0)
                node_ptr.load().data_id.store(0)
                node_ptr.load().grad_id.store(0)

    fn clear(self, reset_static_nodes: Bool = False) raises:
        self.clear_cache(reset_static_nodes)

        self.nodes.load().free()
        self.nodes.free()
        self.memory_pool.load().free()
        self.memory_pool.free()

        @unroll
        for i in range(memory_pool_size):
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
        self.compiled.free()

    fn forward_recursive(
        self, node_ptr: Pointer[Node], keep_forward_order: Bool = False
    ) raises -> Pointer[Node]:
        if node_ptr.load().load_computed():
            return node_ptr

        let operator_id = node_ptr.load().operator_id_ptr.load()
        if node_ptr.load().load_num_parents() == 1:
            let parent1_ptr = self.forward_recursive(
                self.nodes.load().load(node_ptr.load().load_parent_id(0)),
                keep_forward_order,
            )
            self.get_free_data_ptr(node_ptr)
            self.kernels.load(operator_id).get[1, unary_op]()(
                node_ptr.load(), parent1_ptr.load()
            )
            self.release_data(parent1_ptr)
        else:
            let parent1_ptr = self.forward_recursive(
                self.nodes.load().load(node_ptr.load().load_parent_id(0)),
                keep_forward_order,
            )
            let parent2_ptr = self.forward_recursive(
                self.nodes.load().load(node_ptr.load().load_parent_id(1)),
                keep_forward_order,
            )
            self.get_free_data_ptr(node_ptr)
            self.kernels.load(operator_id).get[2, binary_op]()(
                node_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
            )

            self.release_data(parent1_ptr)
            self.release_data(parent2_ptr)

        if keep_forward_order:
            self.forward_order.load().push_back(node_ptr.load().load_id())

        node_ptr.load().computed_ptr.store(True)

        return node_ptr

    fn forward(
        self, node_ptr: Pointer[Node], keep_forward_order: Bool = False
    ) raises -> Pointer[Node]:
        self.last_node_id.store(node_ptr.load().load_id())
        self.compiled.store(False)
        let res = self.forward_recursive(node_ptr, keep_forward_order)
        return res

    fn forward_static(self, node_ptr: Pointer[Node]) raises -> Pointer[Node]:
        self.release_data_forced(node_ptr)

        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node.load().is_single_ptr.load():
                continue

            if not node.load().is_static_ptr.load():
                node.load().computed_ptr.store(False)
                node.load().grad_id.store(-1)
                node.load().data_id.store(-1)
            node.load().dependencies_ptr.store(
                node.load().children_ptr.load().len.load()
            )

        _ = self.forward_recursive(node_ptr)

        return self.nodes.load().load(self.last_node_id.load())

    fn forward_recursive_graph_slice(
        self, node_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        if node_ptr.load().computed_ptr.load():
            return node_ptr

        let operator_id = node_ptr.load().operator_id_ptr.load()
        if node_ptr.load().load_num_parents() == 1:
            let parent1_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node_ptr.load().parents_ptr.load().load(0))
            )
            self.get_free_data_ptr(node_ptr, True)

            self.kernels.load(operator_id).get[1, unary_op]()(
                node_ptr.load(), parent1_ptr.load()
            )
        else:
            let parent1_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node_ptr.load().parents_ptr.load().load(0))
            )
            let parent2_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node_ptr.load().parents_ptr.load().load(1))
            )

            self.get_free_data_ptr(node_ptr, True)
            self.kernels.load(operator_id).get[2, binary_op]()(
                node_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
            )

        node_ptr.load().computed_ptr.store(True)

        return node_ptr

    fn backward_recursive(self, node_ptr: Pointer[Node]) raises -> Pointer[Node]:
        if node_ptr.load().grad_computed_ptr.load():
            return node_ptr

        for i in range(node_ptr.load().children_ptr.load().len.load()):
            let child_id = node_ptr.load().children_ptr.load().load(i)
            let child_ptr = self.nodes.load().load(child_id)
            _ = self.backward_recursive(child_ptr)

            let grad_operator_id = child_ptr.load().grad_operator_id_ptr.load()
            if child_ptr.load().parents_ptr.load().len.load() == 1:
                let parent1_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(0)
                )
                _ = self.forward_recursive_graph_slice(parent1_ptr)

                if parent1_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent1_ptr)

                parent1_ptr.load().grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[1, unary_op]()(
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

                self.kernels.load(grad_operator_id).get[2, binary_op]()(
                    child_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
                )

            if child_ptr.load().load_id() != self.last_node_id.load():
                self.release_data_forced(child_ptr)
            self.release_grad_forced(child_ptr)

        return node_ptr

    fn find_grad_nodes_order(self, node_ptr: Pointer[Node]) raises:
        self.grad_nodes_order.store(Vector[Int]())
        for i in range(self.nodes.load().len.load()):
            self.nodes.load().load(i).load().tmp_visited_ptr.store(False)
        self.grad_nodes_order.load().clear()

        var backward = DynamicVector[Int]()
        backward.push_back(node_ptr.load().load_id())
        var it = 0
        while it < len(backward):
            let currId = backward[it]
            let curr = self.nodes.load().load(currId)
            for i in range(curr.load().parents_ptr.load().len.load()):
                let parId = curr.load().parents_ptr.load().load(i)
                let par = self.nodes.load().load(parId)
                if not par.load().tmp_visited_ptr.load():
                    backward.push_back(parId)
            if (
                curr.load().requires_grad_ptr.load()
                or curr.load().checkpoint_ptr.load()
            ):
                self.grad_nodes_order.load().push_back(currId)
            self.nodes.load().load(currId).load().tmp_visited_ptr.store(True)
            it += 1

    fn backward(self, node_ptr: Pointer[Node]) raises:
        self.find_grad_nodes_order(node_ptr)

        self.last_node_id.store(node_ptr.load().load_id())

        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            node.load().grad_computed_ptr.store(False)

            if (
                node.load().is_single_ptr.load()
                or node.load().load_id() == self.last_node_id.load()
            ):
                continue

            if not node.load().is_static_ptr.load():
                node.load().grad_id.store(-1)
                if not node.load().checkpoint_ptr.load():
                    node.load().computed_ptr.store(False)
                    node.load().data_id.store(-1)
            else:
                if node.load().grad_id.load() != -1:
                    memset_zero(
                        node.load().data.load(1),
                        self.load_ceiled_cap(node.load().cap_ptr.load()),
                    )

        self.get_free_grad_ptr(node_ptr)
        node_ptr.load().fill_grad(1.0)
        node_ptr.load().grad_computed_ptr.store(True)
        for i in range(self.grad_nodes_order.load().len.load()):
            let curr_node_ptr = self.nodes.load().load(
                self.grad_nodes_order.load().load(i)
            )
            _ = self.backward_recursive(curr_node_ptr)

    fn optimizer_step[type: String, learning_rate: Float32](self) raises:
        # TODO: Switch to Dict
        @parameter
        if type == "sgd":
            SGD.step[learning_rate](self.nodes)
        elif type == "adafactor":
            Adafactor.step[learning_rate](self.nodes)
        elif type == "adam":
            Adam.step[learning_rate](self.nodes)

    fn copy(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = copy_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, True, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn mmul(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = mmul_code
        let checkpoint = True
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        shape[len(shape) - 2] = (
            a.load().shape_ptr.load().copy().load(a.load().num_dims_ptr.load() - 2)
        )
        shape[len(shape) - 1] = (
            b.load().shape_ptr.load().copy().load(b.load().num_dims_ptr.load() - 1)
        )
        if a.load().shape_ptr.load().load(
            a.load().num_dims_ptr.load() - 1
        ) != b.load().shape_ptr.load().load(b.load().num_dims_ptr.load() - 2):
            raise "Shapes don't fit for matrix multiplication. Got shapes: " + str(
                a.load().shape_ptr.load().load(a.load().num_dims_ptr.load() - 1)
            ) + " " + str(
                b.load().shape_ptr.load().load(b.load().num_dims_ptr.load() - 2)
            )
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn conv_2d(
        self, a: Pointer[Node], b: Pointer[Node], padding: Int, stride: Int
    ) raises -> Pointer[Node]:
        let a_num_dims = a.load().num_dims_ptr.load()
        let b_num_dims = b.load().num_dims_ptr.load()
        let batch_size = a.load().shape_ptr.load().load(0)
        let in_channels = a.load().shape_ptr.load().load(1)
        let width = a.load().shape_ptr.load().load(2)
        let height = a.load().shape_ptr.load().load(3)
        let out_channels = b.load().shape_ptr.load().load(0)
        if in_channels != out_channels:
            raise "Channels don't fit for 2D Convolution. Got channels: " + str(
                in_channels
            ) + " " + str(out_channels)
        let kernel_width = b.load().shape_ptr.load().load(2)
        let kernel_height = b.load().shape_ptr.load().load(3)
        let shape = shape(
            batch_size,
            out_channels,
            (width - kernel_width + 2 * padding) // stride + 1,
            (height - kernel_height + 2 * padding) // stride + 1,
        )
        let operator_id = 58
        let checkpoint = True
        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)
        return self.node(
            shape, True, False, checkpoint, operator_id, other_params, a, b
        )

    fn max_pool_2d(
        self,
        a: Pointer[Node],
        kernel_width: Int,
        kernel_height: Int,
        stride: Int,
        padding: Int,
    ) raises -> Pointer[Node]:
        let new_shape = shape(
            a.load().shape_ptr.load().load(0),
            a.load().shape_ptr.load().load(1),
            (a.load().shape_ptr.load().load(2) - kernel_width + 2 * padding) // stride
            + 1,
            (a.load().shape_ptr.load().load(3) - kernel_height + 2 * padding) // stride
            + 1,
        )
        let operator_id = mpool2dd_code
        let checkpoint = False
        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)
        other_params.push_back(kernel_width)
        other_params.push_back(kernel_height)

        let b = self.node(
            new_shape, False, False, checkpoint, operator_id, other_params, a
        )

        return b

    fn dropout(
        self, a: Pointer[Node], dropout_rate: Float32, noise_shape: DynamicVector[Int]
    ) raises -> Pointer[Node]:
        let operator_id = dropout_code
        let checkpoint = False
        let shape = a.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(round(dropout_rate * 1000000.0).to_int())
        for i in range(len(noise_shape)):
            other_params.push_back(noise_shape[i])
        return self.node(shape, False, False, checkpoint, operator_id, other_params, a)

    fn reshape(
        self, parent1_ptr: Pointer[Node], shape: Vector[Int]
    ) raises -> Pointer[Node]:
        let operator_id = reshape_code
        let checkpoint = False
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn transp(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = transp_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        shape.store(
            shape.len.load() - 2,
            parent1_ptr.load().shape_ptr.load().load(shape.len.load() - 1),
        )
        shape.store(
            shape.len.load() - 1,
            parent1_ptr.load().shape_ptr.load().load(shape.len.load() - 2),
        )
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sum(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = sum_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn function_general[
        operator_id: Int
    ](self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn arithmetic_general[
        operator_id: Int
    ](self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn activation_general[
        operator_id: Int,
        arg1: Float32 = 0.0,
    ](self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(round(arg1 * 1000000.0).to_int())
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn loss_general[
        operator_id: Int
    ](self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]) raises -> Pointer[
        Node
    ]:
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape,
            False,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )
