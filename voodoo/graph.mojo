from memory import memset_zero
from math import log, log2, exp, exp2, ceil, round
from algorithm import vectorize, unswitch

from .node import Node
from .utils import Vector, get_broadcasted_shape_for_ew_op
from .utils.shape import shape

from .cpu_kernels.operations import *
from .cpu_kernels.binary_operations import *
from .cpu_kernels.arithmetic import *
from .cpu_kernels.binary_arithmetic import *
from .cpu_kernels.activations import *
from .cpu_kernels.losses import *
from .cpu_kernels.optimizers import *

alias VectorF32 = DTypePointer[DType.float32]
alias VectorInt = Vector[Int]
alias DTVector = Vector[VectorF32]
alias NodeVector = Vector[Pointer[Node]]
alias nelts = simdwidthof[DType.float32]()

alias unary_op = fn (b: Node, a: Node) -> None
alias binary_op = fn (c: Node, a: Node, b: Node) -> None
alias view_op = fn (b: Node, a: Node) -> None
alias reduce_op = fn (c: Node, a: Node, b: Node) -> None
alias op_tuple = Tuple[StringRef, unary_op, binary_op, view_op, reduce_op]

alias memory_pool_size = 30


fn _u(b: Node, a: Node):
    ...


fn _b(c: Node, a: Node, b: Node):
    ...


fn _v(b: Node, a: Node):
    ...


fn _r(c: Node, a: Node, b: Node):
    ...


# TODO: Could combine more unary / binary functions into one main caller (See activations / losses)
@register_passable("trivial")
struct Graph:
    var nodes: Pointer[NodeVector]
    var memory_pool: Pointer[DTVector]
    var memory_pool_manager: Pointer[VectorInt]
    var free_node_ids: Pointer[VectorInt]
    var free_data_ids: Pointer[VectorInt]
    var last_node_id: Pointer[Int]
    var kernels: Pointer[op_tuple]
    var forward_order: Pointer[VectorInt]
    var grad_nodes_order: Pointer[VectorInt]
    var compiled: Pointer[Bool]

    fn __init__() -> Self:
        let nodes = Pointer[NodeVector].alloc(1)
        nodes.store(NodeVector())

        let memory_pool = Pointer[DTVector].alloc(1)
        memory_pool.store(DTVector())

        let memory_pool_manager = Pointer[VectorInt].alloc(memory_pool_size)

        @unroll
        for i in range(memory_pool_size):
            memory_pool_manager.store(i, VectorInt())

        let free_node_ids = Pointer[VectorInt].alloc(1)
        free_node_ids.store(VectorInt())

        let free_data_ids = Pointer[VectorInt].alloc(1)
        free_data_ids.store(VectorInt())

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        let kernels = Pointer[op_tuple].alloc(120)
        kernels.store(cos_code, op_tuple("cos", Cos.fw, _b, _v, _r))
        kernels.store(bwcos_code, op_tuple("bwcos", Cos.bw, _b, _v, _r))
        kernels.store(sin_code, op_tuple("sin", Sin.fw, _b, _v, _r))
        kernels.store(bwsin_code, op_tuple("bwsin", Sin.bw, _b, _v, _r))
        kernels.store(tan_code, op_tuple("tan", Tan.fw, _b, _v, _r))
        kernels.store(bwtan_code, op_tuple("bwtan", Tan.bw, _b, _v, _r))
        kernels.store(acos_code, op_tuple("acos", Acos.fw, _b, _v, _r))
        kernels.store(bwacos_code, op_tuple("bwacos", Acos.bw, _b, _v, _r))
        kernels.store(asin_code, op_tuple("asin", Asin.fw, _b, _v, _r))
        kernels.store(bwasin_code, op_tuple("bwasin", Asin.bw, _b, _v, _r))
        kernels.store(atan_code, op_tuple("atan", Atan.fw, _b, _v, _r))
        kernels.store(bwatan_code, op_tuple("bwatan", Atan.bw, _b, _v, _r))
        kernels.store(cosh_code, op_tuple("cosh", Cosh.fw, _b, _v, _r))
        kernels.store(bwcos_code, op_tuple("bwcosh", Cosh.bw, _b, _v, _r))
        kernels.store(sinh_code, op_tuple("sinh", Sinh.fw, _b, _v, _r))
        kernels.store(bwsin_code, op_tuple("bwsinh", Sinh.bw, _b, _v, _r))
        kernels.store(log_code, op_tuple("log", Log.fw, _b, _v, _r))
        kernels.store(bwlog_code, op_tuple("bwlog", Log.bw, _b, _v, _r))
        kernels.store(log2_code, op_tuple("log2", Log2.fw, _b, _v, _r))
        kernels.store(bwlog2_code, op_tuple("bwlog", Log2.bw, _b, _v, _r))
        kernels.store(exp2_code, op_tuple("exp2", Exp2.fw, _b, _v, _r))
        kernels.store(bwexp2_code, op_tuple("bwexp2", Exp2.bw, _b, _v, _r))
        kernels.store(sqrt_code, op_tuple("sqrt", Sqrt.fw, _b, _v, _r))
        kernels.store(bwsqrt_code, op_tuple("bwsqrt", Sqrt.bw, _b, _v, _r))
        kernels.store(abs_code, op_tuple("abs", Abs.fw, _b, _v, _r))
        kernels.store(bwabs_code, op_tuple("bwabs", Abs.bw, _b, _v, _r))
        kernels.store(copy_code, op_tuple("copy", Copy.fw, _b, _v, _r))
        kernels.store(bwcopy_code, op_tuple("bwcopy", Copy.bw, _b, _v, _r))

        kernels.store(add_code, op_tuple("add", _u, Add.fw, _v, _r))
        kernels.store(bwadd_code, op_tuple("bwadd", _u, Add.bw, _v, _r))
        kernels.store(sub_code, op_tuple("sub", _u, Sub.fw, _v, _r))
        kernels.store(bwsub_code, op_tuple("bwsub", _u, Sub.bw, _v, _r))
        kernels.store(mul_code, op_tuple("mul", _u, Mul.fw, _v, _r))
        kernels.store(bwmul_code, op_tuple("bwmul", _u, Mul.bw, _v, _r))
        kernels.store(div_code, op_tuple("div", _u, Div.fw, _v, _r))
        kernels.store(bwdiv_code, op_tuple("bwdiv", _u, Div.bw, _v, _r))
        kernels.store(pow_code, op_tuple("pow", _u, Pow.fw, _v, _r))
        kernels.store(bwpow_code, op_tuple("bwpow", _u, Pow.bw, _v, _r))
        kernels.store(mmul_code, op_tuple("mmul", _u, MMul.fw, _v, _r))
        kernels.store(bwmmul_code, op_tuple("bwmmul", _u, MMul.bw, _v, _r))
        kernels.store(reshape_code, op_tuple("reshape", Reshape.fw, _b, _v, _r))
        kernels.store(bwreshape_code, op_tuple("bwreshape", Reshape.bw, _b, _v, _r))
        kernels.store(transp_code, op_tuple("transp", Transpose.fw, _b, _v, _r))
        kernels.store(bwtransp_code, op_tuple("bwtransp", Transpose.bw, _b, _v, _r))
        kernels.store(sum_code, op_tuple("sum", Sum.fw, _b, _v, _r))
        kernels.store(bwsum_code, op_tuple("bwsum", Sum.bw, _b, _v, _r))
        kernels.store(conv2d_code, op_tuple("conv2d", _u, Conv2D.fw, _v, _r))
        kernels.store(bwconv2d_code, op_tuple("bwconv2d", _u, Conv2D.bw, _v, _r))

        kernels.store(mpool2dd_code, op_tuple("mpool2dd", MaxPool2D.fw, _b, _v, _r))
        kernels.store(bwmpool2d_code, op_tuple("bwmpool2d", MaxPool2D.bw, _b, _v, _r))
        kernels.store(elu_code, op_tuple("elu", Elu.fw, _b, _v, _r))
        kernels.store(bwelu_code, op_tuple("bwelu", Elu.bw, _b, _v, _r))
        kernels.store(exp_code, op_tuple("exp", Exp.fw, _b, _v, _r))
        kernels.store(bwexp_code, op_tuple("bwexp", Exp.bw, _b, _v, _r))
        kernels.store(gelu_code, op_tuple("gelu", Gelu.fw, _b, _v, _r))
        kernels.store(bwgelu_code, op_tuple("bwgelu", Gelu.bw, _b, _v, _r))
        kernels.store(h_sig_code, op_tuple("h_sig", HardSigmoid.fw, _b, _v, _r))
        kernels.store(bwh_sig_code, op_tuple("bwh_sig", HardSigmoid.bw, _b, _v, _r))
        kernels.store(linear_code, op_tuple("linear", Linear.fw, _b, _v, _r))
        kernels.store(bwlinear_code, op_tuple("bwlinear", Linear.bw, _b, _v, _r))
        kernels.store(mish_code, op_tuple("mish", Mish.fw, _b, _v, _r))
        kernels.store(bwmish_code, op_tuple("bwmish", Mish.bw, _b, _v, _r))
        kernels.store(relu_code, op_tuple("relu", ReLu.fw, _b, _v, _r))
        kernels.store(bwrelu_code, op_tuple("bwrelu", ReLu.bw, _b, _v, _r))
        kernels.store(selu_code, op_tuple("selu", Selu.fw, _b, _v, _r))
        kernels.store(bwselu_code, op_tuple("bwselu", Selu.bw, _b, _v, _r))
        kernels.store(sig_code, op_tuple("sig", Sigmoid.fw, _b, _v, _r))
        kernels.store(bwsig_code, op_tuple("bwsig", Sigmoid.bw, _b, _v, _r))
        kernels.store(softmax_code, op_tuple("softmax", Softmax.fw, _b, _v, _r))
        kernels.store(bwsoftmax_code, op_tuple("bwsoftmax", Softmax.bw, _b, _v, _r))
        kernels.store(softplus_code, op_tuple("softplus", Softplus.fw, _b, _v, _r))
        kernels.store(bwsoftplus_code, op_tuple("bwsoftplus", Softplus.bw, _b, _v, _r))
        kernels.store(softsign_code, op_tuple("softsign", Softsign.fw, _b, _v, _r))
        kernels.store(bwsoftsign_code, op_tuple("bwsoftsign", Softsign.bw, _b, _v, _r))
        kernels.store(swish_code, op_tuple("swish", Swish.fw, _b, _v, _r))
        kernels.store(bwswish_code, op_tuple("bwswish", Swish.bw, _b, _v, _r))
        kernels.store(tanh_code, op_tuple("tanh", Tanh.fw, _b, _v, _r))
        kernels.store(bwtanh_code, op_tuple("bwtanh", Tanh.bw, _b, _v, _r))
        kernels.store(lrelu_code, op_tuple("fwlrelu", LeakyReLu.fw, _b, _v, _r))
        kernels.store(bwlrelu_code, op_tuple("bwlrelu", LeakyReLu.bw, _b, _v, _r))

        kernels.store(mae_code, op_tuple("mae", _u, MAE.fw, _v, _r))
        kernels.store(bwmae_code, op_tuple("bwmae", _u, MAE.bw, _v, _r))
        kernels.store(mape_code, op_tuple("mape", _u, MAPE.fw, _v, _r))
        kernels.store(bwmape_code, op_tuple("bwmape", _u, MAPE.bw, _v, _r))
        kernels.store(mse_code, op_tuple("mse", _u, MSE.fw, _v, _r))
        kernels.store(bwmse_code, op_tuple("bwmse", _u, MSE.bw, _v, _r))
        kernels.store(msle_code, op_tuple("msle", _u, MSLE.fw, _v, _r))
        kernels.store(bwmsle_code, op_tuple("bwmsle", _u, MSLE.bw, _v, _r))
        kernels.store(bce_code, op_tuple("bce", _u, BCE.fw, _v, _r))
        kernels.store(bwbce_code, op_tuple("bwbce", _u, BCE.bw, _v, _r))
        kernels.store(cce_code, op_tuple("cce", _u, CCE.fw, _v, _r))
        kernels.store(bwcce_code, op_tuple("bwcce", _u, CCE.bw, _v, _r))
        kernels.store(cfce_code, op_tuple("cfce", _u, CFCE.fw, _v, _r))
        kernels.store(bwcfce_code, op_tuple("bwcfce", _u, CFCE.bw, _v, _r))
        kernels.store(dropout_code, op_tuple("dropout", Dropout.fw, _b, _v, _r))
        kernels.store(bwdropout_code, op_tuple("bwdropout", Dropout.bw, _b, _v, _r))

        let forward_order = Pointer[VectorInt].alloc(1)
        forward_order.store(VectorInt())

        let grad_nodes_order = Pointer[VectorInt].alloc(1)
        grad_nodes_order.store(VectorInt())

        let backward_order = Pointer[VectorInt].alloc(1)
        backward_order.store(VectorInt())

        let compiled = Pointer[Bool].alloc(1)
        compiled.store(False)

        return Graph {
            nodes: nodes,
            memory_pool: memory_pool,
            memory_pool_manager: memory_pool_manager,
            free_node_ids: free_node_ids,
            free_data_ids: free_data_ids,
            last_node_id: last_node_id,
            kernels: kernels,
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
                let new_data_ptr = VectorF32.alloc(ceiled_cap)
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
            let new_grad_ptr = VectorF32.alloc(ceiled_cap)
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
                    self.memory_pool.load().store(i, VectorF32.get_null())

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
                and not self.memory_pool.load().load(i) == VectorF32.get_null()
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
