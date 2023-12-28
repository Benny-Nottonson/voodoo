from memory import memset_zero
from math import log, log2, exp, exp2, ceil, max, abs
from algorithm import vectorize

from .node import Node
from .utils import Vector, get_broadcasted_shape_for_ew_op
from .utils.shape import shape
from .kernels import *

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


alias cos_code = 0
alias bwcos_code = 1
alias sin_code = 2
alias bwsin_code = 3
alias tan_code = 4
alias bwtan_code = 5
alias acos_code = 6
alias bwacos_code = 7
alias asin_code = 8
alias bwasin_code = 9
alias atan_code = 10
alias bwatan_code = 11
alias cosh_code = 12
alias bwcosh_code = 13
alias sinh_code = 14
alias bwsinh_code = 15
alias log_code = 16
alias bwlog_code = 17
alias log2_code = 18
alias bwlog2_code = 19
alias exp2_code = 20
alias bwexp2_code = 21
alias sqrt_code = 22
alias bwsqrt_code = 23
alias abs_code = 24
alias bwabs_code = 25
alias copy_code = 26
alias bwcopy_code = 27
alias add_code = 28
alias bwadd_code = 29
alias sub_code = 30
alias bwsub_code = 31
alias mul_code = 32
alias bwmul_code = 33
alias div_code = 34
alias bwdiv_code = 35
alias pow_code = 36
alias bwpow_code = 37
alias mmul_code = 38
alias bwmmul_code = 39
alias reshape_code = 40
alias bwreshape_code = 41
alias transpose_code = 42
alias bwtranspose_code = 43
alias sum_code = 44
alias bwsum_code = 45
alias conv2d_code = 46
alias bwconv2d_code = 47
alias maxpool2dd_code = 48
alias bwmaxpool2d_code = 49

# Activation Functions
alias elu_code = 60
alias bwelu_code = 61
alias exp_code = 62
alias bwexp_code = 63
alias gelu_code = 64
alias bwgelu_code = 65
alias hard_sigmoid_code = 66
alias bwhard_sigmoid_code = 67
alias linear_code = 68
alias bwlinear_code = 69
alias mish_code = 70
alias bwmish_code = 71
alias relu_code = 72
alias bwrelu_code = 73
alias selu_code = 74
alias bwselu_code = 75
alias sigmoid_code = 76
alias bwsigmoid_code = 77
alias softmax_code = 78
alias bwsoftmax_code = 79
alias softplus_code = 80
alias bwsoftplus_code = 81
alias softsign_code = 82
alias bwsoftsign_code = 83
alias swish_code = 84
alias bwswish_code = 85
alias tanh_code = 86
alias bwtanh_code = 87

# Loss functions
alias kld_code = 90
alias bwkld_code = 91
alias mae_code = 92
alias bwmae_code = 93
alias mape_code = 94
alias bwmape_code = 95
alias mse_code = 96
alias bwmse_code = 97
alias msle_code = 98
alias bwmsle_code = 99
alias bce_code = 100
alias bwbce_code = 101
alias cce_code = 102
alias bwcce_code = 103
alias cfce_code = 104
alias bwcfce_code = 105
alias cs_code = 106
alias bwcs_code = 107
alias huber_code = 108
alias bwhuber_code = 109
alias logcosh_code = 110
alias bwlogcosh_code = 111
alias poisson_code = 112
alias bwpoisson_code = 113
alias scce_code = 114
alias bwscce_code = 115


fn unary(b: Node, a: Node):
    ...


fn binary(c: Node, a: Node, b: Node):
    ...


fn view(b: Node, a: Node):
    ...


fn reduce(c: Node, a: Node, b: Node):
    ...


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

        let memory_pool_manager = Pointer[VectorInt].alloc(30)
        for i in range(30):
            memory_pool_manager.store(i, VectorInt())

        let free_node_ids = Pointer[VectorInt].alloc(1)
        free_node_ids.store(VectorInt())

        let free_data_ids = Pointer[VectorInt].alloc(1)
        free_data_ids.store(VectorInt())

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        let kernels = Pointer[op_tuple].alloc(120)
        kernels.store(cos_code, op_tuple("cos", fw_cos, binary, view, reduce))
        kernels.store(bwcos_code, op_tuple("bwcos", bw_cos, binary, view, reduce))
        kernels.store(sin_code, op_tuple("sin", fw_sin, binary, view, reduce))
        kernels.store(bwsin_code, op_tuple("bwsin", bw_sin, binary, view, reduce))
        kernels.store(tan_code, op_tuple("tan", fw_tan, binary, view, reduce))
        kernels.store(bwtan_code, op_tuple("bwtan", bw_tan, binary, view, reduce))
        kernels.store(acos_code, op_tuple("acos", fw_acos, binary, view, reduce))
        kernels.store(bwacos_code, op_tuple("bwacos", bw_acos, binary, view, reduce))
        kernels.store(asin_code, op_tuple("asin", fw_asin, binary, view, reduce))
        kernels.store(bwasin_code, op_tuple("bwasin", bw_asin, binary, view, reduce))
        kernels.store(atan_code, op_tuple("atan", fw_atan, binary, view, reduce))
        kernels.store(bwatan_code, op_tuple("bwatan", bw_atan, binary, view, reduce))
        kernels.store(cosh_code, op_tuple("cosh", fw_cosh, binary, view, reduce))
        kernels.store(bwcos_code, op_tuple("bwcosh", bw_cosh, binary, view, reduce))
        kernels.store(sinh_code, op_tuple("sinh", fw_sinh, binary, view, reduce))
        kernels.store(bwsin_code, op_tuple("bwsinh", bw_sinh, binary, view, reduce))
        kernels.store(log_code, op_tuple("log", fw_log, binary, view, reduce))
        kernels.store(bwlog_code, op_tuple("bwlog", bw_log, binary, view, reduce))
        kernels.store(log2_code, op_tuple("log2", fw_log2, binary, view, reduce))
        kernels.store(bwlog2_code, op_tuple("bwlog", bw_log2, binary, view, reduce))
        kernels.store(exp2_code, op_tuple("exp2", fw_exp2, binary, view, reduce))
        kernels.store(bwexp2_code, op_tuple("bwexp2", bw_exp2, binary, view, reduce))
        kernels.store(sqrt_code, op_tuple("sqrt", fw_sqrt, binary, view, reduce))
        kernels.store(bwsqrt_code, op_tuple("bwsqrt", bw_sqrt, binary, view, reduce))
        kernels.store(abs_code, op_tuple("abs", fw_abs, binary, view, reduce))
        kernels.store(bwabs_code, op_tuple("bwabs", bw_abs, binary, view, reduce))
        kernels.store(copy_code, op_tuple("copy", fw_copy, binary, view, reduce))
        kernels.store(bwcopy_code, op_tuple("bwcopy", bw_copy, binary, view, reduce))
        kernels.store(add_code, op_tuple("add", unary, fw_add, view, reduce))
        kernels.store(bwadd_code, op_tuple("bwadd", unary, bw_add, view, reduce))
        kernels.store(sub_code, op_tuple("sub", unary, fw_sub, view, reduce))
        kernels.store(bwsub_code, op_tuple("bwsub", unary, bw_sub, view, reduce))
        kernels.store(mul_code, op_tuple("mul", unary, fw_mul, view, reduce))
        kernels.store(bwmul_code, op_tuple("bwmul", unary, bw_mul, view, reduce))
        kernels.store(div_code, op_tuple("div", unary, fw_div, view, reduce))
        kernels.store(bwdiv_code, op_tuple("bwdiv", unary, bw_div, view, reduce))
        kernels.store(pow_code, op_tuple("pow", unary, fw_pow, view, reduce))
        kernels.store(bwpow_code, op_tuple("bwpow", unary, bw_pow, view, reduce))
        kernels.store(mmul_code, op_tuple("mmul", unary, fw_mmul, view, reduce))
        kernels.store(bwmmul_code, op_tuple("bwmmul", unary, bw_mmul, view, reduce))
        kernels.store(
            reshape_code, op_tuple("reshape", fw_reshape, binary, view, reduce)
        )
        kernels.store(
            bwreshape_code, op_tuple("bwreshape", bw_reshape, binary, view, reduce)
        )
        kernels.store(
            transpose_code, op_tuple("transpose", fw_transpose, binary, view, reduce)
        )
        kernels.store(
            bwtranspose_code,
            op_tuple("bwtranspose", bw_transpose, binary, view, reduce),
        )
        kernels.store(sum_code, op_tuple("sum", fw_sum, binary, view, reduce))
        kernels.store(bwsum_code, op_tuple("bwsum", bw_sum, binary, view, reduce))
        kernels.store(conv2d_code, op_tuple("conv2d", unary, conv_2d, view, reduce))
        kernels.store(
            bwconv2d_code, op_tuple("bwconv2d", unary, bw_conv_2d, view, reduce)
        )
        kernels.store(
            maxpool2dd_code, op_tuple("maxpool2dd", max_pool_2d, binary, view, reduce)
        )
        kernels.store(
            bwmaxpool2d_code,
            op_tuple("bwmaxpool2d", bw_max_pool_2d, binary, view, reduce),
        )
        kernels.store(elu_code, op_tuple("elu", fw_elu, binary, view, reduce))
        kernels.store(bwelu_code, op_tuple("bwelu", bw_elu, binary, view, reduce))
        kernels.store(exp_code, op_tuple("exp", fw_exp, binary, view, reduce))
        kernels.store(bwexp_code, op_tuple("bwexp", bw_exp, binary, view, reduce))
        kernels.store(gelu_code, op_tuple("gelu", fw_gelu, binary, view, reduce))
        kernels.store(bwgelu_code, op_tuple("bwgelu", bw_gelu, binary, view, reduce))
        kernels.store(
            hard_sigmoid_code,
            op_tuple("hard_sigmoid", fw_hard_sigmoid, binary, view, reduce),
        )
        kernels.store(
            bwhard_sigmoid_code,
            op_tuple("bwhard_sigmoid", bw_hard_sigmoid, binary, view, reduce),
        )
        kernels.store(linear_code, op_tuple("linear", fw_linear, binary, view, reduce))
        kernels.store(
            bwlinear_code, op_tuple("bwlinear", bw_linear, binary, view, reduce)
        )
        kernels.store(mish_code, op_tuple("mish", fw_mish, binary, view, reduce))
        kernels.store(bwmish_code, op_tuple("bwmish", bw_mish, binary, view, reduce))
        kernels.store(relu_code, op_tuple("relu", fw_relu, binary, view, reduce))
        kernels.store(bwrelu_code, op_tuple("bwrelu", bw_relu, binary, view, reduce))
        kernels.store(selu_code, op_tuple("selu", fw_selu, binary, view, reduce))
        kernels.store(bwselu_code, op_tuple("bwselu", bw_selu, binary, view, reduce))
        kernels.store(
            sigmoid_code, op_tuple("sigmoid", fw_sigmoid, binary, view, reduce)
        )
        kernels.store(
            bwsigmoid_code, op_tuple("bwsigmoid", bw_sigmoid, binary, view, reduce)
        )
        kernels.store(
            softmax_code, op_tuple("softmax", fw_softmax, binary, view, reduce)
        )
        kernels.store(
            bwsoftmax_code, op_tuple("bwsoftmax", bw_softmax, binary, view, reduce)
        )
        kernels.store(
            softplus_code, op_tuple("softplus", fw_softplus, binary, view, reduce)
        )
        kernels.store(
            bwsoftplus_code, op_tuple("bwsoftplus", bw_softplus, binary, view, reduce)
        )
        kernels.store(
            softsign_code, op_tuple("softsign", fw_softsign, binary, view, reduce)
        )
        kernels.store(
            bwsoftsign_code, op_tuple("bwsoftsign", bw_softsign, binary, view, reduce)
        )
        kernels.store(swish_code, op_tuple("swish", fw_swish, binary, view, reduce))
        kernels.store(bwswish_code, op_tuple("bwswish", bw_swish, binary, view, reduce))
        kernels.store(tanh_code, op_tuple("tanh", fw_tanh, binary, view, reduce))
        kernels.store(bwtanh_code, op_tuple("bwtanh", bw_tanh, binary, view, reduce))

        kernels.store(kld_code, op_tuple("kld", unary, fw_kld, view, reduce))
        kernels.store(bwkld_code, op_tuple("bwkld", unary, bw_kld, view, reduce))
        kernels.store(mae_code, op_tuple("mae", unary, fw_mae, view, reduce))
        kernels.store(bwmae_code, op_tuple("bwmae", unary, bw_mae, view, reduce))
        kernels.store(mape_code, op_tuple("mape", unary, fw_mape, view, reduce))
        kernels.store(bwmape_code, op_tuple("bwmape", unary, bw_mape, view, reduce))
        kernels.store(mse_code, op_tuple("mse", unary, fw_mse, view, reduce))
        kernels.store(bwmse_code, op_tuple("bwmse", unary, bw_mse, view, reduce))
        kernels.store(msle_code, op_tuple("msle", unary, fw_msle, view, reduce))
        kernels.store(bwmsle_code, op_tuple("bwmsle", unary, bw_msle, view, reduce))
        kernels.store(bce_code, op_tuple("bce", unary, fw_bce, view, reduce))
        kernels.store(bwbce_code, op_tuple("bwbce", unary, bw_bce, view, reduce))
        kernels.store(cce_code, op_tuple("cce", unary, fw_cce, view, reduce))
        kernels.store(bwcce_code, op_tuple("bwcce", unary, bw_cce, view, reduce))
        kernels.store(cfce_code, op_tuple("cfce", unary, fw_cfce, view, reduce))
        kernels.store(bwcfce_code, op_tuple("bwcfce", unary, bw_cfce, view, reduce))
        kernels.store(cs_code, op_tuple("cs", unary, fw_cs, view, reduce))
        kernels.store(bwcs_code, op_tuple("bwcs", unary, bw_cs, view, reduce))
        kernels.store(huber_code, op_tuple("huber", unary, fw_huber, view, reduce))
        kernels.store(bwhuber_code, op_tuple("bwhuber", unary, bw_huber, view, reduce))
        kernels.store(
            logcosh_code, op_tuple("logcosh", unary, fw_logcosh, view, reduce)
        )
        kernels.store(
            bwlogcosh_code, op_tuple("bwlogcosh", unary, bw_logcosh, view, reduce)
        )
        kernels.store(
            poisson_code, op_tuple("poisson", unary, fw_poisson, view, reduce)
        )
        kernels.store(
            bwpoisson_code, op_tuple("bwpoisson", unary, bw_poisson, view, reduce)
        )
        kernels.store(scce_code, op_tuple("scce", unary, fw_scce, view, reduce))
        kernels.store(bwscce_code, op_tuple("bwscce", unary, bw_scce, view, reduce))

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
        for i in range(30):
            let ceiled_cap = exp2(Float32(i)).to_int()
            print_no_newline("    cap:", ceiled_cap)
            print_no_newline(" - data_ids: [")
            for j in range(self.memory_pool_manager.load(i).len.load()):
                print_no_newline(self.memory_pool_manager.load(i).load(j))
                if j < self.memory_pool_manager.load(i).len.load() - 1:
                    print_no_newline(", ")
            print("]")

    fn print(self) raises:
        print("\nGraph (Nodes):")
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node == Pointer[Node].get_null():
                continue
            node.load().print()

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
        for i in range(30):
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

    fn optimizer_step(self, learning_rate: Float32, type: String) raises:
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i).load()
            if (
                type == "sgd"
                and node.requires_grad_ptr.load()
                and node.grad_computed_ptr.load()
            ):

                @parameter
                fn v_sgd_update[_nelts: Int](i: Int):
                    node.store_data[_nelts](
                        i,
                        node.load_data[_nelts](i)
                        - node.load_grad[_nelts](i) * learning_rate,
                    )

                vectorize[nelts, v_sgd_update](node.load_cap())

    fn cos(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = cos_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sin(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = sin_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn tan(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = tan_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn acos(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = acos_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn asin(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = asin_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn atan(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = atan_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn cosh(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = cosh_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sinh(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = sinh_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn log(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = log_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn log2(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = log2_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn exp2(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = exp2_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sqrt(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = sqrt_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn abs(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = abs_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn copy(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = copy_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, True, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn add(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = add_code
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn sub(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = sub_code
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn mul(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = mul_code
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn div(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = div_code
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn pow(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = pow_code
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
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
            raise "Shapes don't fit for matrix multiplication"
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
        if in_channels != b.load().shape_ptr.load().load(1):
            raise "Error (at conv_2d): number of channels must be equal in the input and the kernels"
        let kernel_width = b.load().shape_ptr.load().load(2)
        let kernel_height = b.load().shape_ptr.load().load(3)

        let shape = shape(
            batch_size,
            out_channels,
            (width - kernel_width + 2 * padding) // stride + 1,
            (height - kernel_height + 2 * padding) // stride + 1,
        )
        let operator_id = conv2d_code
        let checkpoint = True
        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)
        let c = self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

        return c

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
        let operator_id = maxpool2dd_code
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

    fn reshape(
        self, parent1_ptr: Pointer[Node], shape: Vector[Int]
    ) raises -> Pointer[Node]:
        let operator_id = reshape_code
        let checkpoint = False
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn transpose(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = transpose_code
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

    fn sum(self, parent1_ptr: Pointer[Node], axis: Int) raises -> Pointer[Node]:
        let operator_id = sum_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    # Activation functions
    fn elu(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = elu_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn exp(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = exp_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn gelu(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = gelu_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn hard_sigmoid(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = hard_sigmoid_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn linear(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = linear_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn mish(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = mish_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn relu(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = relu_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn selu(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = selu_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sigmoid(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = sigmoid_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn softmax(self, parent1_ptr: Pointer[Node], axis: Int) raises -> Pointer[Node]:
        let operator_id = softmax_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn softplus(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = softplus_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn softsign(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = softsign_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn swish(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = swish_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn tanh(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = tanh_code
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    # Loss functions
    fn kld(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = kld_code
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

    fn mae(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = mae_code
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

    fn mape(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = mape_code
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

    fn mse(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = mse_code
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

    fn msle(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = msle_code
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

    fn bce(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = bce_code
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

    fn cce(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = cce_code
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

    fn cfce(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = cfce_code
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

    fn cs(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = cs_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape,
            True,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )

    fn huber(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = huber_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape,
            True,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )

    fn logcosh(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = logcosh_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape,
            True,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )

    fn poisson(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = poisson_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape,
            True,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )

    fn scce(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = scce_code
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        other_params.push_back(1)
        return self.node(
            shape,
            True,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )
