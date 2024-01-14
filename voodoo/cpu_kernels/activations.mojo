from math import abs, exp, log, tanh, max, erf
from algorithm import vectorize
from voodoo import Node
from ..constants import DType_F32, nelts, f32_max

# TODO: Rewrite when lambda functions are supported


alias generic_vectorized = fn[nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32] (
    SIMD[DType_F32, nelts]
) -> SIMD[DType_F32, nelts]


struct Generic[
    fw_vec: generic_vectorized,
    bw_vec: generic_vectorized,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
]:
    @staticmethod
    @always_inline
    fn fw(node: Node, parent1: Node):
        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            node.store_data[nelts](
                i,
                fw_vec[nelts, arg1, arg2, arg3](parent1.load_data[nelts](i)),
            )

        vectorize[nelts, vectorized_fw](node.load_cap())

    @staticmethod
    @always_inline
    fn bw(node: Node, parent1: Node):
        @parameter
        @always_inline
        fn vectorized_bw[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i,
                parent1.load_grad[nelts](i)
                + node.load_grad[nelts](i)
                * bw_vec[nelts, arg1, arg2, arg3](parent1.load_data[nelts](i)),
            )

        vectorize[nelts, vectorized_bw](node.load_cap())


struct Relu[arg1: Float32 = 0.0, arg2: Float32 = f32_max, arg3: Float32 = 0.0]:
    alias fw = Generic[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].fw
    alias bw = Generic[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].bw


struct Sigmoid[]:
    alias fw = Generic[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].bw


struct Softmax[]:
    alias fw = _Softmax.fw
    alias bw = _Softmax.bw


struct Softplus[]:
    alias fw = Generic[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].bw


struct Softsign[]:
    alias fw = Generic[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].bw


struct Tanh[]:
    alias fw = Generic[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].bw


struct Selu[]:
    alias fw = Generic[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].bw


struct Elu[alpha: Float32 = 1.0]:
    alias fw = Generic[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].fw
    alias bw = Generic[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].bw


struct Exp[]:
    alias fw = Generic[exp_fw_vec, exp_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[exp_fw_vec, exp_bw_vec, 0.0, 0.0, 0.0].bw


struct LeakyRelu[alpha: Float32 = 0.0]:
    alias fw = Generic[relu_fw_vec, relu_bw_vec, alpha, f32_max, 0.0].fw
    alias bw = Generic[relu_fw_vec, relu_bw_vec, alpha, f32_max, 0.0].bw


struct Relu6[]:
    alias fw = Generic[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].fw
    alias bw = Generic[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].bw


struct Silu[]:
    alias fw = Generic[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].bw


struct Gelu[approximate: Float32 = 0.0]:
    alias fw = Generic[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].fw
    alias bw = Generic[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].bw


struct HardSigmoid[]:
    alias fw = Generic[hard_sigmoid_fw_vec, hard_sigmoid_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[hard_sigmoid_fw_vec, hard_sigmoid_bw_vec, 0.0, 0.0, 0.0].bw


struct Linear[]:
    alias fw = Generic[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].bw


struct Mish[]:
    alias fw = Generic[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].bw


struct LogSoftmax[]:
    alias fw = _LogSoftmax.fw
    alias bw = _LogSoftmax.bw

# TODO!IMPORATNT: Improve using SIMD.fma, select, etc.

@parameter
@always_inline
fn relu_fw_vec[
    nelts: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = f32_max,
    threshold: Float32 = 0.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > threshold ? (x > max_value ? max_value : x) : negative_slope * x
    @parameter
    if negative_slope == 0.0 and max_value == f32_max:
        return (x > threshold).select(x, 0.0)

    return (x > max_value).select(
        max_value, (x > threshold).select(x, negative_slope * x)
    )


@parameter
@always_inline
fn relu_bw_vec[
    nelts: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = f32_max,
    threshold: Float32 = 0.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > threshold ? (x > max_value ? 0 : 1) : negative_slope
    let threshold_mask = (x > threshold).cast[DType_F32]()

    @parameter
    if negative_slope == 0.0 and max_value == f32_max:
        return (x > threshold).cast[DType_F32]()
    return (
        ((x > threshold) & (x <= max_value)).cast[DType_F32]()
    )


@parameter
@always_inline
fn sigmoid_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = 1 / (1 + e^-x)
    return 1.0 / (1.0 + exp(-x))


@parameter
@always_inline
fn sigmoid_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = f(x)(1-f(x))
    # simplifies to e^x / (e^x + 1)^2
    let e_x = (exp(x))
    return e_x / (e_x + 1.0) ** 2


@parameter
@always_inline
fn softplus_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = log(1 + e^x)
    return log(1.0 + exp(x))


@parameter
@always_inline
fn softplus_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = e^x / (1 + e^x)
    let e_x = (exp(x))
    return e_x / (1.0 + e_x)


@parameter
@always_inline
fn softsign_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x / (1 + |x|)
    return x / (1.0 + abs(x))


@parameter
@always_inline
fn softsign_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = 1 / (1 + |x|)^2
    return 1.0 / (1.0 + abs(x)) ** 2


@parameter
@always_inline
fn tanh_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = tanh(x)
    return tanh(x)


@parameter
@always_inline
fn tanh_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = 1 - tanh(x)^2
    return 1.0 - tanh(x) ** 2


@parameter
@always_inline
fn selu_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > 0 ? 1.05070098 * x : 1.05070098 * 1.67326324 * (e^x - 1)
    return (x > 0.0).select(1.05070098 * x, 1.75809932607 * (exp(x) - 1.0))


@parameter
@always_inline
fn selu_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > 0 ? 1.05070098 : 1.05070098 * 1.67326324 * e^x
    return (x > 0.0).select(1.05070098, 1.75809932607 * exp(x))


@parameter
@always_inline
fn elu_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    alpha: Float32 = 1.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > 0 ? x : alpha * (e^x - 1)
    @parameter
    if alpha == 1.0:
        return (x > 0.0).select(x, exp(x) - 1.0)
    return (x > 0.0).select(x, alpha * (exp(x) - 1.0))


@parameter
@always_inline
fn elu_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    alpha: Float32 = 1.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > 0 ? 1 : alpha * e^x
    @parameter
    if alpha == 1.0:
        return (x > 0.0).select(1.0, exp(x))
    return (x > 0.0).select(1.0, alpha * exp(x))


@parameter
@always_inline
fn exp_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = e^x
    return exp(x)


@parameter
@always_inline
fn exp_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = e^x
    return exp(x)


@parameter
@always_inline
fn silu_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x / (1 + e^-x)
    return x / (1.0 + exp(-x))


@parameter
@always_inline
fn silu_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = (e^x * x + e^x + e^2x) / (e^x + 1)^2
    let e_x = exp(x)
    return (e_x.fma(x, e_x) + exp(2.0 * x)) / (e_x + 1.0) ** 2


@parameter
@always_inline
fn gelu_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    approximate: Float32 = 0.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) when approximate == 0.0 = 0.5 * x * (1 + erf(x / sqrt(2)))
    # f(x) when approximate != 0.0 = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    @parameter
    if approximate == 0.0:
        return x * erf(x / 1.4142135623730951).fma(0.5, 0.5)
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


@parameter
@always_inline
fn gelu_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    approximate: Float32 = 0.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) when approximate == 0.0 = 0.5 * (erf(0.7071067811865475 * x) + 1) + 0.3989422804014327 * x * exp(-0.5 * x^2)
    # f'(x) when approximate != 0.0 = 0.5 * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3))^2) + 0.7978845608028654 * x * (1 - tanh(0.7978845608028654 * (x + 0.044715 * x^3))^2)
    @parameter
    if approximate == 0.0:
        return 0.5 * (erf(0.7071067811865475 * x) + 1.0) + 0.3989422804014327 * x * exp(
            -0.5 * x**2
        )
    let tanh_x_2 = tanh(0.7978845608028654 * (x + 0.044715 * x**3)) ** 2
    return 0.5 * (1.0 + tanh_x_2) + 0.7978845608028654 * x * (1.0 - tanh_x_2)


@parameter
@always_inline
fn hard_sigmoid_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > 2.5 ? 1 : x < -2.5 ? 0 : 0.2 * x + 0.5
    return (x > 2.5).cast[DType_F32]() + (x > -2.5).cast[DType_F32]() * (x < 2.5).cast[
        DType_F32
    ]() * (0.2 * x + 0.5)


@parameter
@always_inline
fn hard_sigmoid_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > -2.5 ? x < 2.5 ? 0.2 : 0 : 0
    return (x > -2.5).cast[DType_F32]() * (x < 2.5).cast[DType_F32]() * 0.2


@parameter
@always_inline
fn linear_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x
    return x


@parameter
@always_inline
fn linear_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = 1
    return 1.0


@parameter
@always_inline
fn mish_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x * tanh(log(1 + e^x))
    return x * tanh(log(1.0 + exp(x)))


@parameter
@always_inline
fn mish_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = (e^x (4 e^x x + 4 x + 6 e^x + 4 e^(2 x) + e^(3 x) + 4))/(2 e^x + e^(2 x) + 2)^2
    let e_x = exp(x)
    let e_2x = exp(2.0 * x)
    return (
        e_x
        * (4.0 * e_x * x + 4.0 * x + 6.0 * e_x + 4.0 * e_2x + exp(3.0 * x) + 4.0)
        / (2.0 * e_x + e_2x + 2.0) ** 2
    )


struct _Softmax:
    # f(x) = e^wx_i / sum(e^wx_i)
    # f'x(x) = f(x) * (1 - f(x))
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N
            var max_el: Float32 = 0.0

            @parameter
            fn vectorized_max[nelts: Int](i: Int):
                max_el = max(max_el, parent1.load_data[nelts](offset + i).reduce_max())

            vectorize[nelts, vectorized_max](N)
            var sum: Float32 = 0.0

            @parameter
            fn vectorized_exp[nelts: Int](i: Int):
                let temp = exp(parent1.load_data[nelts](offset + i) - max_el)
                node.store_data[nelts](offset + i, temp)
                sum += temp.reduce_add()

            vectorize[nelts, vectorized_exp](N)

            @parameter
            fn vectorized_div[nelts: Int](i: Int):
                node.store_data[nelts](
                    offset + i, node.load_data[nelts](offset + i) / sum
                )

            vectorize[nelts, vectorized_div](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N

            @parameter
            fn vectorized_softmax_bw_outer[nelts: Int](j: Int):
                var grad: Float32 = 0.0

                @parameter
                fn vectorized_softmax_bw[nelts: Int](i: Int):
                    if i == j:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * node.load_data[nelts](offset + i)
                            * (1.0 - node.load_data[nelts](offset + i))
                        ).reduce_add()
                    else:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * node.load_data[nelts](offset + i)
                            * node.load_data[nelts](offset + j)
                            * -1
                        ).reduce_add()

                vectorize[nelts, vectorized_softmax_bw](N)
                parent1.store_grad[nelts](
                    offset + j, parent1.load_grad[nelts](offset + j) + grad
                )

            vectorize[nelts, vectorized_softmax_bw_outer](N)


struct _LogSoftmax:
    # f(x) = log(e^wx_i / sum(e^wx_i))
    # f'x(x) = f(x) * (1 - f(x))
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N
            var max_el: Float32 = 0.0

            @parameter
            fn vectorized_max[nelts: Int](i: Int):
                max_el = max(max_el, parent1.load_data[nelts](offset + i).reduce_max())

            vectorize[nelts, vectorized_max](N)
            var sum: Float32 = 0.0

            @parameter
            fn vectorized_exp[nelts: Int](i: Int):
                let temp = exp(parent1.load_data[nelts](offset + i) - max_el)
                node.store_data[nelts](offset + i, temp)
                sum += temp.reduce_add()

            vectorize[nelts, vectorized_exp](N)

            @parameter
            fn vectorized_div[nelts: Int](i: Int):
                node.store_data[nelts](
                    offset + i, node.load_data[nelts](offset + i) / sum
                )

            vectorize[nelts, vectorized_div](N)

            @parameter
            fn vectorized_log[nelts: Int](i: Int):
                node.store_data[nelts](
                    offset + i, log(node.load_data[nelts](offset + i))
                )

            vectorize[nelts, vectorized_log](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N

            @parameter
            fn vectorized_log_softmax_bw_outer[nelts: Int](j: Int):
                var grad: Float32 = 0.0

                @parameter
                fn vectorized_log_softmax_bw[nelts: Int](i: Int):
                    if i == j:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * (1.0 - node.load_data[nelts](offset + i))
                        ).reduce_add()
                    else:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * node.load_data[nelts](offset + j)
                            * -1
                        ).reduce_add()

                vectorize[nelts, vectorized_log_softmax_bw](N)
                parent1.store_grad[nelts](
                    offset + j, parent1.load_grad[nelts](offset + j) + grad
                )

            vectorize[nelts, vectorized_log_softmax_bw_outer](N)
