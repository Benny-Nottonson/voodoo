from math import abs, exp, log, tanh, max, erf, cosh
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
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_fw[nelts: Int](i: Int):
            node.store_data[nelts](
                i,
                fw_vec[nelts, arg1, arg2, arg3](parent1.load_data[nelts](i)),
            )

        vectorize[nelts, vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
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
    alias fw = Generic[softmax_fw_vec, softmax_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[softmax_fw_vec, softmax_bw_vec, 0.0, 0.0, 0.0].bw


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
    alias fw = Generic[exp_vec, exp_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[exp_vec, exp_vec, 0.0, 0.0, 0.0].bw


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
    alias fw = Generic[log_softmax_fw_vec, log_softmax_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = Generic[log_softmax_fw_vec, log_softmax_bw_vec, 0.0, 0.0, 0.0].bw


@parameter
@always_inline
fn relu_fw_vec[
    nelts: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = f32_max,
    threshold: Float32 = 0.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > threshold ? (x > max_value ? max_value : x) : negative_slope * x
    # Best is 4 instructions (compare, select, mul, min), 2 if max == f32_max and slope == 0
    @parameter
    if negative_slope == 0.0 and max_value == f32_max:
        return (x > threshold).select(x, 0.0)
    return (x > threshold).select(x, negative_slope * x).min(max_value)


@parameter
@always_inline
fn relu_bw_vec[
    nelts: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = f32_max,
    threshold: Float32 = 0.0,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > threshold ? (x > max_value ? 0 : 1) : negative_slope
    # Best is 4 instructions (compare, select, compare, select), 2 max == f32_max and slope == 0
    @parameter
    if negative_slope == 0.0 and max_value == f32_max:
        return (x > threshold).cast[DType_F32]()
    return (x < max_value).select((x > threshold).select(1.0, negative_slope), 0.0)


@parameter
@always_inline
fn sigmoid_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = 1 / (1 + e^-x)
    # Best is 3 instructions (exp, add, div)
    return 1.0 / (1.0 + exp(-x))


@parameter
@always_inline
fn sigmoid_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = e^-x / (1 + e^-x)^2
    # Best is 6 instructions (exp, div, fma, exp, mul, add)
    let e_nx = (exp(-x))
    return e_nx / (1.0 + e_nx) ** 2


@parameter
@always_inline
fn softplus_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = log(1 + e^x)
    # Best is 3 instructions (exp, add, log)
    return log(1.0 + exp(x))


@parameter
@always_inline
fn softplus_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = e^x / (1 + e^x)
    # Best is 3 instructions (exp, add, div)
    let e_x = (exp(x))
    return e_x / (1.0 + e_x)


@parameter
@always_inline
fn softsign_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x / (1 + |x|)
    # Best is 3 instructions (abs, add, div)
    return x / (1.0 + abs(x))


@parameter
@always_inline
fn softsign_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = 1 / (1 + |x|)^2
    # Simplifies to 1 / (1 + x^2 + 2|x|)
    # Best is 4 instructions (div, abs, fma, fma)
    return 1.0 / abs(x).fma(2.0, x.fma(x, 1.0))


@parameter
@always_inline
fn tanh_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = tanh(x)
    # Best is 1 instruction (tanh)
    return tanh(x)


@parameter
@always_inline
fn tanh_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = 1 / cosh(x)^2
    # Best is 3 instructions (cosh, pow, div)
    return 1.0 / cosh(x) ** 2


@parameter
@always_inline
fn selu_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > 0 ? 1.05070098 * x : 1.05070098 * 1.67326324 * (e^x - 1)
    # Best is 5 instructions (compare, select, mul, exp, fma)
    return (x > 0.0).select(1.05070098 * x, exp(x).fma(1.75809932607, -1.75809932607))


@parameter
@always_inline
fn selu_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > 0 ? 1.05070098 : 1.05070098 * 1.67326324 * e^x
    # Best is 4 instructions (compare, select, mul, exp)
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
    # Best is 5 instructions (compare, select, mul, exp, sub), 4 if alpha == 1
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
    # Best is 4 instructions (compare, select, mul, exp), 3 if alpha == 1
    @parameter
    if alpha == 1.0:
        return (x > 0.0).select(1.0, exp(x))
    return (x > 0.0).select(1.0, alpha * exp(x))


@parameter
@always_inline
fn exp_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = e^x
    # Best is 1 instruction (exp)
    return exp(x)


@parameter
@always_inline
fn silu_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x / (1 + e^-x)
    # Best is 4 instructions (div, add, exp, inverse)
    return x / (1.0 + exp(-x))


@parameter
@always_inline
fn silu_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = (e^x * x + e^x + e^2x) / (e^x + 1)^2
    # Best is 8 instructions (exp, fma, add, exp, mul, div, add, pow)
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
    # Best is 6 instructions (mul, tanh, fma, mul, pow, fma), 4 if approximate == 0
    let x_05 = x * 0.5

    @parameter
    if approximate == 0.0:
        return erf(x / 1.4142135623730951).fma(x_05, x_05)
    return tanh(x.fma(0.7978845608028654, 0.03567740813 * x**3)).fma(x_05, x_05)


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
    # Best is 7 instructions (tanh, fma, fma, mul, mul, sub, pow), 7 if approximate == 0
    @parameter
    if approximate == 0.0:
        return x.fma(
            0.3989422804014327 * exp(-0.5 * x**2),
            erf(0.7071067811865475 * x).fma(0.5, 0.5),
        )
    let tanh_x = tanh(x.fma(0.7978845608028654, 0.03567740813 * x**3))
    return tanh_x.fma(tanh_x, 1.0).fma(
        0.5, 0.7978845608028654 * x * (1.0 - tanh_x**2)
    )


@parameter
@always_inline
fn hard_sigmoid_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x > 2.5 ? 1 : x < -2.5 ? 0 : 0.2 * x + 0.5
    # Best is 5 instructions (compare, select, compare, select, fma)
    return (x > 2.5).select(1.0, (x > -2.5).select(x.fma(0.2, 0.5), 0.0))


@parameter
@always_inline
fn hard_sigmoid_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = x > -2.5 ? x < 2.5 ? 0.2 : 0 : 0
    # Best is 5 instructions (compare, and, compare, cast, mul)
    return ((x > -2.5) & (x < 2.5)).cast[DType_F32]() * 0.2


@parameter
@always_inline
fn linear_fw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x
    # Best is 1 instruction (mov)
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
    # Best is 1 instruction (mov)
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
    # Best is 5 instructions (mul, tanh, log, add, exp)
    return x * tanh(log(1.0 + exp(x)))


@parameter
@always_inline
fn mish_bw_vec[
    nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = tanh(log(exp(x) + 1)) + (x * exp(x) * (1 / cosh(ln(exp(x) + 1)) ^ 2)) / (exp(x) + 1)
    # Best is 14 instructions (exp, tanh, log, add, add, mul, mul, div, cosh, log, add, pow, div, add)
    let e_x = exp(x)
    return tanh(log(e_x + 1)) + (x * e_x * (1 / cosh(log(e_x + 1)) ** 2)) / (e_x + 1)


@parameter
@always_inline
fn softmax_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = e^x / sum(e^x)
    # Best is 4 instructions (exp, add, div, reduce)
    return exp(x) / exp(x).reduce_add()


@parameter
@always_inline
fn softmax_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = e^x * (sum(e^x) - e^x) / sum(e^x)^2
    # Best is 6 instructions (exp, reduce, fma, exp, mul, div)
    let e_x = exp(x)
    return e_x * (e_x.reduce_add() - e_x) / e_x.reduce_add() ** 2


@parameter
@always_inline
fn log_softmax_fw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x) = x - log(sum(e^x))
    # Best is 4 instructions (exp, add, log, reduce)
    return x - log(exp(x).reduce_add())


@parameter
@always_inline
fn log_softmax_bw_vec[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f'(x) = 1 - e^x / sum(e^x)
    # Best is 4 instructions (exp, reduce, div, sub)
    return 1.0 - exp(x) / exp(x).reduce_add()
