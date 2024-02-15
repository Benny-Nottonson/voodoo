from math import exp, log, abs, tanh, cosh, erf

from voodoo.constants import F32_MAX, NELTS
from voodoo.autograd.kernels.generics import GenericActivation


trait Activation:
    ...


@register_passable("trivial")
struct Relu[arg1: Float32, arg2: Float32, arg3: Float32](Activation):
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].bw


@register_passable("trivial")
struct Sigmoid[](Activation):
    alias fw = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Softplus[](Activation):
    alias fw = GenericActivation[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Softsign[](Activation):
    alias fw = GenericActivation[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Tanh[](Activation):
    alias fw = GenericActivation[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Selu[](Activation):
    alias fw = GenericActivation[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Elu[alpha: Float32](Activation):
    alias fw = GenericActivation[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].fw
    alias bw = GenericActivation[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].bw


@register_passable("trivial")
struct Exp[](Activation):
    alias fw = GenericActivation[exp_vec, exp_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[exp_vec, exp_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct LeakyRelu[alpha: Float32](Activation):
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, alpha, F32_MAX, 0.0].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, alpha, F32_MAX, 0.0].bw


@register_passable("trivial")
struct Relu6[](Activation):
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].bw


@register_passable("trivial")
struct Silu[](Activation):
    alias fw = GenericActivation[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Gelu[approximate: Float32](Activation):
    alias fw = GenericActivation[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].fw
    alias bw = GenericActivation[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].bw


@register_passable("trivial")
struct HardSigmoid[](Activation):
    alias fw = GenericActivation[hsig_fw_vec, hsig_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[hsig_fw_vec, hsig_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Linear[](Activation):
    alias fw = GenericActivation[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].bw


@register_passable("trivial")
struct Mish[](Activation):
    alias fw = GenericActivation[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].bw


fn relu_fw_vec[
    NELTS: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = F32_MAX,
    threshold: Float32 = 0.0,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x > threshold ? (x > max_value ? max_value : x) : negative_slope * x
    # Best is 4 instructions (compare, select, mul, min), 2 if max == F32_MAX and slope == 0
    @parameter
    if negative_slope == 0.0 and max_value == F32_MAX:
        return (x > threshold).select(x, 0.0)
    return (x > threshold).select(x, negative_slope * x).min(max_value)


fn relu_bw_vec[
    NELTS: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = F32_MAX,
    threshold: Float32 = 0.0,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = x > threshold ? (x > max_value ? 0 : 1) : negative_slope
    # Best is 4 instructions (compare, select, compare, select), 2 max == F32_MAX and slope == 0
    @parameter
    if negative_slope == 0.0 and max_value == F32_MAX:
        return (x > threshold).select[DType.float32](1.0, 0.0)
    return (x < max_value).select((x > threshold).select(1.0, negative_slope), 0.0)


fn sigmoid_fw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = 1 / (1 + e^-x)
    # Best is 3 instructions (exp, add, div)
    return 1.0 / (1.0 + exp(-x))


fn sigmoid_bw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = e^x / (1 + e^x)^2
    # Best is 6 instructions (exp, div, fma, exp, mul, add)
    let e_x = (exp(x))
    return e_x / (1.0 + e_x) ** 2


fn softplus_fw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = log(1 + e^x)
    # Best is 3 instructions (exp, add, log)
    return log(1.0 + exp(x))


fn softplus_bw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = e^x / (1 + e^x)
    # Best is 3 instructions (exp, add, div)
    let e_x = (exp(x))
    return e_x / (1.0 + e_x)


fn softsign_fw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x / (1 + |x|)
    # Best is 3 instructions (abs, add, div)
    return x / (1.0 + abs(x))


fn softsign_bw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = 1 / (1 + |x|)^2
    # Simplifies to 1 / (1 + x^2 + 2|x|)
    # Best is 4 instructions (div, abs, fma, fma)
    return 1.0 / abs(x).fma(2.0, x.fma(x, 1.0))


fn tanh_fw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = tanh(x)
    # Best is 1 instruction (tanh)
    return tanh(x)


fn tanh_bw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = 1 / cosh(x)^2
    # Best is 3 instructions (cosh, pow, div)
    return 1.0 / cosh(x) ** 2


fn selu_fw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x > 0 ? 1.05070098 * x : 1.05070098 * 1.67326324 * (e^x - 1)
    # Best is 5 instructions (compare, select, mul, exp, fma)
    return (x > 0.0).select(1.05070098 * x, exp(x).fma(1.75809932607, -1.75809932607))


fn selu_bw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = x > 0 ? 1.05070098 : 1.05070098 * 1.67326324 * e^x
    # Best is 4 instructions (compare, select, mul, exp)
    return (x > 0.0).select(1.05070098, 1.75809932607 * exp(x))


fn elu_fw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    alpha: Float32 = 1.0,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x > 0 ? x : alpha * (e^x - 1)
    # Best is 5 instructions (compare, select, mul, exp, sub), 4 if alpha == 1
    @parameter
    if alpha == 1.0:
        return (x > 0.0).select(x, exp(x) - 1.0)
    return (x > 0.0).select(x, alpha * (exp(x) - 1.0))


fn elu_bw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    alpha: Float32 = 1.0,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = x > 0 ? 1 : alpha * e^x
    # Best is 4 instructions (compare, select, mul, exp), 3 if alpha == 1
    @parameter
    if alpha == 1.0:
        return (x > 0.0).select(1.0, exp(x))
    return (x > 0.0).select(1.0, alpha * exp(x))


fn exp_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = e^x
    # Best is 1 instruction (exp)
    return exp(x)


fn silu_fw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x / (1 + e^-x)
    # Best is 4 instructions (div, add, exp, inverse)
    return x / (1.0 + exp(-x))


fn silu_bw_vec[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = (e^x * x + e^x + e^2x) / (e^x + 1)^2
    # Best is 8 instructions (exp, fma, add, exp, mul, div, add, pow)
    let e_x = exp(x)
    return (e_x.fma(x, e_x) + exp(2.0 * x)) / (e_x + 1.0) ** 2


fn gelu_fw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    approximate: Float32 = 0.0,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) when approximate == 0.0 = 0.5 * x * (1 + erf(x / sqrt(2)))
    # f(x) when approximate != 0.0 = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    # Best is 6 instructions (mul, tanh, fma, mul, pow, fma), 4 if approximate == 0
    let x_05 = x * 0.5

    @parameter
    if approximate == 0.0:
        return erf(x / 1.4142135623730951).fma(x_05, x_05)
    return tanh(x.fma(0.7978845608028654, 0.03567740813 * x**3)).fma(x_05, x_05)


fn gelu_bw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    approximate: Float32 = 0.0,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
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


fn hsig_fw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x > 2.5 ? 1 : x < -2.5 ? 0 : 0.2 * x + 0.5
    # Best is 5 instructions (compare, select, compare, select, fma)
    return (x > 2.5).select(1.0, (x > -2.5).select(x.fma(0.2, 0.5), 0.0))


fn hsig_bw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = x > -2.5 ? x < 2.5 ? 0.2 : 0 : 0
    # Best is 5 instructions (compare, and, compare, cast, mul)
    return ((x > -2.5) & (x < 2.5)).select[DType.float32](0.2, 0.0)


fn linear_fw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x
    # Best is 1 instruction (mov)
    return x


fn linear_bw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = 1
    # Best is 1 instruction (mov)
    return 1.0


fn mish_fw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f(x) = x * tanh(log(1 + e^x))
    # Best is 5 instructions (mul, tanh, log, add, exp)
    return x * tanh(log(1.0 + exp(x)))


fn mish_bw_vec[
    NELTS: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    # f'(x) = tanh(log(exp(x) + 1)) + (x * exp(x) * (1 / cosh(ln(exp(x) + 1)) ^ 2)) / (exp(x) + 1)
    # Best is 14 instructions (exp, tanh, log, add, add, mul, mul, div, cosh, log, add, pow, div, add)
    let e_x = exp(x)
    return tanh(log(e_x + 1)) + (x * e_x * (1 / cosh(log(e_x + 1)) ** 2)) / (e_x + 1)
