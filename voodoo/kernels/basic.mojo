from ..constants import DType_F32, nelts, f32_max
from .generics import (
    GenericActivation,
    GenericArithmetic,
    GenericBinaryArithmetic,
    GenericLoss,
    GenericOptimizer,
)
from math import (
    sqrt,
    exp2,
    log2,
    log,
    cos,
    sin,
    tan,
    asin,
    acos,
    atan,
    cosh,
    sinh,
    tanh,
    erf,
    abs,
    max,
    exp,
)


struct Relu[arg1: Float32, arg2: Float32, arg3: Float32]:
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].bw


struct Sigmoid[]:
    alias fw = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].bw


struct Softplus[]:
    alias fw = GenericActivation[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].bw


struct Softsign[]:
    alias fw = GenericActivation[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].bw


struct Tanh[]:
    alias fw = GenericActivation[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].bw


struct Selu[]:
    alias fw = GenericActivation[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].bw


struct Elu[alpha: Float32]:
    alias fw = GenericActivation[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].fw
    alias bw = GenericActivation[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].bw


struct Exp[]:
    alias fw = GenericActivation[exp_vec, exp_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[exp_vec, exp_vec, 0.0, 0.0, 0.0].bw


struct LeakyRelu[alpha: Float32]:
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, alpha, f32_max, 0.0].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, alpha, f32_max, 0.0].bw


struct Relu6[]:
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].bw


struct Silu[]:
    alias fw = GenericActivation[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].bw


struct Gelu[approximate: Float32]:
    alias fw = GenericActivation[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].fw
    alias bw = GenericActivation[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].bw


struct HardSigmoid[]:
    alias fw = GenericActivation[hsig_fw_vec, hsig_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[hsig_fw_vec, hsig_bw_vec, 0.0, 0.0, 0.0].bw


struct Linear[]:
    alias fw = GenericActivation[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].bw


struct Mish[]:
    alias fw = GenericActivation[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].bw


struct Sqrt[]:
    alias fw = GenericArithmetic[sqrt_fw_vec, sqrt_bw_vec].fw
    alias bw = GenericArithmetic[sqrt_fw_vec, sqrt_bw_vec].bw


struct Abs[]:
    alias fw = GenericArithmetic[abs_fw_vec, abs_bw_vec].fw
    alias bw = GenericArithmetic[abs_fw_vec, abs_bw_vec].bw


struct Exp2[]:
    alias fw = GenericArithmetic[exp2_fw_vec, exp2_bw_vec].fw
    alias bw = GenericArithmetic[exp2_fw_vec, exp2_bw_vec].bw


struct Log2[]:
    alias fw = GenericArithmetic[log2_fw_vec, log2_bw_vec].fw
    alias bw = GenericArithmetic[log2_fw_vec, log2_bw_vec].bw


struct Log[]:
    alias fw = GenericArithmetic[log_fw_vec, log_bw_vec].fw
    alias bw = GenericArithmetic[log_fw_vec, log_bw_vec].bw


struct Sin[]:
    alias fw = GenericArithmetic[sin_fw_vec, sin_bw_vec].fw
    alias bw = GenericArithmetic[sin_fw_vec, sin_bw_vec].bw


struct Cos[]:
    alias fw = GenericArithmetic[cos_fw_vec, cos_bw_vec].fw
    alias bw = GenericArithmetic[cos_fw_vec, cos_bw_vec].bw


struct Tan[]:
    alias fw = GenericArithmetic[tan_fw_vec, tan_bw_vec].fw
    alias bw = GenericArithmetic[tan_fw_vec, tan_bw_vec].bw


struct Asin[]:
    alias fw = GenericArithmetic[asin_fw_vec, asin_bw_vec].fw
    alias bw = GenericArithmetic[asin_fw_vec, asin_bw_vec].bw


struct Acos[]:
    alias fw = GenericArithmetic[acos_fw_vec, acos_bw_vec].fw
    alias bw = GenericArithmetic[acos_fw_vec, acos_bw_vec].bw


struct Atan[]:
    alias fw = GenericArithmetic[atan_fw_vec, atan_bw_vec].fw
    alias bw = GenericArithmetic[atan_fw_vec, atan_bw_vec].bw


struct Sinh[]:
    alias fw = GenericArithmetic[sinh_fw_vec, sinh_bw_vec].fw
    alias bw = GenericArithmetic[sinh_fw_vec, sinh_bw_vec].bw


struct Cosh[]:
    alias fw = GenericArithmetic[cosh_fw_vec, cosh_bw_vec].fw
    alias bw = GenericArithmetic[cosh_fw_vec, cosh_bw_vec].bw


struct Add[]:
    alias fw = GenericBinaryArithmetic[add_fw, add_bw, add_bw].fw
    alias bw = GenericBinaryArithmetic[add_fw, add_bw, add_bw].bw


struct Sub[]:
    alias fw = GenericBinaryArithmetic[sub_fw, sub_bw_a, sub_bw_b].fw
    alias bw = GenericBinaryArithmetic[sub_fw, sub_bw_a, sub_bw_b].bw


struct Mul[]:
    alias fw = GenericBinaryArithmetic[mul_fw, mul_bw_a, mul_bw_b].fw
    alias bw = GenericBinaryArithmetic[mul_fw, mul_bw_a, mul_bw_b].bw


struct Div[]:
    alias fw = GenericBinaryArithmetic[div_fw, div_bw_a, div_bw_b].fw
    alias bw = GenericBinaryArithmetic[div_fw, div_bw_a, div_bw_b].bw


struct Pow[]:
    alias fw = GenericBinaryArithmetic[pow_fw, pow_bw_a, pow_bw_b].fw
    alias bw = GenericBinaryArithmetic[pow_fw, pow_bw_a, pow_bw_b].bw


struct MSE[]:
    alias fw = GenericLoss[mse_error, mse_grad].fw
    alias bw = GenericLoss[mse_error, mse_grad].bw


struct MAE[]:
    alias fw = GenericLoss[mae_error, mae_grad].fw
    alias bw = GenericLoss[mae_error, mae_grad].bw


struct MAPE[]:
    alias fw = GenericLoss[mape_error, mape_grad].fw
    alias bw = GenericLoss[mape_error, mape_grad].bw


struct MSLE[]:
    alias fw = GenericLoss[msle_error, msle_grad].fw
    alias bw = GenericLoss[msle_error, msle_grad].bw


struct SGD[learning_rate: Float32]:
    alias step = GenericOptimizer[sgd].step[learning_rate]


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
fn hsig_fw_vec[
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
fn hsig_bw_vec[
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
fn sqrt_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sqrt(x)


@parameter
@always_inline
fn sqrt_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 0.5 / sqrt(x)


@parameter
@always_inline
fn abs_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return (x > 0).select(x, -x)


@parameter
@always_inline
fn abs_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return (x > 0).select(Float32(1.0), Float32(-1.0))


@parameter
@always_inline
fn exp2_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return exp2(x)


@parameter
@always_inline
fn exp2_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return x * 0.69314718056


@parameter
@always_inline
fn log2_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return log2(x)


@parameter
@always_inline
fn log2_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / (x * 0.69314718056)


@parameter
@always_inline
fn log_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return log(x)


@parameter
@always_inline
fn log_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / x


@parameter
@always_inline
fn sin_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sin(x)


@parameter
@always_inline
fn sin_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cos(x)


@parameter
@always_inline
fn cos_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cos(x)


@parameter
@always_inline
fn cos_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return -sin(x)


@parameter
@always_inline
fn tan_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return tan(x)


@parameter
@always_inline
fn tan_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / (cos(x) ** 2)


@parameter
@always_inline
fn asin_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return asin(x)


@parameter
@always_inline
fn asin_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / sqrt(1.0 - x**2)


@parameter
@always_inline
fn acos_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return acos(x)


@parameter
@always_inline
fn acos_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return -1.0 / sqrt(1.0 - x**2)


@parameter
@always_inline
fn atan_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return atan(x)


@parameter
@always_inline
fn atan_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / (1.0 + x**2)


@parameter
@always_inline
fn sinh_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sinh(x)


@parameter
@always_inline
fn sinh_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cosh(x)


@parameter
@always_inline
fn cosh_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cosh(x)


@parameter
@always_inline
fn cosh_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sinh(x)


@parameter
@always_inline
fn add_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x, y) = x + y
    return a + b


@parameter
@always_inline
fn add_bw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) = 1
    return 1


@parameter
@always_inline
fn sub_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x, y) = x - y
    return a - b


@parameter
@always_inline
fn sub_bw_a[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = 1
    return 1


@parameter
@always_inline
fn sub_bw_b[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1
    return -1


@parameter
@always_inline
fn mul_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x, y) = x * y
    return a * b


@parameter
@always_inline
fn mul_bw_a[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = y
    return b


@parameter
@always_inline
fn mul_bw_b[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = x
    return a


@parameter
@always_inline
fn div_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x, y) = x / y
    return a / b


@parameter
@always_inline
fn div_bw_a[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = 1/y
    return 1 / b


@parameter
@always_inline
fn div_bw_b[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -x/y^2
    return -a / (b * b)


@parameter
@always_inline
fn pow_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    # f(x, y) = x^y
    return a**b


@parameter
@always_inline
fn pow_bw_a[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = y * x^(y-1)
    return b * (a ** (b - 1.0))


@parameter
@always_inline
fn pow_bw_b[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts],) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = x^y * log(x)
    return (a**b) * log(a)


@parameter
@always_inline
fn mse_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = (x - y)^2
    return (y_pred - y_true) ** 2.0


@parameter
@always_inline
fn mse_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -2(x - y)
    return -2.0 * (y_pred - y_true)


@parameter
@always_inline
fn mae_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = |x - y|
    return abs(y_pred - y_true)


@parameter
@always_inline
fn mae_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).select(Float32(-1.0), 1.0)


@parameter
@always_inline
fn mape_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = |x - y| / y
    return abs(y_pred - y_true) / (y_true + epsilon)


@parameter
@always_inline
fn mape_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).cast[DType_F32]() * Float32(-2.0) + Float32(1.0)


@parameter
@always_inline
fn msle_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = (log(x + 1) - log(y + 1))^2
    let y_pred_clipped = (y_pred > 0.0).cast[DType_F32]() * y_pred
    let y_true_clipped = (y_true > 0.0).cast[DType_F32]() * y_true
    return (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))) * (
        log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))
    )


@parameter
@always_inline
fn msle_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -2(log(x + 1) - log(y + 1)) / (y + 1)
    let y_pred_clipped = (y_pred > 0.0).cast[DType_F32]() * y_pred
    let y_true_clipped = (y_true > 0.0).cast[DType_F32]() * y_true
    return (
        -Float32(2.0)
        * (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0)))
        / (y_true_clipped + Float32(1.0))
    )


@parameter
@always_inline
fn sgd[
    nelts: Int, learning_rate: Float32
](grad: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return grad * learning_rate
