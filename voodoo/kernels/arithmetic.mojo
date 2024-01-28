from .generics import GenericArithmetic, GenericBinaryArithmetic
from math import sqrt, exp2, exp, log2, sin, cos, tan, log, asin, acos, atan, sinh, cosh


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


@always_inline("nodebug")
fn sqrt_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sqrt(x)


@always_inline("nodebug")
fn sqrt_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 0.5 / sqrt(x)


@always_inline("nodebug")
fn abs_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return (x > 0).select(x, -x)


@always_inline("nodebug")
fn abs_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return (x > 0).select(Float32(1.0), Float32(-1.0))


@always_inline("nodebug")
fn exp2_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return exp2(x)


@always_inline("nodebug")
fn exp2_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return x * 0.69314718056


@always_inline("nodebug")
fn log2_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return log2(x)


@always_inline("nodebug")
fn log2_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / (x * 0.69314718056)


@always_inline("nodebug")
fn log_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return log(x)


@always_inline("nodebug")
fn log_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / x


@always_inline("nodebug")
fn sin_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sin(x)


@always_inline("nodebug")
fn sin_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cos(x)


@always_inline("nodebug")
fn cos_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cos(x)


@always_inline("nodebug")
fn cos_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return -sin(x)


@always_inline("nodebug")
fn tan_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return tan(x)


@always_inline("nodebug")
fn tan_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / (cos(x) ** 2)


@always_inline("nodebug")
fn asin_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return asin(x)


@always_inline("nodebug")
fn asin_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / sqrt(1.0 - x**2)


@always_inline("nodebug")
fn acos_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return acos(x)


@always_inline("nodebug")
fn acos_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return -1.0 / sqrt(1.0 - x**2)


@always_inline("nodebug")
fn atan_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return atan(x)


@always_inline("nodebug")
fn atan_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / (1.0 + x**2)


@always_inline("nodebug")
fn sinh_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sinh(x)


@always_inline("nodebug")
fn sinh_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cosh(x)


@always_inline("nodebug")
fn cosh_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cosh(x)


@always_inline("nodebug")
fn cosh_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sinh(x)


@always_inline("nodebug")
fn add_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x + y
    return a + b


@always_inline("nodebug")
fn add_bw[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) = 1
    return 1


@always_inline("nodebug")
fn sub_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x - y
    return a - b


@always_inline("nodebug")
fn sub_bw_a[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to x = 1
    return 1


@always_inline("nodebug")
fn sub_bw_b[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to y = -1
    return -1


@always_inline("nodebug")
fn mul_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x * y
    return a * b


@always_inline("nodebug")
fn mul_bw_a[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to x = y
    return b


@always_inline("nodebug")
fn mul_bw_b[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to y = x
    return a


@always_inline("nodebug")
fn div_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x / y
    return a / b


@always_inline("nodebug")
fn div_bw_a[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to x = 1/y
    return 1 / b


@always_inline("nodebug")
fn div_bw_b[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to y = -x/y^2
    return -a / (b * b)


@always_inline("nodebug")
fn pow_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x^y
    return a**b


@always_inline("nodebug")
fn pow_bw_a[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to x = y * x^(y-1)
    return b * (a ** (b - 1.0))


@always_inline("nodebug")
fn pow_bw_b[
    NELTS: Int
](
    a: SIMD[DType.float32, NELTS],
    b: SIMD[DType.float32, NELTS],
) -> SIMD[
    DType.float32, NELTS
]:
    # f'(x, y) with respect to y = x^y * log(x)
    return (a**b) * log(a)
