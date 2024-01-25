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


@parameter
@always_inline
fn sqrt_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return sqrt(x)


@parameter
@always_inline
fn sqrt_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return 0.5 / sqrt(x)


@parameter
@always_inline
fn abs_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return (x > 0).select(x, -x)


@parameter
@always_inline
fn abs_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return (x > 0).select(Float32(1.0), Float32(-1.0))


@parameter
@always_inline
fn exp2_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return exp2(x)


@parameter
@always_inline
fn exp2_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return x * 0.69314718056


@parameter
@always_inline
fn log2_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return log2(x)


@parameter
@always_inline
fn log2_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return 1.0 / (x * 0.69314718056)


@parameter
@always_inline
fn log_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return log(x)


@parameter
@always_inline
fn log_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return 1.0 / x


@parameter
@always_inline
fn sin_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return sin(x)


@parameter
@always_inline
fn sin_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return cos(x)


@parameter
@always_inline
fn cos_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return cos(x)


@parameter
@always_inline
fn cos_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return -sin(x)


@parameter
@always_inline
fn tan_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return tan(x)


@parameter
@always_inline
fn tan_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return 1.0 / (cos(x) ** 2)


@parameter
@always_inline
fn asin_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return asin(x)


@parameter
@always_inline
fn asin_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return 1.0 / sqrt(1.0 - x**2)


@parameter
@always_inline
fn acos_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return acos(x)


@parameter
@always_inline
fn acos_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return -1.0 / sqrt(1.0 - x**2)


@parameter
@always_inline
fn atan_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return atan(x)


@parameter
@always_inline
fn atan_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return 1.0 / (1.0 + x**2)


@parameter
@always_inline
fn sinh_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return sinh(x)


@parameter
@always_inline
fn sinh_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return cosh(x)


@parameter
@always_inline
fn cosh_fw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return cosh(x)


@parameter
@always_inline
fn cosh_bw_vec[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return sinh(x)


@parameter
@always_inline
fn add_fw[
    nelts: Int
](a: SIMD[DType.float32, nelts], b: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    # f(x, y) = x + y
    return a + b


@parameter
@always_inline
fn add_bw[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) = 1
    return 1


@parameter
@always_inline
fn sub_fw[
    nelts: Int
](a: SIMD[DType.float32, nelts], b: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    # f(x, y) = x - y
    return a - b


@parameter
@always_inline
fn sub_bw_a[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to x = 1
    return 1


@parameter
@always_inline
fn sub_bw_b[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to y = -1
    return -1


@parameter
@always_inline
fn mul_fw[
    nelts: Int
](a: SIMD[DType.float32, nelts], b: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    # f(x, y) = x * y
    return a * b


@parameter
@always_inline
fn mul_bw_a[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to x = y
    return b


@parameter
@always_inline
fn mul_bw_b[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to y = x
    return a


@parameter
@always_inline
fn div_fw[
    nelts: Int
](a: SIMD[DType.float32, nelts], b: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    # f(x, y) = x / y
    return a / b


@parameter
@always_inline
fn div_bw_a[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to x = 1/y
    return 1 / b


@parameter
@always_inline
fn div_bw_b[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to y = -x/y^2
    return -a / (b * b)


@parameter
@always_inline
fn pow_fw[
    nelts: Int
](a: SIMD[DType.float32, nelts], b: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    # f(x, y) = x^y
    return a**b


@parameter
@always_inline
fn pow_bw_a[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to x = y * x^(y-1)
    return b * (a ** (b - 1.0))


@parameter
@always_inline
fn pow_bw_b[
    nelts: Int
](
    a: SIMD[DType.float32, nelts],
    b: SIMD[DType.float32, nelts],
) -> SIMD[
    DType.float32, nelts
]:
    # f'(x, y) with respect to y = x^y * log(x)
    return (a**b) * log(a)