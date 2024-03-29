from math import sqrt, exp2, exp, log2, sin, cos, tan, log, asin, acos, atan, sinh, cosh

from voodoo.autograd.kernels.generics import GenericArithmetic, GenericBinaryArithmetic


trait Aritmetic:
    ...


@register_passable("trivial")
struct Sqrt[](Aritmetic):
    alias fw = GenericArithmetic[sqrt_fw_vec, sqrt_bw_vec].fw
    alias bw = GenericArithmetic[sqrt_fw_vec, sqrt_bw_vec].bw


@register_passable("trivial")
struct Abs[](Aritmetic):
    alias fw = GenericArithmetic[abs_fw_vec, abs_bw_vec].fw
    alias bw = GenericArithmetic[abs_fw_vec, abs_bw_vec].bw


@register_passable("trivial")
struct Exp2[](Aritmetic):
    alias fw = GenericArithmetic[exp2_fw_vec, exp2_bw_vec].fw
    alias bw = GenericArithmetic[exp2_fw_vec, exp2_bw_vec].bw


@register_passable("trivial")
struct Log2[](Aritmetic):
    alias fw = GenericArithmetic[log2_fw_vec, log2_bw_vec].fw
    alias bw = GenericArithmetic[log2_fw_vec, log2_bw_vec].bw


@register_passable("trivial")
struct Log[](Aritmetic):
    alias fw = GenericArithmetic[log_fw_vec, log_bw_vec].fw
    alias bw = GenericArithmetic[log_fw_vec, log_bw_vec].bw


@register_passable("trivial")
struct Sin[](Aritmetic):
    alias fw = GenericArithmetic[sin_fw_vec, sin_bw_vec].fw
    alias bw = GenericArithmetic[sin_fw_vec, sin_bw_vec].bw


@register_passable("trivial")
struct Cos[](Aritmetic):
    alias fw = GenericArithmetic[cos_fw_vec, cos_bw_vec].fw
    alias bw = GenericArithmetic[cos_fw_vec, cos_bw_vec].bw


@register_passable("trivial")
struct Tan[](Aritmetic):
    alias fw = GenericArithmetic[tan_fw_vec, tan_bw_vec].fw
    alias bw = GenericArithmetic[tan_fw_vec, tan_bw_vec].bw


@register_passable("trivial")
struct Asin[](Aritmetic):
    alias fw = GenericArithmetic[asin_fw_vec, asin_bw_vec].fw
    alias bw = GenericArithmetic[asin_fw_vec, asin_bw_vec].bw


@register_passable("trivial")
struct Acos[](Aritmetic):
    alias fw = GenericArithmetic[acos_fw_vec, acos_bw_vec].fw
    alias bw = GenericArithmetic[acos_fw_vec, acos_bw_vec].bw


@register_passable("trivial")
struct Atan[](Aritmetic):
    alias fw = GenericArithmetic[atan_fw_vec, atan_bw_vec].fw
    alias bw = GenericArithmetic[atan_fw_vec, atan_bw_vec].bw


@register_passable("trivial")
struct Sinh[](Aritmetic):
    alias fw = GenericArithmetic[sinh_fw_vec, sinh_bw_vec].fw
    alias bw = GenericArithmetic[sinh_fw_vec, sinh_bw_vec].bw


@register_passable("trivial")
struct Cosh[](Aritmetic):
    alias fw = GenericArithmetic[cosh_fw_vec, cosh_bw_vec].fw
    alias bw = GenericArithmetic[cosh_fw_vec, cosh_bw_vec].bw


@register_passable("trivial")
struct Add[](Aritmetic):
    alias fw = GenericBinaryArithmetic[add_fw, add_bw, add_bw].fw
    alias bw = GenericBinaryArithmetic[add_fw, add_bw, add_bw].bw


@register_passable("trivial")
struct Sub[](Aritmetic):
    alias fw = GenericBinaryArithmetic[sub_fw, sub_bw_a, sub_bw_b].fw
    alias bw = GenericBinaryArithmetic[sub_fw, sub_bw_a, sub_bw_b].bw


@register_passable("trivial")
struct Mul[](Aritmetic):
    alias fw = GenericBinaryArithmetic[mul_fw, mul_bw_a, mul_bw_b].fw
    alias bw = GenericBinaryArithmetic[mul_fw, mul_bw_a, mul_bw_b].bw


@register_passable("trivial")
struct Div[](Aritmetic):
    alias fw = GenericBinaryArithmetic[div_fw, div_bw_a, div_bw_b].fw
    alias bw = GenericBinaryArithmetic[div_fw, div_bw_a, div_bw_b].bw


@register_passable("trivial")
struct Pow[](Aritmetic):
    alias fw = GenericBinaryArithmetic[pow_fw, pow_bw_a, pow_bw_b].fw
    alias bw = GenericBinaryArithmetic[pow_fw, pow_bw_a, pow_bw_b].bw


fn sqrt_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sqrt(x)


fn sqrt_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 0.5 / sqrt(x)


fn abs_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return (x > 0).select(x, -x)


fn abs_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return (x > 0).select(Float32(1.0), Float32(-1.0))


fn exp2_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return exp2(x)


fn exp2_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return x * 0.69314718056


fn log2_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return log2(x)


fn log2_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / (x * 0.69314718056)


fn log_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return log(x)


fn log_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / x


fn sin_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sin(x)


fn sin_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cos(x)


fn cos_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cos(x)


fn cos_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return -sin(x)


fn tan_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return tan(x)


fn tan_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / (cos(x) ** 2)


fn asin_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return asin(x)


fn asin_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / sqrt(1.0 - x**2)


fn acos_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return acos(x)


fn acos_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return -1.0 / sqrt(1.0 - x**2)


fn atan_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return atan(x)


fn atan_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return 1.0 / (1.0 + x**2)


fn sinh_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sinh(x)


fn sinh_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cosh(x)


fn cosh_fw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return cosh(x)


fn cosh_bw_vec[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return sinh(x)


fn add_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x + y
    return a + b


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


fn sub_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x - y
    return a - b


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


fn mul_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x * y
    return a * b


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


fn div_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x / y
    return a / b


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


fn pow_fw[
    NELTS: Int
](a: SIMD[DType.float32, NELTS], b: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = x^y
    return a**b


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
