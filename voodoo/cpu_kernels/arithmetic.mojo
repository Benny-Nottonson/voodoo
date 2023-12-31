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
)
from algorithm import vectorize
from voodoo import Node
from .constants import DType_F32, nelts

# TODO: Rewrite when lambda functions are supported
# TODO: Add comments for each function and check for optimized estimatorss


alias generic_vectorized = fn[_nelts: Int] (SIMD[DType_F32, _nelts]) -> SIMD[
    DType_F32, _nelts
]


struct GenericUnaryArithmetic[fw_vec: generic_vectorized, bw_vec: generic_vectorized]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn generic_vectorized_fw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                fw_vec[_nelts](x),
            )

        vectorize[nelts, generic_vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn generic_vectorized_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * bw_vec[_nelts](x),
            )

        vectorize[nelts, generic_vectorized_bw](node.load_cap())


alias Sqrt = GenericUnaryArithmetic[sqrt_fw_vec, sqrt_bw_vec]
alias Abs = GenericUnaryArithmetic[abs_fw_vec, abs_bw_vec]
alias Exp2 = GenericUnaryArithmetic[exp2_fw_vec, exp2_bw_vec]
alias Log2 = GenericUnaryArithmetic[log2_fw_vec, log2_bw_vec]
alias Log = GenericUnaryArithmetic[log_fw_vec, log_bw_vec]
alias Sin = GenericUnaryArithmetic[sin_fw_vec, sin_bw_vec]
alias Cos = GenericUnaryArithmetic[cos_fw_vec, cos_bw_vec]
alias Tan = GenericUnaryArithmetic[tan_fw_vec, tan_bw_vec]
alias Asin = GenericUnaryArithmetic[asin_fw_vec, asin_bw_vec]
alias Acos = GenericUnaryArithmetic[acos_fw_vec, acos_bw_vec]
alias Atan = GenericUnaryArithmetic[atan_fw_vec, atan_bw_vec]
alias Sinh = GenericUnaryArithmetic[sinh_fw_vec, sinh_bw_vec]
alias Cosh = GenericUnaryArithmetic[cosh_fw_vec, cosh_bw_vec]


@parameter
fn sqrt_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sqrt(x)


@parameter
fn sqrt_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 0.5 / sqrt(x)


@parameter
fn abs_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x >= 0.0).cast[DType_F32]() * x + (x < 0.0).cast[DType_F32]() * (-x)


@parameter
fn abs_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 2.0 * (x >= 0.0).cast[DType_F32]() - 1.0


@parameter
fn exp2_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return exp2(x)


@parameter
fn exp2_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x * 0.69314718056


@parameter
fn log2_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return log2(x)


@parameter
fn log2_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (x * 0.69314718056)


@parameter
fn log_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return log(x)


@parameter
fn log_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / x


@parameter
fn sin_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sin(x)


@parameter
fn sin_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cos(x)


@parameter
fn cos_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cos(x)


@parameter
fn cos_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return -sin(x)


@parameter
fn tan_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return tan(x)


@parameter
fn tan_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (cos(x) ** 2)


@parameter
fn asin_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return asin(x)


@parameter
fn asin_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / sqrt(1.0 - x**2)


@parameter
fn acos_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return acos(x)


@parameter
fn acos_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return -1.0 / sqrt(1.0 - x**2)


@parameter
fn atan_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return atan(x)


@parameter
fn atan_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (1.0 + x**2)


@parameter
fn sinh_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sinh(x)


@parameter
fn sinh_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cosh(x)


@parameter
fn cosh_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cosh(x)


@parameter
fn cosh_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sinh(x)
