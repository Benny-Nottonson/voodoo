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
from ..constants import DType_F32, nelts

# TODO: Rewrite when lambda functions are supported

alias generic_vectorized = fn[nelts: Int] (SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]


struct Generic[fw_vec: generic_vectorized, bw_vec: generic_vectorized]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_fw[nelts: Int](i: Int):
            let x = parent1.load_data[nelts](i)
            node.store_data[nelts](
                i,
                fw_vec[nelts](x),
            )

        vectorize[nelts, vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_bw[nelts: Int](i: Int):
            let x = parent1.load_data[nelts](i)
            parent1.store_grad[nelts](
                i,
                parent1.load_grad[nelts](i)
                + node.load_grad[nelts](i) * bw_vec[nelts](x),
            )

        vectorize[nelts, vectorized_bw](node.load_cap())


alias Sqrt = Generic[sqrt_fw_vec, sqrt_bw_vec]
alias Abs = Generic[abs_fw_vec, abs_bw_vec]
alias Exp2 = Generic[exp2_fw_vec, exp2_bw_vec]
alias Log2 = Generic[log2_fw_vec, log2_bw_vec]
alias Log = Generic[log_fw_vec, log_bw_vec]
alias Sin = Generic[sin_fw_vec, sin_bw_vec]
alias Cos = Generic[cos_fw_vec, cos_bw_vec]
alias Tan = Generic[tan_fw_vec, tan_bw_vec]
alias Asin = Generic[asin_fw_vec, asin_bw_vec]
alias Acos = Generic[acos_fw_vec, acos_bw_vec]
alias Atan = Generic[atan_fw_vec, atan_bw_vec]
alias Sinh = Generic[sinh_fw_vec, sinh_bw_vec]
alias Cosh = Generic[cosh_fw_vec, cosh_bw_vec]


@parameter
fn sqrt_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sqrt(x)


@parameter
fn sqrt_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 0.5 / sqrt(x)


@parameter
fn abs_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return (x >= 0.0).cast[DType_F32]() * x + (x < 0.0).cast[DType_F32]() * (-x)


@parameter
fn abs_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 2.0 * (x >= 0.0).cast[DType_F32]() - 1.0


@parameter
fn exp2_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return exp2(x)


@parameter
fn exp2_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return x * 0.69314718056


@parameter
fn log2_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return log2(x)


@parameter
fn log2_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / (x * 0.69314718056)


@parameter
fn log_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return log(x)


@parameter
fn log_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / x


@parameter
fn sin_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sin(x)


@parameter
fn sin_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cos(x)


@parameter
fn cos_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cos(x)


@parameter
fn cos_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return -sin(x)


@parameter
fn tan_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return tan(x)


@parameter
fn tan_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / (cos(x) ** 2)


@parameter
fn asin_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return asin(x)


@parameter
fn asin_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / sqrt(1.0 - x**2)


@parameter
fn acos_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return acos(x)


@parameter
fn acos_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return -1.0 / sqrt(1.0 - x**2)


@parameter
fn atan_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return atan(x)


@parameter
fn atan_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return 1.0 / (1.0 + x**2)


@parameter
fn sinh_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sinh(x)


@parameter
fn sinh_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cosh(x)


@parameter
fn cosh_fw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return cosh(x)


@parameter
fn cosh_bw_vec[nelts: Int](x: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return sinh(x)
