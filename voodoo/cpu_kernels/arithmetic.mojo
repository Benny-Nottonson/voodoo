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


fn sqrt_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sqrt(x)


fn sqrt_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 0.5 / sqrt(x)


alias Sqrt = GenericUnaryArithmetic[sqrt_fw_vec, sqrt_bw_vec]


fn abs_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x >= 0.0).cast[DType_F32]() * x + (x < 0.0).cast[DType_F32]() * (-x)


fn abs_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 2.0 * (x >= 0.0).cast[DType_F32]() - 1.0


alias Abs = GenericUnaryArithmetic[abs_fw_vec, abs_bw_vec]


fn exp2_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return exp2(x)


fn exp2_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x * 0.69314718056


alias Exp2 = GenericUnaryArithmetic[exp2_fw_vec, exp2_bw_vec]


fn log2_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return log2(x)


fn log2_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (x * 0.69314718056)


alias Log2 = GenericUnaryArithmetic[log2_fw_vec, log2_bw_vec]


fn log_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return log(x)


fn log_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / x


alias Log = GenericUnaryArithmetic[log_fw_vec, log_bw_vec]


fn sin_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sin(x)


fn sin_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cos(x)


alias Sin = GenericUnaryArithmetic[sin_fw_vec, sin_bw_vec]


fn cos_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cos(x)


fn cos_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return -sin(x)


alias Cos = GenericUnaryArithmetic[cos_fw_vec, cos_bw_vec]


fn tan_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return tan(x)


fn tan_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (cos(x) ** 2)


alias Tan = GenericUnaryArithmetic[tan_fw_vec, tan_bw_vec]


fn asin_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return asin(x)


fn asin_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / sqrt(1.0 - x**2)


alias Asin = GenericUnaryArithmetic[asin_fw_vec, asin_bw_vec]


fn acos_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return acos(x)


fn acos_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return -1.0 / sqrt(1.0 - x**2)


alias Acos = GenericUnaryArithmetic[acos_fw_vec, acos_bw_vec]


fn atan_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return atan(x)


fn atan_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (1.0 + x**2)


alias Atan = GenericUnaryArithmetic[atan_fw_vec, atan_bw_vec]


fn sinh_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sinh(x)


fn sinh_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cosh(x)


alias Sinh = GenericUnaryArithmetic[sinh_fw_vec, sinh_bw_vec]


fn cosh_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return cosh(x)


fn cosh_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return sinh(x)


alias Cosh = GenericUnaryArithmetic[cosh_fw_vec, cosh_bw_vec]
