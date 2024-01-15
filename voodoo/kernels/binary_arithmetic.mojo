from math import max, log
from algorithm import vectorize, parallelize
from voodoo import Node
from voodoo.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import DType_F32, nelts, workers

alias generic_vectorized = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
) -> SIMD[DType_F32, nelts]


struct Generic[
    fw_vec: generic_vectorized,
    bw_a_vec: generic_vectorized,
    bw_b_vec: generic_vectorized,
]:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_fw[fw_vec], base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_bw[bw_a_vec, True], base_case_strides](
                c, a, b
            )
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_bw[bw_b_vec, False], base_case_strides](
                c, a, b
            )

    @parameter
    @staticmethod
    fn kernel_fw[
        generic_func: generic_vectorized
    ](
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_fw[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                generic_func(
                    a.load_data[nelts](offset_a + i), b.load_data[nelts](offset_b + i)
                ),
            )

        vectorize[nelts, vectorized_fw](c_rest)

    @parameter
    @staticmethod
    fn kernel_bw[
        generic_func: generic_vectorized,
        is_a: Bool,
    ](
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_bw_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i)
                + generic_func(
                    a.load_data[nelts](offset_a + i),
                    b.load_data[nelts](offset_b + i),
                )
                * c.load_grad[nelts](offset_c + i),
            )

        @parameter
        fn vectorized_bw_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i)
                + generic_func(
                    a.load_data[nelts](offset_a + i),
                    b.load_data[nelts](offset_b + i),
                )
                * c.load_grad[nelts](offset_c + i),
            )

        @parameter
        if is_a:
            vectorize[nelts, vectorized_bw_a](c_rest)
        else:
            vectorize[nelts, vectorized_bw_b](c_rest)


alias Add = Generic[add_fw, add_bw, add_bw]
alias Sub = Generic[sub_fw, sub_bw_a, sub_bw_b]
alias Mul = Generic[mul_fw, mul_bw_a, mul_bw_b]
alias Div = Generic[div_fw, div_bw_a, div_bw_b]
alias Pow = Generic[pow_fw, pow_bw_a, pow_bw_b]


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
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
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
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to x = 1
    return 1


@parameter
@always_inline
fn sub_bw_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
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
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to x = y
    return b_data


@parameter
@always_inline
fn mul_bw_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to y = x
    return a_data


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
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to x = 1/y
    return 1 / b_data


@parameter
@always_inline
fn div_bw_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to y = -x/y^2
    return -a_data / (b_data * b_data)


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
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to x = y * x^(y-1)
    return b_data * (a_data ** (b_data - 1.0))


@parameter
@always_inline
fn pow_bw_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
) -> SIMD[
    DType_F32, nelts
]:
    # f'(x, y) with respect to y = x^y * log(x)
    return (a_data**b_data) * log(a_data)


@parameter
@always_inline
fn base_case_strides(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)
