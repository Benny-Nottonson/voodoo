from math import max, log
from algorithm import vectorize, parallelize
from voodoo.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)
from voodoo import Node
from ..constants import DType_F32, nelts, workers

# TODO: Rewrite to use generic functions where possible

alias generic_vectorized = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
) -> SIMD[DType_F32, nelts]

alias generic_vectorized_bw = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
) -> SIMD[DType_F32, nelts]


@parameter
fn base_case_strides(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


struct GenericBinaryArithmetic[
    generic_func: generic_vectorized,
    kernel_bw_a: generic_vectorized_bw,
    kernel_bw_b: generic_vectorized_bw,
]:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[kernel_generic_fw[generic_func], base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[kernel_generic_bw_a[kernel_bw_a], base_case_strides](
                c, a, b
            )
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[kernel_generic_bw_b[kernel_bw_b], base_case_strides](
                c, a, b
            )


alias Add = GenericBinaryArithmetic[add_fw, bw_add, bw_add]
alias Mul = GenericBinaryArithmetic[mul_fw, bw_mul_a, bw_mul_b]
alias Sub = GenericBinaryArithmetic[sub_fw, bw_sub, bw_sub]
alias Div = GenericBinaryArithmetic[div_fw, bw_div_a, bw_div_b]
alias Pow = GenericBinaryArithmetic[pow_fw, bw_pow_a, bw_pow_b]
alias MMul = _MMul


@parameter
fn kernel_generic_fw[
    generic_func: fn[nelts: Int] (
        SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
    ) -> SIMD[DType_F32, nelts]
](
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn vectorized_generic[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            generic_func(
                a.load_data[nelts](offset_a + i), b.load_data[nelts](offset_b + i)
            ),
        )

    vectorize[nelts, vectorized_generic](c_rest)


@parameter
fn kernel_generic_bw_a[
    generic_func: fn[nelts: Int] (
        SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
    ) -> SIMD[DType_F32, nelts]
](
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn vectorized_generic[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + generic_func(
                a.load_data[nelts](offset_a + i),
                b.load_data[nelts](offset_b + i),
                c.load_grad[nelts](offset_c + i),
            ),
        )

    vectorize[nelts, vectorized_generic](c_rest)


@parameter
fn kernel_generic_bw_b[
    generic_func: fn[nelts: Int] (
        SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
    ) -> SIMD[DType_F32, nelts]
](
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn vectorized_generic[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            + generic_func(
                a.load_data[nelts](offset_a + i),
                b.load_data[nelts](offset_b + i),
                c.load_grad[nelts](offset_c + i),
            ),
        )

    vectorize[nelts, vectorized_generic](c_rest)


@parameter
fn add_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return a + b


@parameter
fn mul_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return a * b


@parameter
fn sub_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return a - b


@parameter
fn div_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return a / b


@parameter
fn pow_fw[
    nelts: Int
](a: SIMD[DType_F32, nelts], b: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return a**b


@parameter
fn bw_add[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return c_grad


@parameter
fn bw_mul_a[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return b_data * c_grad


@parameter
fn bw_sub[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return -c_grad


@parameter
fn bw_div_a[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return c_grad / b_data


@parameter
fn bw_pow_a[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return b_data * (a_data ** (b_data - 1.0)) * c_grad


@parameter
fn bw_mul_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return a_data * c_grad


@parameter
fn bw_div_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return -a_data * c_grad / (b_data**2)


@parameter
fn bw_pow_b[
    nelts: Int
](
    a_data: SIMD[DType_F32, nelts],
    b_data: SIMD[DType_F32, nelts],
    c_grad: SIMD[DType_F32, nelts],
) -> SIMD[DType_F32, nelts]:
    return a_data**b_data * log(a_data) * c_grad


struct _MMul:
    @parameter
    @staticmethod
    fn base_case_depth(depth: Int, a: Node, b: Node) -> Bool:
        return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2

    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_mmul_fw, Self.base_case_depth](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mmul_bw_a, Self.base_case_depth](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mmul_bw_b, Self.base_case_depth](c, a, b)

    @parameter
    @staticmethod
    fn kernel_mmul_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * a.shape_ptr.load().load(
            a.num_dims_ptr.load() - 2
        ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * b.shape_ptr.load().load(
            b.num_dims_ptr.load() - 2
        ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * c.shape_ptr.load().load(
            c.num_dims_ptr.load() - 2
        ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
        let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
        let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

        @parameter
        fn calc_row_fw(m: Int):
            for k in range(K):

                @parameter
                fn dot_fw[_nelts: Int](n: Int):
                    c.store_data[_nelts](
                        offset_c + m * N + n,
                        c.load_data[_nelts](offset_c + m * N + n)
                        + a.load_data(offset_a + m * K + k)
                        * b.load_data[_nelts](offset_b + k * N + n),
                    )

                vectorize[nelts, dot_fw](N)

        parallelize[calc_row_fw](M, workers if workers > 0 else M)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * a.shape_ptr.load().load(
            a.num_dims_ptr.load() - 2
        ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * b.shape_ptr.load().load(
            b.num_dims_ptr.load() - 2
        ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * c.shape_ptr.load().load(
            c.num_dims_ptr.load() - 2
        ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
        let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
        let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

        @parameter
        fn calc_row_1(m: Int):
            for n in range(N):

                @parameter
                fn dot_bw_a[nelts: Int](k: Int):
                    let val = a.load_grad(offset_a + m * K + k) + c.load_grad(
                        offset_c + m * N + n
                    ) * b.load_data(offset_b + k * N + n)
                    a.store_grad(offset_a + m * K + k, val)

                vectorize[1, dot_bw_a](K)

        parallelize[calc_row_1](M, workers if workers > 0 else M)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * a.shape_ptr.load().load(
            a.num_dims_ptr.load() - 2
        ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * b.shape_ptr.load().load(
            b.num_dims_ptr.load() - 2
        ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * c.shape_ptr.load().load(
            c.num_dims_ptr.load() - 2
        ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
        let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
        let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

        @parameter
        fn calc_row_2(k: Int):
            for m in range(M):

                @parameter
                fn dot_bw_b[nelts: Int](n: Int):
                    let val = b.load_grad(offset_b + k * N + n) + a.load_data(
                        offset_a + m * K + k
                    ) * c.load_grad(offset_c + m * N + n)
                    b.store_grad(offset_b + k * N + n, val)

                vectorize[1, dot_bw_b](N)

        parallelize[calc_row_2](K, workers if workers > 0 else K)
