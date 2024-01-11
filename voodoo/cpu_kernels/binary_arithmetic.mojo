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


@parameter
fn base_case_strides(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn base_case_depth(depth: Int, a: Node, b: Node) -> Bool:
    return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2


trait BinaryArithmetic:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        ...

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        ...


struct Add(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_add_fw, base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_add_bw_a, base_case_strides](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_add_bw_b, base_case_strides](c, a, b)

    @parameter
    @staticmethod
    fn kernel_add_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_add[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                a.load_data[nelts](offset_a + i) + b.load_data[nelts](offset_b + i),
            )

        vectorize[nelts, vectorized_add](c_rest)

    @parameter
    @staticmethod
    fn kernel_add_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_add_grad_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_add_grad_a](c_rest)

    @parameter
    @staticmethod
    fn kernel_add_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_add_grad_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i) + c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_add_grad_b](c_rest)


struct Mul(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_mul_fw, base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mul_bw_a, base_case_strides](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mul_bw_b, base_case_strides](c, a, b)

    @parameter
    @staticmethod
    fn kernel_mul_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_mul[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                a.load_data[nelts](offset_a + i) * b.load_data[nelts](offset_b + i),
            )

        vectorize[nelts, vectorized_mul](c_rest)

    @parameter
    @staticmethod
    fn kernel_mul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_mul_grad_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i)
                + b.load_data[nelts](offset_b + i) * c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_mul_grad_a](c_rest)

    @parameter
    @staticmethod
    fn kernel_mul_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_mul_grad_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i)
                + a.load_data[nelts](offset_a + i) * c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_mul_grad_b](c_rest)


struct Sub(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_sub_fw, base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_sub_bw_a, base_case_strides](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_sub_bw_b, base_case_strides](c, a, b)

    @parameter
    @staticmethod
    fn kernel_sub_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_sub[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                a.load_data[nelts](offset_a + i) - b.load_data[nelts](offset_b + i),
            )

        vectorize[nelts, vectorized_sub](c_rest)

    @parameter
    @staticmethod
    fn kernel_sub_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_sub_grad_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_sub_grad_a](c_rest)

    @parameter
    @staticmethod
    fn kernel_sub_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_sub_grad_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i) - c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_sub_grad_b](c_rest)


struct Div(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_divectorized_fw, base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_divectorized_bw_a, base_case_strides](
                c, a, b
            )
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_divectorized_bw_b, base_case_strides](
                c, a, b
            )

    @parameter
    @staticmethod
    fn kernel_divectorized_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_div[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                a.load_data[nelts](offset_a + i) / b.load_data[nelts](offset_b + i),
            )

        vectorize[nelts, vectorized_div](c_rest)

    @parameter
    @staticmethod
    fn kernel_divectorized_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_divectorized_grad_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i)
                + c.load_grad[nelts](offset_c + i) / b.load_data[nelts](offset_b + i),
            )

        vectorize[nelts, vectorized_divectorized_grad_a](c_rest)

    @parameter
    @staticmethod
    fn kernel_divectorized_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_divectorized_grad_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i)
                - a.load_data[nelts](offset_a + i)
                * c.load_grad[nelts](offset_c + i)
                / (b.load_data[nelts](offset_b + i)) ** 2,
            )

        vectorize[nelts, vectorized_divectorized_grad_b](c_rest)


struct MMul(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_mmul_fw, base_case_depth](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mmul_bw_a, base_case_depth](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mmul_bw_b, base_case_depth](c, a, b)

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


struct Pow(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_pow_fw, base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_pow_bw_a, base_case_strides](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_pow_bw_b, base_case_strides](c, a, b)

    @parameter
    @staticmethod
    fn kernel_pow_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_pow[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                a.load_data[nelts](offset_a + i) ** b.load_data[nelts](offset_b + i),
            )

        vectorize[nelts, vectorized_pow](c_rest)

    @parameter
    @staticmethod
    fn kernel_pow_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_pow_bw_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i)
                + b.load_data[nelts](offset_b + i)
                * (
                    a.load_data[nelts](offset_a + i)
                    ** (b.load_data[nelts](offset_b + i) - 1.0)
                )
                * c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_pow_bw_a](c_rest)

    @parameter
    @staticmethod
    fn kernel_pow_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_pow_bw_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i)
                + c.load_data[nelts](offset_c + i)
                * log(a.load_data[nelts](offset_a + i))
                * c.load_grad[nelts](offset_c + i),
            )

        vectorize[nelts, vectorized_pow_bw_b](c_rest)


struct Avg(BinaryArithmetic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_avg_fw, base_case_strides](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_avg_bw_a, base_case_strides](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_avg_bw_b, base_case_strides](c, a, b)

    @parameter
    @staticmethod
    fn kernel_avg_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ):
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)

        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_avg[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                (a.load_data[nelts](offset_a + i) + b.load_data[nelts](offset_b + i))
                / 2.0,
            )

        vectorize[nelts, vectorized_avg](c_rest)

    @parameter
    @staticmethod
    fn kernel_avg_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ):
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)

        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_avg_bw_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i, a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](
                    offset_c + i
                )
            )

        vectorize[nelts, vectorized_avg_bw_a](c_rest)

    @parameter
    @staticmethod
    fn kernel_avg_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ):
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)

        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        fn vectorized_avg_bw_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i, b.load_grad[nelts](offset_b + i) + c.load_grad[nelts](
                    offset_c + i
                )
            )

        vectorize[nelts, vectorized_avg_bw_b](c_rest)