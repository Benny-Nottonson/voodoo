from random import random_float64
from algorithm import vectorize, parallelize
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import DType_F32, nelts, workers


struct MMul:
    @parameter
    @staticmethod
    @always_inline
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
                fn dot_fw[nelts: Int](n: Int):
                    c.store_data[nelts](
                        offset_c + m * N + n,
                        c.load_data[nelts](offset_c + m * N + n)
                        + a.load_data(offset_a + m * K + k)
                        * b.load_data[nelts](offset_b + k * N + n),
                    )

                vectorize[nelts, dot_fw](N)

        parallelize[calc_row_fw](M, workers if workers > 0 else M)

    # IMPORTANT: These two functions take BY FAR the most time in the entire program.
    # How can they be optimized?
    @parameter
    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let shape_a = a.shape_ptr.load()
        let shape_b = b.shape_ptr.load()
        let shape_c = c.shape_ptr.load()

        let M = shape_a.load(a.num_dims_ptr.load() - 2)
        let K = shape_b.load(b.num_dims_ptr.load() - 2)
        let N = shape_c.load(b.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * shape_a.load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * K * shape_b.load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * N * shape_c.load(c.num_dims_ptr.load() - 1)

        @parameter
        fn calc_row_1(m: Int):
            for n in range(N):

                @parameter
                fn dot_bw_a[nelts: Int](k: Int):
                    let val = b.load_data[nelts](offset_b + k * N + n).fma(
                        c.load_grad[nelts](offset_c + m * N + n),
                        a.load_grad[nelts](offset_a + m * K + k),
                    )
                    a.store_grad[nelts](offset_a + m * K + k, val)

                vectorize[1, dot_bw_a](K)

        parallelize[calc_row_1](M, workers if workers > 0 else M)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let shape_a = a.shape_ptr.load()
        let shape_b = b.shape_ptr.load()
        let shape_c = c.shape_ptr.load()

        let M = shape_a.load(a.num_dims_ptr.load() - 2)
        let K = shape_b.load(b.num_dims_ptr.load() - 2)
        let N = shape_c.load(b.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * shape_a.load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * K * shape_b.load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * N * shape_c.load(c.num_dims_ptr.load() - 1)

        @parameter
        fn calc_row_2(k: Int):
            for m in range(M):

                @parameter
                fn dot_bw_b[nelts: Int](n: Int):
                    let val = a.load_data[nelts](offset_a + m * K + k).fma(
                        c.load_grad[nelts](offset_c + m * N + n),
                        b.load_grad[nelts](offset_b + k * N + n),
                    )
                    b.store_grad[nelts](offset_b + k * N + n, val)

                vectorize[1, dot_bw_b](N)

        parallelize[calc_row_2](K, workers if workers > 0 else K)
