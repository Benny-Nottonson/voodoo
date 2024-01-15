from random import random_float64
from algorithm import *
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)


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
        fn calc_row_fw(m: Int):
            for k in range(K):
                let a_off = offset_a + m * K + k
                let a_scalar = a.load_data(a_off)

                @parameter
                fn dot_fw[nelts: Int](n: Int):
                    let b_off = offset_b + k * N + n
                    let c_off = offset_c + m * N + n
                    c.store_data[nelts](
                        c_off,
                        b.load_data[nelts](b_off).fma(
                            a_scalar,
                            c.load_data[nelts](c_off),
                        ),
                    )

                vectorize[nelts, dot_fw](N)
                
        parallelize[calc_row_fw](M, workers if workers > 0 else M // 2)

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

        for m in range(M):
            for n in range(N):
                let c_offset = offset_c + m * N + n
                let c_grad = c.load_grad(c_offset)

                @parameter
                fn dot_bw[nelts: Int](k: Int):
                    let a_off = offset_a + m * K + k
                    let b_off = offset_b + k * N + n
                    a.store_grad[nelts](
                        a_off,
                        b.load_data[nelts](b_off).fma(
                            c_grad,
                            a.load_grad[nelts](a_off),
                        ),
                    )

                vectorize[nelts, dot_bw](K)

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

        for k in range(K):
            for m in range(M):
                let a_offset = offset_a + m * K + k
                let a_data = a.load_data(a_offset)

                @parameter
                fn dot_bw[nelts: Int](n: Int):
                    let b_off = offset_b + k * N + n
                    let c_off = offset_c + m * N + n
                    b.store_grad[nelts](
                        b_off,
                        c.load_grad[nelts](c_off).fma(
                            a_data,
                            b.load_grad[nelts](b_off),
                        ),
                    )

                vectorize[nelts, dot_bw](N)
