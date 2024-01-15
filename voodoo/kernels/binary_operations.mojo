from random import random_float64
from algorithm import *
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from sys.intrinsics import prefetch, PrefetchOptions

alias prefetch_options = PrefetchOptions().for_read().high_locality().to_data_cache()


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

        let a_dims = a.num_dims_ptr.load()
        let b_dims = b.num_dims_ptr.load()

        let M = shape_a.load(a_dims - 2)
        let K = shape_b.load(b_dims - 2)
        let N = shape_c.load(b_dims - 1)

        let offset_a = a_index * M * shape_a.load(a_dims - 1)
        let offset_b = b_index * K * shape_b.load(b_dims - 1)
        let offset_c = c_index * N * shape_c.load(c.num_dims_ptr.load() - 1)

        for m in range(M):
            let _a_off = offset_a + m * K
            let _c_off = offset_c + m * N

            prefetch[prefetch_options](a.data.load() + _a_off)

            for k in range(K):
                let a_off = _a_off + k
                let a_scalar = a.load_data(a_off)
                let _b_off = offset_b + k * N

                prefetch[prefetch_options](b.data.load() + _b_off)
                prefetch[prefetch_options](c.data.load() + _c_off)

                @parameter
                fn dot_fw[nelts: Int](n: Int):
                    let b_off = _b_off + n
                    let c_off = _c_off + n

                    c.store_data[nelts](
                        c_off,
                        b.load_data[nelts](b_off).fma(
                            a_scalar,
                            c.load_data[nelts](c_off),
                        ),
                    )

                vectorize[nelts, dot_fw](N)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let shape_a = a.shape_ptr.load()
        let shape_b = b.shape_ptr.load()
        let shape_c = c.shape_ptr.load()

        let a_dims = a.num_dims_ptr.load()
        let b_dims = b.num_dims_ptr.load()

        let M = shape_a.load(a_dims - 2)
        let K = shape_b.load(b_dims - 2)
        let N = shape_c.load(b_dims - 1)

        let offset_a = a_index * M * shape_a.load(a_dims - 1)
        let offset_b = b_index * K * shape_b.load(b_dims - 1)
        let offset_c = c_index * N * shape_c.load(c.num_dims_ptr.load() - 1)

        for m in range(M):
            let _a_off = offset_a + m * K
            let _c_off = offset_c + m * N
            for n in range(N):
                let c_offset = _c_off + n
                let c_grad = c.load_grad(c_offset)
                let _b_off = offset_b + n

                prefetch[prefetch_options](a.data.load() + _a_off)
                prefetch[prefetch_options](b.data.load() + _b_off)

                @parameter
                fn dot_bw[nelts: Int](k: Int):
                    let a_off = _a_off + k

                    a.store_grad[nelts](
                        a_off,
                        b.load_data[nelts](_b_off + k * N).fma(
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

        let a_dims = a.num_dims_ptr.load()
        let b_dims = b.num_dims_ptr.load()

        let M = shape_a.load(a_dims - 2)
        let K = shape_b.load(b_dims - 2)
        let N = shape_c.load(b_dims - 1)

        let offset_a = a_index * M * shape_a.load(a_dims - 1)
        let offset_b = b_index * K * shape_b.load(b_dims - 1)
        let offset_c = c_index * N * shape_c.load(c.num_dims_ptr.load() - 1)

        for k in range(K):
            let _a_off = offset_a + k
            let _b_off = offset_b + k * N

            prefetch[prefetch_options](a.data.load() + _a_off)

            for m in range(M):
                let a_data = a.load_data(_a_off + m * K)
                let _c_off = offset_c + m * N

                prefetch[prefetch_options](c.data.load() + _c_off)

                @parameter
                fn dot_bw[nelts: Int](n: Int):
                    let b_off = _b_off + n

                    b.store_grad[nelts](
                        b_off,
                        c.load_grad[nelts](_c_off + n).fma(
                            a_data,
                            b.load_grad[nelts](b_off),
                        ),
                    )

                vectorize[nelts, dot_bw](N)
