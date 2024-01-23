from algorithm import vectorize_unroll
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import prefetch_read, prefetch_write, f32_max, nelts


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
        let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
        let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
        let N = c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * K * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * N * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let a_data = a.data.load(0)
        let b_data = b.data.load(0)
        let c_data = c.data.load(0)

        DTypePointer.prefetch[prefetch_read](a_data)
        DTypePointer.prefetch[prefetch_read](b_data)
        DTypePointer.prefetch[prefetch_read](c_data)
        DTypePointer.prefetch[prefetch_write](c_data)

        for m in range(0, M, 4):
            for k in range(K):
                let b_off = offset_b + k * N

                @parameter
                @always_inline
                fn dot_fw[nelts: Int](n: Int):
                    let b_data_n = b_data.simd_load[nelts](b_off + n)

                    @parameter
                    @always_inline
                    fn dot_store(c_off_n: Int, a_off: Int):
                        c_data.simd_store[nelts](
                            c_off_n,
                            b_data_n.fma(
                                a_data.load(a_off + k),
                                c_data.simd_load[nelts](c_off_n),
                            ),
                        )

                    let start_offset_c = offset_c + m * N
                    let start_offset_a = offset_a + m * K

                    dot_store(start_offset_c + n, start_offset_a)
                    dot_store(start_offset_c + N + n, start_offset_a + K)
                    dot_store(start_offset_c + 2 * N + n, start_offset_a + 2 * K)
                    dot_store(start_offset_c + 3 * N + n, start_offset_a + 3 * K)
                vectorize_unroll[nelts, 1, dot_fw](N)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
        let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
        let N = c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
        let offset_b = b_index * K * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
        let offset_c = c_index * N * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

        let a_grad = a.data.load(1)
        let b_data = b.data.load(0)
        let c_grad = c.data.load(1)

        DTypePointer.prefetch[prefetch_read](a_grad)
        DTypePointer.prefetch[prefetch_write](a_grad)
        DTypePointer.prefetch[prefetch_read](b_data)
        DTypePointer.prefetch[prefetch_read](c_grad)

        for m in range(0, M, 2):
            for n in range(0, N, 2):
                let c_grad_0 = c_grad.load(offset_c + m * N + n)
                let c_grad_1 = c_grad.load(offset_c + (m + 1) * N + n)

                @parameter
                @always_inline
                fn dot_bw[nelts: Int](k: Int):
                    @parameter
                    @always_inline
                    fn dot_store(a_off: Int, b_off: Int, scalar: Float32):
                        a_grad.simd_store[nelts](
                            a_off,
                            b_data.simd_load[nelts](b_off).fma(
                                scalar,
                                a_grad.simd_load[nelts](a_off),
                            ),
                        )

                    let start_offset_a = offset_a + m * K
                    let start_offset_b = offset_b + k * N

                    dot_store(start_offset_a + k, start_offset_b + n, c_grad_0)
                    dot_store(start_offset_a + K + k, start_offset_b + n, c_grad_1)

                vectorize_unroll[nelts, 1, dot_bw](K)

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

        let a_data = a.data.load(0)
        let b_grad = b.data.load(1)
        let c_grad = c.data.load(1)

        DTypePointer.prefetch[prefetch_read](a_data)
        DTypePointer.prefetch[prefetch_read](b_grad)
        DTypePointer.prefetch[prefetch_write](b_grad)
        DTypePointer.prefetch[prefetch_read](c_grad)

        for k in range(K):
            let _a_off = offset_a + k
            let _b_off = offset_b + k * N

            for m in range(M):
                let a_data = a_data.load(_a_off + m * K)
                let _c_off = offset_c + m * N

                @parameter
                @always_inline
                fn dot_bw[nelts: Int](n: Int):
                    let b_off = _b_off + n

                    b_grad.simd_store[nelts](
                        b_off,
                        c_grad.simd_load[nelts](_c_off + n).fma(
                            a_data,
                            b_grad.simd_load[nelts](b_off),
                        ),
                    )

                vectorize_unroll[nelts, 1, dot_bw](N)
