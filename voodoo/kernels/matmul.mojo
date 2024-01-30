from algorithm import vectorize
from math import max, min
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import PREFETCH_READ, PREFETCH_WRITE, F32_MAX, NELTS

# TODO: Add cleanup for tiling


struct MMul:
    @staticmethod
    @always_inline("nodebug")
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

    @staticmethod
    fn kernel_mmul_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let a_num_dims = a.num_dims_ptr.load()
        let b_num_dims = b.num_dims_ptr.load()

        let M = a.shape.load(a_num_dims - 2)
        let K = b.shape.load(b_num_dims - 2)
        let N = c.shape.load(c.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * a.shape.load(a_num_dims - 1)
        let offset_b = b_index * K * b.shape.load(b_num_dims - 1)
        let offset_c = c_index * N * N

        let a_data = a.data_ptr.load(0)
        let b_data = b.data_ptr.load(0)
        let c_data = c.data_ptr.load(0)

        DTypePointer.prefetch[PREFETCH_READ](a_data)
        DTypePointer.prefetch[PREFETCH_READ](b_data)
        DTypePointer.prefetch[PREFETCH_READ](c_data)
        DTypePointer.prefetch[PREFETCH_WRITE](c_data)

        for m in range(0, M, 4):
            let start_offset_c = offset_c + m * N
            let start_offset_a = offset_a + m * K
            for kb in range(0, K, NELTS):
                for k in range(kb, min(kb + NELTS, K)):
                    let b_off = offset_b + k * N

                    @parameter
                    @always_inline("nodebug")
                    fn dot_fw[NELTS: Int](n: Int):
                        let b_data_n = b_data.simd_load[NELTS](b_off + n)

                        @parameter
                        @always_inline("nodebug")
                        fn dot_store(c_off_n: Int, a_off: Int):
                            c_data.simd_store[NELTS](
                                c_off_n,
                                b_data_n.fma(
                                    a_data.load(a_off + k),
                                    c_data.simd_load[NELTS](c_off_n),
                                ),
                            )

                        dot_store(start_offset_c + n, start_offset_a)
                        dot_store(start_offset_c + N + n, start_offset_a + K)
                        dot_store(start_offset_c + 2 * N + n, start_offset_a + 2 * K)
                        dot_store(start_offset_c + 3 * N + n, start_offset_a + 3 * K)

                    vectorize[NELTS, dot_fw](N)

    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let a_num_dims = a.num_dims_ptr.load()
        let b_num_dims = b.num_dims_ptr.load()

        let M = a.shape.load(a_num_dims - 2)
        let K = b.shape.load(b_num_dims - 2)
        let N = c.shape.load(c.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * a.shape.load(a_num_dims - 1)
        let offset_b = b_index * K * b.shape.load(b_num_dims - 1)
        let offset_c = c_index * N * N

        let a_grad = a.data_ptr.load(1)
        let b_data = b.data_ptr.load(0)
        let c_grad = c.data_ptr.load(1)

        DTypePointer.prefetch[PREFETCH_READ](a_grad)
        DTypePointer.prefetch[PREFETCH_WRITE](a_grad)
        DTypePointer.prefetch[PREFETCH_READ](b_data)
        DTypePointer.prefetch[PREFETCH_READ](c_grad)

        for m in range(0, M, 2):
            let _offset_c = offset_c + m * N
            let _offset_c_1 = offset_c + (m + 1) * N
            let start_offset_a = offset_a + m * K
            for nb in range(0, N, NELTS):
                for n in range(nb, min(nb + NELTS, N), 2):
                    let c_grad_0 = c_grad.load(_offset_c + n)
                    let c_grad_1 = c_grad.load(_offset_c_1 + n)

                    @parameter
                    @always_inline("nodebug")
                    fn dot_bw[NELTS: Int](k: Int):
                        @parameter
                        @always_inline("nodebug")
                        fn dot_store(a_off: Int, b_off: Int, scalar: Float32):
                            a_grad.simd_store[NELTS](
                                a_off,
                                b_data.simd_load[NELTS](b_off).fma(
                                    scalar,
                                    a_grad.simd_load[NELTS](a_off),
                                ),
                            )

                        let start_offset_b = offset_b + k * N

                        dot_store(start_offset_a + k, start_offset_b + n, c_grad_0)
                        dot_store(start_offset_a + K + k, start_offset_b + n, c_grad_1)

                    vectorize[NELTS, dot_bw](K)

    @staticmethod
    fn kernel_mmul_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let a_num_dims = a.num_dims_ptr.load()
        let b_num_dims = b.num_dims_ptr.load()

        let M = a.shape.load(a_num_dims - 2)
        let K = b.shape.load(b_num_dims - 2)
        let N = c.shape.load(c.num_dims_ptr.load() - 1)

        let offset_a = a_index * M * a.shape.load(a_num_dims - 1)
        let offset_b = b_index * K * b.shape.load(b_num_dims - 1)
        let offset_c = c_index * N * N

        let a_data = a.data_ptr.load(0)
        let b_grad = b.data_ptr.load(1)
        let c_grad = c.data_ptr.load(1)

        DTypePointer.prefetch[PREFETCH_READ](a_data)
        DTypePointer.prefetch[PREFETCH_READ](b_grad)
        DTypePointer.prefetch[PREFETCH_WRITE](b_grad)
        DTypePointer.prefetch[PREFETCH_READ](c_grad)

        if K == 1:
            let _a_off = offset_a
            let _b_off = offset_b

            for m in range(M):
                let a_data = a_data.load(_a_off + m)
                let _c_off = offset_c + m * N

                @parameter
                @always_inline("nodebug")
                fn dot_bw_single[NELTS: Int](n: Int):
                    let b_off = _b_off + n

                    b.data_ptr.load(1).simd_store[NELTS](
                        b_off,
                        c_grad.simd_load[NELTS](_c_off + n).fma(
                            a_data,
                            b_grad.simd_load[NELTS](b_off),
                        ),
                    )

                vectorize[NELTS, dot_bw_single](N)
        else:
            for k in range(0, K, 2):
                let _a_off_1 = offset_a + k
                let _a_off_2 = offset_a + k + 1
                let _b_off_1 = offset_b + k * N
                let _b_off_2 = offset_b + (k + 1) * N

                for m in range(M):
                    let a_data_1 = a_data.load(_a_off_1 + m * K)
                    let a_data_2 = a_data.load(_a_off_2 + m * K)
                    let _c_off = offset_c + m * N

                    @parameter
                    @always_inline("nodebug")
                    fn dot_bw_inner[NELTS: Int](n: Int):
                        let b_off_1 = _b_off_1 + n
                        let b_off_2 = _b_off_2 + n

                        b.data_ptr.load(1).simd_store[NELTS](
                            b_off_1,
                            c_grad.simd_load[NELTS](_c_off + n).fma(
                                a_data_1,
                                b_grad.simd_load[NELTS](b_off_1),
                            ),
                        )

                        b.data_ptr.load(1).simd_store[NELTS](
                            b_off_2,
                            c_grad.simd_load[NELTS](_c_off + n).fma(
                                a_data_2,
                                b_grad.simd_load[NELTS](b_off_2),
                            ),
                        )

                    vectorize[NELTS, dot_bw_inner](N)
