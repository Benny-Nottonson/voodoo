from algorithm import vectorize
from math import max, min
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import PREFETCH_READ, PREFETCH_WRITE, F32_MAX, NELTS


struct MMul:
    @parameter
    @staticmethod
    @always_inline
    fn base_case_depth(depth: Int, a: Node, b: Node) -> Bool:
        return depth == max(a.num_dims, b.num_dims) - 2

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
        let shape_info = load_shapes_and_dims(a, b, c, a_index, b_index, c_index)

        let M = shape_info[0]
        let K = shape_info[1]
        let N = shape_info[2]

        let offset_a = shape_info[3]
        let offset_b = shape_info[4]
        let offset_c = shape_info[5]

        let a_data = a.data.load(0)
        let b_data = b.data.load(0)
        let c_data = c.data.load(0)

        DTypePointer.prefetch[PREFETCH_READ](a_data)
        DTypePointer.prefetch[PREFETCH_READ](b_data)
        DTypePointer.prefetch[PREFETCH_READ](c_data)
        DTypePointer.prefetch[PREFETCH_WRITE](c_data)

        for m in range(0, M, 4):
            for kb in range(0, K, NELTS):
                for k in range(kb, min(kb + NELTS, K)):
                    let b_off = offset_b + k * N

                    @parameter
                    @always_inline
                    fn dot_fw[NELTS: Int](n: Int):
                        let b_data_n = b_data.simd_load[NELTS](b_off + n)

                        @parameter
                        @always_inline
                        fn dot_store(c_off_n: Int, a_off: Int):
                            c_data.simd_store[NELTS](
                                c_off_n,
                                b_data_n.fma(
                                    a_data.load(a_off + k),
                                    c_data.simd_load[NELTS](c_off_n),
                                ),
                            )

                        let start_offset_c = offset_c + m * N
                        let start_offset_a = offset_a + m * K

                        dot_store(start_offset_c + n, start_offset_a)
                        dot_store(start_offset_c + N + n, start_offset_a + K)
                        dot_store(start_offset_c + 2 * N + n, start_offset_a + 2 * K)
                        dot_store(start_offset_c + 3 * N + n, start_offset_a + 3 * K)

                    vectorize[NELTS, dot_fw](N)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let shape_info = load_shapes_and_dims(a, b, c, a_index, b_index, c_index)

        let M = shape_info[0]
        let K = shape_info[1]
        let N = shape_info[2]

        let offset_a = shape_info[3]
        let offset_b = shape_info[4]
        let offset_c = shape_info[5]

        let a_grad = a.data.load(1)
        let b_data = b.data.load(0)
        let c_grad = c.data.load(1)

        DTypePointer.prefetch[PREFETCH_READ](a_grad)
        DTypePointer.prefetch[PREFETCH_WRITE](a_grad)
        DTypePointer.prefetch[PREFETCH_READ](b_data)
        DTypePointer.prefetch[PREFETCH_READ](c_grad)

        for m in range(0, M, 2):
            for nb in range(0, N, NELTS):
                for n in range(nb, min(nb + NELTS, N), 2):
                    let c_grad_0 = c_grad.load(offset_c + m * N + n)
                    let c_grad_1 = c_grad.load(offset_c + (m + 1) * N + n)

                    @parameter
                    @always_inline
                    fn dot_bw[NELTS: Int](k: Int):
                        @parameter
                        @always_inline
                        fn dot_store(a_off: Int, b_off: Int, scalar: Float32):
                            a_grad.simd_store[NELTS](
                                a_off,
                                b_data.simd_load[NELTS](b_off).fma(
                                    scalar,
                                    a_grad.simd_load[NELTS](a_off),
                                ),
                            )

                        let start_offset_a = offset_a + m * K
                        let start_offset_b = offset_b + k * N

                        dot_store(start_offset_a + k, start_offset_b + n, c_grad_0)
                        dot_store(start_offset_a + K + k, start_offset_b + n, c_grad_1)

                    vectorize[NELTS, dot_bw](K)

    @parameter
    @staticmethod
    fn kernel_mmul_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let shape_info = load_shapes_and_dims(a, b, c, a_index, b_index, c_index)

        let M = shape_info[0]
        let K = shape_info[1]
        let N = shape_info[2]

        let offset_a = shape_info[3]
        let offset_b = shape_info[4]
        let offset_c = shape_info[5]

        let a_data = a.data.load(0)
        let b_grad = b.data.load(1)
        let c_grad = c.data.load(1)

        DTypePointer.prefetch[PREFETCH_READ](a_data)
        DTypePointer.prefetch[PREFETCH_READ](b_grad)
        DTypePointer.prefetch[PREFETCH_WRITE](b_grad)
        DTypePointer.prefetch[PREFETCH_READ](c_grad)

        for k in range(K):
            let _a_off = offset_a + k
            let _b_off = offset_b + k * N

            for m in range(M):
                let a_data = a_data.load(_a_off + m * K)
                let _c_off = offset_c + m * N

                @parameter
                @always_inline
                fn dot_bw[NELTS: Int](n: Int):
                    let b_off = _b_off + n

                    b.store_grad[NELTS](
                        b_off,
                        c_grad.simd_load[NELTS](_c_off + n).fma(
                            a_data,
                            b_grad.simd_load[NELTS](b_off),
                        ),
                    )

                vectorize[NELTS, dot_bw](N)


@parameter
@always_inline
fn load_shapes_and_dims(
    a: Node, b: Node, c: Node, a_index: Int, b_index: Int, c_index: Int
) -> StaticIntTuple[6]:
    let a_shape = a.shape
    let b_shape = b.shape
    let c_shape = c.shape

    let a_dims = a.num_dims
    let b_dims = b.num_dims
    let c_dims = c.num_dims

    let M = a_shape.load(a_dims - 2)
    let K = b_shape.load(b_dims - 2)
    let N = c_shape.load(c_dims - 1)

    let offset_a = a_index * M * a_shape.load(a_dims - 1)
    let offset_b = b_index * K * b_shape.load(b_dims - 1)
    let offset_c = c_index * N * N

    return StaticIntTuple[6](M, K, N, offset_a, offset_b, offset_c)
