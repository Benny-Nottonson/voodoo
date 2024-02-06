from algorithm import vectorize
from math import max, min
from voodoo import Node
from voodoo.utils import recursive_broadcast
from ..constants import PREFETCH_READ, PREFETCH_WRITE, F32_MAX, NELTS


alias bw_b_tile_size = 2


@register_passable("trivial")
struct MMul:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_mmul_fw, False](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.get_is_single():
            recursive_broadcast[Self.kernel_mmul_bw_a, False](c, a, b)
        if not b.get_is_single():
            recursive_broadcast[Self.kernel_mmul_bw_b, False](c, a, b)

    @staticmethod
    fn kernel_mmul_fw(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let a_num_dims = a.get_num_dims()
        let b_num_dims = b.get_num_dims()

        let M = a.get_shape()[a_num_dims - 2]
        let K = b.get_shape()[b_num_dims - 2]
        let N = c.get_shape()[c.get_num_dims() - 1]

        let offset_a = a_index * M * a.get_shape()[a_num_dims - 1]
        let offset_b = b_index * K * b.get_shape()[b_num_dims - 1]
        let offset_c = c_index * N * N

        let a_data = a.get_data()
        let b_data = b.get_data()
        let c_data = c.get_data()

        DTypePointer.prefetch[PREFETCH_READ](a_data)
        DTypePointer.prefetch[PREFETCH_READ](b_data)
        DTypePointer.prefetch[PREFETCH_READ](c_data)
        DTypePointer.prefetch[PREFETCH_WRITE](c_data)

        for m in range(0, M):
            let start_offset_a = offset_a + m * K
            let start_offset_c = offset_c + m * N
            for kb in range(0, K, NELTS):
                for k in range(kb, min(kb + NELTS, K)):
                    let start_offset_b = offset_b + k * N
                    let a_scalar = a_data.load(start_offset_a + k)

                    @parameter
                    fn dot_fw[NELTS: Int](n: Int):
                        c_data.simd_store[NELTS](
                            start_offset_c + n,
                            b_data.simd_load[NELTS](start_offset_b + n).fma(
                                a_scalar,
                                c_data.simd_load[NELTS](start_offset_c + n),
                            ),
                        )

                    vectorize[NELTS, dot_fw](N)

    @staticmethod
    fn kernel_mmul_bw_a(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let a_num_dims = a.get_num_dims()
        let b_num_dims = b.get_num_dims()

        let M = a.get_shape()[a_num_dims - 2]
        let K = b.get_shape()[b_num_dims - 2]
        let N = c.get_shape()[c.get_num_dims() - 1]

        let offset_a = a_index * M * a.get_shape()[a_num_dims - 1]
        let offset_b = b_index * K * b.get_shape()[b_num_dims - 1]
        let offset_c = c_index * N * N

        let a_grad = a.get_grad()
        let b_data = b.get_data()
        let c_grad = c.get_grad()

        DTypePointer.prefetch[PREFETCH_READ](a_grad)
        DTypePointer.prefetch[PREFETCH_WRITE](a_grad)
        DTypePointer.prefetch[PREFETCH_READ](b_data)
        DTypePointer.prefetch[PREFETCH_READ](c_grad)

        for m in range(0, M):
            let start_offset_a = offset_a + m * K
            let start_offset_c = offset_c + m * N
            for nb in range(0, N, NELTS):
                for n in range(nb, min(nb + NELTS, N)):
                    let start_offset_b = offset_b + n * N
                    let c_grad_scalar = c_grad.load(start_offset_c + n)

                    @parameter
                    fn dot_bw[NELTS: Int](n: Int):
                        a_grad.simd_store[NELTS](
                            start_offset_a + n,
                            b_data.simd_load[NELTS](start_offset_b + n).fma(
                                c_grad_scalar,
                                a_grad.simd_load[NELTS](start_offset_a + n),
                            ),
                        )

                    vectorize[NELTS, dot_bw](K)

    @staticmethod
    fn kernel_mmul_bw_b(
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let a_num_dims = a.get_num_dims()
        let b_num_dims = b.get_num_dims()

        let M = a.get_shape()[a_num_dims - 2]
        let K = b.get_shape()[b_num_dims - 2]
        let N = c.get_shape()[c.get_num_dims() - 1]

        let offset_a = a_index * M * a.get_shape()[a_num_dims - 1]
        let offset_b = b_index * K * b.get_shape()[b_num_dims - 1]
        let offset_c = c_index * N * N

        let a_data = a.get_data()
        let b_grad = b.get_grad()
        let c_grad = c.get_grad()

        DTypePointer.prefetch[PREFETCH_READ](a_data)
        DTypePointer.prefetch[PREFETCH_READ](b_grad)
        DTypePointer.prefetch[PREFETCH_WRITE](b_grad)
        DTypePointer.prefetch[PREFETCH_READ](c_grad)

        for k in range(0, K):
            let start_offset_a = offset_a + k
            let start_offset_b = offset_b + k * N

            for m in range(M):
                let start_offset_c = offset_c + m * N
                let a_scalar = a_data.load(start_offset_a + m * K)

                @parameter
                fn dot_bw_b[NELTS: Int](n: Int):
                    b_grad.simd_store[NELTS](
                        start_offset_b + n,
                        c_grad.simd_load[NELTS](start_offset_c + n).fma(
                            a_scalar, b_grad.simd_load[NELTS](start_offset_b + n)
                        ),
                    )

                vectorize[NELTS, dot_bw_b](N)
