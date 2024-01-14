from random import random_float64
from algorithm import vectorize, parallelize
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import DType_F32, nelts, workers


struct Conv2D:
    @staticmethod
    @always_inline
    fn fw(c: Node, a: Node, b: Node):
        let padding = c.other_params_ptr.load().load(0)
        let stride = c.other_params_ptr.load().load(1)

        @parameter
        fn batch_loop(i: Int):
            for j in range(b.shape_ptr.load().load(0)):
                for x in range(c.shape_ptr.load().load(2)):
                    for y in range(c.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        for k in range(a.shape_ptr.load().load(1)):
                            for dx in range(b.shape_ptr.load().load(2)):

                                @parameter
                                fn inner_loop[nelts: Int](dy: Int):
                                    let ix = x * stride - padding + dx
                                    let iy = y * stride - padding + dy
                                    if not (
                                        ix < 0
                                        or iy < 0
                                        or ix >= a.shape_ptr.load().load(2)
                                        or iy >= a.shape_ptr.load().load(3)
                                    ):
                                        let a_index = Self.index(
                                            i,
                                            k,
                                            ix,
                                            iy,
                                            a.shape_ptr.load().load(1),
                                            a.shape_ptr.load().load(2),
                                            a.shape_ptr.load().load(3),
                                        )
                                        let b_index = Self.index(
                                            j,
                                            k,
                                            dx,
                                            dy,
                                            a.shape_ptr.load().load(1),
                                            b.shape_ptr.load().load(2),
                                            b.shape_ptr.load().load(3),
                                        )
                                        patch_sum += (
                                            a.load_data[nelts](a_index)
                                            * b.load_data[nelts](b_index)
                                        ).reduce_add()

                                vectorize[nelts, inner_loop](b.shape_ptr.load().load(3))
                        let c_index = Self.index(
                            i,
                            j,
                            x,
                            y,
                            b.shape_ptr.load().load(0),
                            c.shape_ptr.load().load(2),
                            c.shape_ptr.load().load(3),
                        )
                        c.store_data(c_index, patch_sum)

        parallelize[batch_loop](
            a.shape_ptr.load().load(0),
            workers if workers > 0 else a.shape_ptr.load().load(0),
        )

    @staticmethod
    @always_inline
    fn bw(c: Node, a: Node, b: Node):
        let padding = c.other_params_ptr.load().load(0)
        let stride = c.other_params_ptr.load().load(1)

        for i in range(a.shape_ptr.load().load(1)):
            for j in range(b.shape_ptr.load().load(0)):
                for x in range(b.shape_ptr.load().load(2)):
                    for y in range(b.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        for b in range(a.shape_ptr.load().load(0)):
                            for dx in range(c.shape_ptr.load().load(2)):
                                for dy in range(c.shape_ptr.load().load(3)):
                                    let ix = x * stride - padding + dx
                                    let iy = y * stride - padding + dy
                                    if not (
                                        ix < 0
                                        or iy < 0
                                        or ix >= a.shape_ptr.load().load(2)
                                        or iy >= a.shape_ptr.load().load(3)
                                    ):
                                        let a_index = Self.index(
                                            b,
                                            i,
                                            ix,
                                            iy,
                                            a.shape_ptr.load().load(1),
                                            a.shape_ptr.load().load(2),
                                            a.shape_ptr.load().load(3),
                                        )
                                        let c_grad_index = Self.index(
                                            b,
                                            j,
                                            dx,
                                            dy,
                                            c.shape_ptr.load().load(1),
                                            c.shape_ptr.load().load(2),
                                            c.shape_ptr.load().load(3),
                                        )
                                        patch_sum += (
                                            a.load_data(a_index)
                                            * c.load_grad(c_grad_index)
                                        ).reduce_add()
                        let b_grad_index = Self.index(
                            i,
                            j,
                            x,
                            y,
                            b.shape_ptr.load().load(0),
                            b.shape_ptr.load().load(2),
                            b.shape_ptr.load().load(3),
                        )
                        b.store_grad(b_grad_index, patch_sum)

        @parameter
        fn batch_loop(p: Int):
            for j in range(a.shape_ptr.load().load(1)):
                for i in range(b.shape_ptr.load().load(0)):
                    for x in range(a.shape_ptr.load().load(2)):
                        for y in range(a.shape_ptr.load().load(3)):
                            var patch_sum: Float32 = 0.0
                            for dx in range(b.shape_ptr.load().load(2)):

                                @parameter
                                fn dy_loop[nelts: Int](dy: Int):
                                    let ix = x * stride - dx + padding
                                    let iy = y * stride - dy + padding
                                    if not (
                                        ix < 0
                                        or iy < 0
                                        or ix >= c.shape_ptr.load().load(2)
                                        or iy >= c.shape_ptr.load().load(3)
                                    ):
                                        let c_grad_index = Self.index(
                                            p,
                                            i,
                                            ix,
                                            iy,
                                            c.shape_ptr.load().load(1),
                                            c.shape_ptr.load().load(2),
                                            c.shape_ptr.load().load(3),
                                        )
                                        let b_index = Self.index(
                                            i,
                                            j,
                                            b.shape_ptr.load().load(2) - dx - 1,
                                            b.shape_ptr.load().load(3) - dy - 1,
                                            b.shape_ptr.load().load(1),
                                            b.shape_ptr.load().load(2),
                                            b.shape_ptr.load().load(3),
                                        )
                                        patch_sum += (
                                            c.load_grad[nelts](c_grad_index)
                                            * c.load_data[nelts](b_index)
                                        ).reduce_add()

                                vectorize[nelts, dy_loop](b.shape_ptr.load().load(3))
                            let a_grad_index = Self.index(
                                p,
                                j,
                                x,
                                y,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            a.store_grad(
                                a_grad_index, a.load_grad(a_grad_index) + patch_sum
                            )

        parallelize[batch_loop](
            a.shape_ptr.load().load(0),
            workers if workers > 0 else a.shape_ptr.load().load(0),
        )

    @always_inline
    @staticmethod
    fn index(
        n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
    ) -> Int:
        return (
            n * (num_channels * height * width) + c * (height * width) + h * width + w
        )


struct MMul:
    @parameter
    @staticmethod
    @always_inline
    fn base_case_depth(depth: Int, a: Node, b: Node) -> Bool:
        return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2

    @staticmethod
    @always_inline
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_mmul_fw, Self.base_case_depth](c, a, b)

    @staticmethod
    @always_inline
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mmul_bw_a, Self.base_case_depth](c, a, b)
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_mmul_bw_b, Self.base_case_depth](c, a, b)

    @parameter
    @staticmethod
    @always_inline
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

    @parameter
    @staticmethod
    @always_inline
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
    @always_inline
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
