from random import random_float64
from algorithm import *
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)
from sys.intrinsics import PrefetchOptions

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

        DTypePointer.prefetch[prefetch_options](a.data.load())
        DTypePointer.prefetch[prefetch_options](b.data.load())
        DTypePointer.prefetch[prefetch_options](c.data.load())

        for m in range(M):
            let _a_off = offset_a + m * K
            let _c_off = offset_c + m * N

            for k in range(K):
                let a_off = _a_off + k
                let a_scalar = a.load_data(a_off)
                let _b_off = offset_b + k * N

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

        DTypePointer.prefetch[prefetch_options](a.data.load(1))
        DTypePointer.prefetch[prefetch_options](b.data.load(0))
        DTypePointer.prefetch[prefetch_options](c.data.load(1))

        for m in range(M):
            let _a_off = offset_a + m * K
            let _c_off = offset_c + m * N

            for n in range(N):
                let c_offset = _c_off + n
                let c_grad = c.load_grad(c_offset)
                let _b_off = offset_b + n

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

        DTypePointer.prefetch[prefetch_options](a.data.load(0))
        DTypePointer.prefetch[prefetch_options](b.data.load(1))
        DTypePointer.prefetch[prefetch_options](c.data.load(1))

        for k in range(K):
            let _a_off = offset_a + k
            let _b_off = offset_b + k * N

            for m in range(M):
                let a_data = a.load_data(_a_off + m * K)
                let _c_off = offset_c + m * N

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


struct Conv2D:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        let padding: StaticIntTuple[2] = (
            c.other_params_ptr.load().load(0),
            c.other_params_ptr.load().load(0),
        )
        let stride: StaticIntTuple[2] = (
            c.other_params_ptr.load().load(1),
            c.other_params_ptr.load().load(1),
        )
        
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
                                    let ix = x * stride[0] - padding[0] + dx
                                    let iy = y * stride[1] - padding[1] + dy
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
    fn bw(c: Node, a: Node, b: Node):
        let padding: StaticIntTuple[2] = (
            c.other_params_ptr.load().load(0),
            c.other_params_ptr.load().load(0),
        )
        let stride: StaticIntTuple[2] = (
            c.other_params_ptr.load().load(1),
            c.other_params_ptr.load().load(1),
        )

        for i in range(a.shape_ptr.load().load(1)):
            for j in range(b.shape_ptr.load().load(0)):
                for x in range(b.shape_ptr.load().load(2)):
                    for y in range(b.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        for b in range(a.shape_ptr.load().load(0)):
                            for dx in range(c.shape_ptr.load().load(2)):
                                for dy in range(c.shape_ptr.load().load(3)):
                                    let ix = x * stride[0] - padding[0] + dx
                                    let iy = y * stride[1] - padding[1] + dy
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
                                    let ix = x * stride[0] - dx + padding[0]
                                    let iy = y * stride[1] - dy + padding[1]
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

    @staticmethod
    @always_inline
    fn index(
        n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
    ) -> Int:
        return (
            n * (num_channels * height * width) + c * (height * width) + h * width + w
        )
