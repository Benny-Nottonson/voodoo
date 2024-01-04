from random import random_float64
from algorithm import vectorize, parallelize
from voodoo import Node
from .constants import DType_F32, nelts, workers


trait BinaryOperation:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        ...

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        ...


struct Conv2D(BinaryOperation):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        let padding = c.other_params_ptr.load().load(0)
        let stride = c.other_params_ptr.load().load(1)

        @parameter
        fn batch_loop(i: Int):
            for j in range(b.shape_ptr.load().load(0)):
                for x in range(c.shape_ptr.load().load(2)):
                    for y in range(c.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        # TODO: Vectorize
                        for k in range(a.shape_ptr.load().load(1)):
                            for dx in range(b.shape_ptr.load().load(2)):

                                @parameter
                                fn inner_loop[_nelts: Int](dy: Int):
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
                                            a.load_data[_nelts](a_index)
                                            * b.load_data[_nelts](b_index)
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
                                        # add to patch sum
                                        patch_sum += (
                                            a.load_data(a_index) * c.load_grad(c_grad_index)
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
                                fn dy_loop[_nelts: Int](dy: Int):
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
                                            c.load_grad[_nelts](c_grad_index)
                                            * c.load_data[_nelts](b_index)
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
