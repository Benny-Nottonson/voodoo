from random import random_float64
from algorithm import vectorize
from voodoo import Node
from .constants import DType_F32, nelts
from algorithm import *


trait Operation:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        ...

    @staticmethod
    fn bw(node: Node, parent1: Node):
        ...


struct Copy(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy[_nelts: Int](i: Int):
            node.store_data[_nelts](i, parent1.load_data[_nelts](i))

        vectorize[nelts, vectorized_copy](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](i, parent1.load_grad[_nelts](i))

        vectorize[nelts, vectorized_copy_bw](node.load_cap())


struct Sum(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_sum[nelts: Int](i: Int):
            sum += parent1.load_data[nelts](i).reduce_add()

        vectorize[nelts, vectorized_sum](parent1.load_cap())
        node.store_data(0, sum)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sum_bw[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i, parent1.load_grad[nelts](i) + node.load_grad(0)
            )

        vectorize[nelts, vectorized_sum_bw](parent1.load_cap())


struct Reshape(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
            let offset = s * parent1.cap_ptr.load()

            @parameter
            fn vectorized_reshape[nelts: Int](i: Int):
                node.store_data[nelts](i, parent1.load_data[nelts](i))

            vectorize[nelts, vectorized_reshape](parent1.cap_ptr.load())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
            let offset = s * parent1.cap_ptr.load()

            @parameter
            fn vectorized_reshape[nelts: Int](i: Int):
                parent1.store_grad[nelts](
                    i, parent1.load_grad[nelts](i) + node.load_grad[nelts](i)
                )

            vectorize[nelts, vectorized_reshape](parent1.cap_ptr.load())


struct Transpose(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let M = parent1.shape_ptr.load().load(num_dims - 2)
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp[nelts: Int](j: Int):
                    node.store_data[nelts](
                        offset + j * M + i, parent1.load_data[nelts](offset + i * N + j)
                    )

                vectorize[nelts, vectorized_transp](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let M = parent1.shape_ptr.load().load(num_dims - 2)
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp_bw[nelts: Int](j: Int):
                    parent1.store_grad[nelts](
                        offset + j * M + i,
                        parent1.load_grad[nelts](offset + j * M + i)
                        + node.load_grad[nelts](offset + i * N + j),
                    )

                vectorize[nelts, vectorized_transp_bw](N)


struct MaxPool2D(Operation):
    @staticmethod
    fn fw(b: Node, a: Node):
        let padding = b.other_params_ptr.load().load(0)
        let stride = b.other_params_ptr.load().load(1)
        let kernel_width = b.other_params_ptr.load().load(2)
        let kernel_height = b.other_params_ptr.load().load(3)

        for p in range(a.shape_ptr.load().load(0)):
            for i in range(a.shape_ptr.load().load(1)):
                for x in range(
                    0,
                    a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding,
                    stride,
                ):
                    for y in range(
                        0,
                        a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                        stride,
                    ):
                        var arg_max: Int = 0
                        var max_val: Float32 = -1000000.0
                        for dx in range(kernel_width):
                            for dy in range(kernel_height):
                                let ix = x - padding + dx
                                let iy = y - padding + dy
                                if (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    continue
                                let idx = Self.index(
                                    p,
                                    i,
                                    ix,
                                    iy,
                                    a.shape_ptr.load().load(1),
                                    a.shape_ptr.load().load(2),
                                    a.shape_ptr.load().load(3),
                                )
                                let entry = a.load_data(idx)
                                if entry > max_val:
                                    max_val = entry
                                    arg_max = idx
                        let idx = Self.index(
                            p,
                            i,
                            (x) // stride,
                            (y) // stride,
                            b.shape_ptr.load().load(1),
                            b.shape_ptr.load().load(2),
                            b.shape_ptr.load().load(3),
                        )
                        b.store_data(idx, max_val)

    @staticmethod
    fn bw(b: Node, a: Node):
        let padding = b.other_params_ptr.load().load(0)
        let stride = b.other_params_ptr.load().load(1)
        let kernel_width = b.other_params_ptr.load().load(2)
        let kernel_height = b.other_params_ptr.load().load(3)

        for p in range(a.shape_ptr.load().load(0)):
            for i in range(a.shape_ptr.load().load(1)):
                for x in range(
                    0,
                    a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding,
                    stride,
                ):
                    for y in range(
                        0,
                        a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                        stride,
                    ):
                        var arg_max: Int = 0
                        var max_val: Float32 = -1000000.0
                        for dx in range(kernel_width):
                            for dy in range(kernel_height):
                                let ix = x - padding + dx
                                let iy = y - padding + dy
                                if (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    continue
                                let idx = Self.index(
                                    p,
                                    i,
                                    ix,
                                    iy,
                                    a.shape_ptr.load().load(1),
                                    a.shape_ptr.load().load(2),
                                    a.shape_ptr.load().load(3),
                                )
                                let entry = a.load_data(idx)
                                if entry > max_val:
                                    max_val = entry
                                    arg_max = idx
                        let b_grad_idx = Self.index(
                            p,
                            i,
                            (x) // stride,
                            (y) // stride,
                            b.shape_ptr.load().load(1),
                            b.shape_ptr.load().load(2),
                            b.shape_ptr.load().load(3),
                        )
                        a.store_grad(
                            arg_max, a.load_grad(arg_max) + b.load_grad(b_grad_idx)
                        )

    @always_inline
    @staticmethod
    fn index(
        n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
    ) -> Int:
        return (
            n * (num_channels * height * width) + c * (height * width) + h * width + w
        )


struct Dropout(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let params = node.other_params_ptr.load()
        let keep_prob = params.load(0) / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout[_nelts: Int](i: Int):
            let data = parent1.load_data[_nelts](i)
            for i in range(_nelts):
                let rand = random_float64()
                node.store_data[_nelts](
                    i, (rand < keep_prob).cast[DType_F32]() * data * scale
                )

        vectorize[nelts, vectorized_dropout](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let params = node.other_params_ptr.load()
        let keep_prob = params.load(0) / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout_bw[_nelts: Int](i: Int):
            let data = parent1.load_data[_nelts](i)
            for i in range(_nelts):
                let rand = random_float64()
                parent1.store_grad[_nelts](
                    i,
                    parent1.load_grad[_nelts](i)
                    + (rand < keep_prob).cast[DType_F32]()
                    * node.load_grad[_nelts](i)
                    * scale,
                )

        vectorize[nelts, vectorized_dropout_bw](node.load_cap())
