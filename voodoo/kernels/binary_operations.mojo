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
        let padding_x =  c.other_params_ptr.load().load(0)
        let padding_y =  c.other_params_ptr.load().load(1)
        let stride_x =  c.other_params_ptr.load().load(2)
        let stride_y =  c.other_params_ptr.load().load(3)

        let input_shape = a.shape_ptr.load()
        let kernel_shape = b.shape_ptr.load()
        let output_shape = c.shape_ptr.load()

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)
        let input_height = input_shape.load(3)

        let kernel_width = kernel_shape.load(1)
        let kernel_height = kernel_shape.load(2)

        let output_width = output_shape.load(2)
        let output_height = output_shape.load(3)

        for batch in range(batches):
            for channel in range(channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        let input_x = output_x - padding_x
                        let input_y = output_y - padding_y

                        let input_off = (
                            batch * channels * input_width * input_height
                            + channel * input_width * input_height
                            + input_y * input_width
                            + input_x
                        )

                        let output_off = (
                            batch * channels * output_width * output_height
                            + channel * output_width * output_height
                            + output_y * output_width
                            + output_x
                        )

                        var output = c.load_data(output_off)

                        for kernel_y in range(kernel_height):
                            for kernel_x in range(kernel_width):
                                let kernel_off = (
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let input_off = (
                                    input_off
                                    + kernel_y * input_width
                                    + kernel_x
                                )

                                let input = a.load_data(input_off)
                                let kernel = b.load_data(kernel_off)

                                output = output + input * kernel

                        c.store_data(output_off, output)

    @staticmethod   
    fn bw(c: Node, a: Node, b: Node):
        let padding_x =  c.other_params_ptr.load().load(0)
        let padding_y =  c.other_params_ptr.load().load(1)
        let stride_x =  c.other_params_ptr.load().load(2)
        let stride_y =  c.other_params_ptr.load().load(3)

        let input_shape = a.shape_ptr.load()
        let kernel_shape = b.shape_ptr.load()
        let output_shape = c.shape_ptr.load()

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)
        let input_height = input_shape.load(3)

        let kernel_width = kernel_shape.load(1)
        let kernel_height = kernel_shape.load(2)

        let output_width = output_shape.load(2)
        let output_height = output_shape.load(3)

        for batch in range(batches):
            for channel in range(channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        let input_x = output_x - padding_x
                        let input_y = output_y - padding_y

                        let input_off = (
                            batch * channels * input_width * input_height
                            + channel * input_width * input_height
                            + input_y * input_width
                            + input_x
                        )

                        let output_off = (
                            batch * channels * output_width * output_height
                            + channel * output_width * output_height
                            + output_y * output_width
                            + output_x
                        )

                        let output_grad = c.load_grad(output_off)

                        for kernel_y in range(kernel_height):
                            for kernel_x in range(kernel_width):
                                let kernel_off = (
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let input_off = (
                                    input_off
                                    + kernel_y * input_width
                                    + kernel_x
                                )

                                let input = a.load_data(input_off)
                                let kernel = b.load_data(kernel_off)
                                let input_grad = a.load_grad(input_off)
                                let kernel_grad = b.load_grad(kernel_off)

                                let final_input_grad = input_grad + output_grad * kernel
                                let final_kernel_grad = kernel_grad + output_grad * input

                                a.store_grad(input_off, final_input_grad)
                                b.store_grad(kernel_off, final_kernel_grad)
        