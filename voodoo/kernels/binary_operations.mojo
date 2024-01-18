from algorithm import vectorize
from math import max
from voodoo import Node
from voodoo.utils import (
    recursive_broadcast,
    recursive_broadcast_bw,
)


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

        DTypePointer.prefetch[prefetch_read](a.data.load())
        DTypePointer.prefetch[prefetch_read](b.data.load())
        DTypePointer.prefetch[prefetch_read](c.data.load())
        DTypePointer.prefetch[prefetch_write](c.data.load())

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

        DTypePointer.prefetch[prefetch_read](a.data.load(1))
        DTypePointer.prefetch[prefetch_write](a.data.load(1))
        DTypePointer.prefetch[prefetch_read](b.data.load(0))
        DTypePointer.prefetch[prefetch_read](c.data.load(1))

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

        DTypePointer.prefetch[prefetch_read](a.data.load(0))
        DTypePointer.prefetch[prefetch_read](b.data.load(1))
        DTypePointer.prefetch[prefetch_write](b.data.load(1))
        DTypePointer.prefetch[prefetch_read](c.data.load(1))

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
        let params = c.other_params_ptr.load()

        let padding_x = params.load(0)
        let padding_y = params.load(1)
        let stride_x = params.load(2)
        let stride_y = params.load(3)

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

        let input_size = input_width * input_height
        let kernel_size = kernel_width * kernel_height
        let output_size = output_width * output_height

        let input_size_c = input_size * channels
        let kernel_size_c = kernel_size * channels
        let output_size_c = output_size * channels

        DTypePointer.prefetch[prefetch_read](a.data.load())
        DTypePointer.prefetch[prefetch_read](b.data.load())
        DTypePointer.prefetch[prefetch_write](c.data.load())

        for batch in range(batches):
            let batch_offset = batch * output_size_c
            for channel in range(channels):
                let channel_input_offset = channel * input_size + batch_offset
                let channel_kernel_offset = channel * kernel_size
                let channel_output_offset = channel * output_size
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        var sum: Float32 = 0.0
                        for kernel_x in range(kernel_width):

                            @parameter
                            fn fw_vec[nelts: Int](kernel_y: Int):
                                sum += (
                                    a.load_data[nelts](
                                        channel_input_offset
                                        + (output_x * stride_x + kernel_x - padding_x)
                                        * input_height
                                        + (output_y * stride_y + kernel_y - padding_y)
                                    )
                                    * b.load_data[nelts](
                                        channel_kernel_offset
                                        + kernel_x * kernel_height
                                        + kernel_y
                                    )
                                ).reduce_add()

                            vectorize[nelts, fw_vec](kernel_height)

                        c.store_data(
                            batch_offset
                            + channel_output_offset
                            + output_x * output_height
                            + output_y,
                            sum,
                        )

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        let params = c.other_params_ptr.load()

        let padding_x = params.load(0)
        let padding_y = params.load(1)
        let stride_x = params.load(2)
        let stride_y = params.load(3)

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

        let input_size = input_width * input_height
        let kernel_size = kernel_width * kernel_height
        let output_size = output_width * output_height

        let input_size_c = input_size * channels
        let kernel_size_c = kernel_size * channels
        let output_size_c = output_size * channels

        DTypePointer.prefetch[prefetch_read](a.data.load())
        DTypePointer.prefetch[prefetch_write](a.data.load(1))
        DTypePointer.prefetch[prefetch_read](b.data.load())
        DTypePointer.prefetch[prefetch_write](b.data.load(1))
        DTypePointer.prefetch[prefetch_read](c.data.load(1))

        for batch in range(batches):
            let batch_offset_output = batch * output_size_c
            let batch_offset_input = batch * input_size_c
            for channel in range(channels):
                let channel_output_offset = channel * output_size + batch_offset_output
                let channel_input_offset = channel * input_size + batch_offset_input
                let channel_kernel_offset = channel * kernel_size
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        let c_grad = c.load_grad(
                            batch_offset_output + output_x * output_height + output_y
                        )

                        for kernel_x in range(kernel_width):

                            @parameter
                            fn bw_vec[nelts: Int](kernel_y: Int):
                                a.store_grad[nelts](
                                    channel_input_offset
                                    + (output_x * stride_x + kernel_x - padding_x)
                                    * input_height
                                    + (output_y * stride_y + kernel_y - padding_y),
                                    b.load_data[nelts](
                                        channel_kernel_offset
                                        + kernel_x * kernel_height
                                        + kernel_y
                                    )
                                    * c_grad,
                                )

                                b.store_grad[nelts](
                                    channel_kernel_offset
                                    + kernel_x * kernel_height
                                    + kernel_y,
                                    a.load_data[nelts](
                                        channel_input_offset
                                        + (output_x * stride_x + kernel_x - padding_x)
                                        * input_height
                                        + (output_y * stride_y + kernel_y - padding_y)
                                    )
                                    * c_grad,
                                )

                            vectorize[nelts, bw_vec](kernel_height)


struct MaxPool2D:
    @staticmethod
    fn fw(c: Node, a: Node):
        let params = c.other_params_ptr.load()

        let kernel_width = params.load(0)
        let kernel_height = params.load(1)
        let stride = params.load(2)
        let padding = params.load(3)

        let input_shape = a.shape_ptr.load()
        let output_shape = c.shape_ptr.load()

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)
        let input_height = input_shape.load(3)

        let output_width = output_shape.load(2)
        let output_height = output_shape.load(3)

        let input_size = input_width * input_height
        let output_size = output_width * output_height

        let input_size_c = input_size * channels
        let output_size_c = output_size * channels

        DTypePointer.prefetch[prefetch_read](a.data.load())
        DTypePointer.prefetch[prefetch_write](c.data.load())

        for batch in range(batches):
            let batch_offset = batch * output_size_c
            for channel in range(channels):
                let channel_input_offset = channel * input_size + batch_offset
                let channel_output_offset = channel * output_size
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        var max_val: Float32 = -f32_max
                        for kernel_x in range(kernel_width):

                            @parameter
                            fn fw_vec[nelts: Int](kernel_y: Int):
                                max_val = max(
                                    max_val,
                                    a.load_data[nelts](
                                        channel_input_offset
                                        + (output_x * stride + kernel_x - padding)
                                        * input_height
                                        + (output_y * stride + kernel_y - padding)
                                    ).reduce_max()
                                )

                            vectorize[nelts, fw_vec](kernel_height)

                        c.store_data(
                            batch_offset
                            + channel_output_offset
                            + output_x * output_height
                            + output_y,
                            max_val,
                        )

    @staticmethod
    fn bw(c: Node, a: Node):
        let params = c.other_params_ptr.load()

        let padding_x = params.load(0)
        let padding_y = params.load(1)
        let stride_x = params.load(2)
        let stride_y = params.load(3)
        let kernel_width = params.load(4)
        let kernel_height = params.load(5)
        
        let input_shape = a.shape_ptr.load()
        let output_shape = c.shape_ptr.load()

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)
        let input_height = input_shape.load(3)

        let output_width = output_shape.load(2)
        let output_height = output_shape.load(3)

        let input_size = input_width * input_height
        let output_size = output_width * output_height

        let input_size_c = input_size * channels
        let output_size_c = output_size * channels

        DTypePointer.prefetch[prefetch_read](a.data.load())
        DTypePointer.prefetch[prefetch_write](a.data.load(1))
        DTypePointer.prefetch[prefetch_read](c.data.load())
        DTypePointer.prefetch[prefetch_read](c.data.load(1))

        for batch in range(batches):
            let batch_offset_output = batch * output_size_c
            let batch_offset_input = batch * input_size_c
            for channel in range(channels):
                let channel_output_offset = channel * output_size + batch_offset_output
                let channel_input_offset = channel * input_size + batch_offset_input
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        let c_grad = c.load_grad(
                            batch_offset_output + output_x * output_height + output_y
                        )

                        for kernel_x in range(kernel_width):

                            @parameter
                            fn bw_vec[nelts: Int](kernel_y: Int):
                                let input_offset = channel_input_offset
                                    + (output_x * stride_x + kernel_x - padding_x)
                                    * input_height
                                    + (output_y * stride_y + kernel_y - padding_y)
                                let output_offset = channel_output_offset
                                    + output_x * output_height
                                    + output_y

                                a.store_grad[nelts](
                                    input_offset,
                                    c_grad if a.load_data[nelts](input_offset) == c.load_data[nelts](output_offset)
                                    else 0.0,
                                )

                            vectorize[nelts, bw_vec](kernel_height)
