from math import max, min
from voodoo import Node
from algorithm import vectorize


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
