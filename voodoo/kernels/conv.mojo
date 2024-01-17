from math import max, min
from voodoo import Node

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

        for batch in range(batches):
            for channel in range(channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        var sum: Float32 = 0.0
                        for kernel_x in range(kernel_width):
                            for kernel_y in range(kernel_height):
                                sum += a.load_data(
                                    batch * channels * input_width * input_height +
                                    channel * input_width * input_height +
                                    (output_x * stride_x + kernel_x - padding_x) * input_height +
                                    (output_y * stride_y + kernel_y - padding_y)
                                ) * b.load_data(
                                    channel * kernel_width * kernel_height +
                                    kernel_x * kernel_height +
                                    kernel_y
                                )

                        c.store_data(
                            batch * channels * output_width * output_height +
                            channel * output_width * output_height +
                            output_x * output_height +
                            output_y,
                            sum
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

        
        for batch in range(batches):
            for channel in range(channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        let c_grad = c.load_grad(
                            batch * channels * output_width * output_height +
                            channel * output_width * output_height +
                            output_x * output_height +
                            output_y
                        )

                        for kernel_x in range(kernel_width):
                            for kernel_y in range(kernel_height):
                                a.store_grad(
                                    batch * channels * input_width * input_height +
                                    channel * input_width * input_height +
                                    (output_x * stride_x + kernel_x - padding_x) * input_height +
                                    (output_y * stride_y + kernel_y - padding_y),
                                    b.load_data(
                                        channel * kernel_width * kernel_height +
                                        kernel_x * kernel_height +
                                        kernel_y
                                    ) * c_grad
                                )

                                b.store_grad(
                                    channel * kernel_width * kernel_height +
                                    kernel_x * kernel_height +
                                    kernel_y,
                                    a.load_data(
                                        batch * channels * input_width * input_height +
                                        channel * input_width * input_height +
                                        (output_x * stride_x + kernel_x - padding_x) * input_height +
                                        (output_y * stride_y + kernel_y - padding_y)
                                    ) * c_grad
                                )