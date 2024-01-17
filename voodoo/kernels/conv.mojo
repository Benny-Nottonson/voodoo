from algorithm import vectorize, tile
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

        let input_size = input_width * input_height
        let output_size = output_width * output_height
        let input_size_c = input_size * channels
        let output_size_c = output_size * channels

        for batch in range(batches):
            let batch_offset_input = batch * input_size_c
            let batch_offset_output = batch * output_size_c
            for channel in range(channels):
                let channel_offset_input = channel * input_size + batch_offset_input
                let channel_offset_output = channel * output_size + batch_offset_output
                let kernel_offset = channel * kernel_width * kernel_height
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        let input_off = channel_offset_input + (
                            output_y - padding_y
                        ) * input_width + (output_x - padding_x)
                        let output_off = channel_offset_output + output_y * output_width + output_x

                        var output: Float32 = 0.0

                        @parameter
                        fn tiled_kernel_add(kernel_y: Int, kernel_x: Int):
                            let kernel_off = kernel_y * kernel_width + kernel_offset
                            let input_off_y = input_off + kernel_off
                            output += (
                                a.load_data(input_off_y + kernel_x)
                                * b.load_data(kernel_off + kernel_x)
                            ).reduce_add()

                        tile[tiled_kernel_add](
                            max(0, padding_x - output_x),
                            min(kernel_width, input_width + padding_x - output_x),
                            kernel_width,
                        )

                        c.store_data(output_off, c.load_data(output_off) + output)

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

        let x_diff = stride_x - padding_x
        let y_diff = stride_y - padding_y

        let input_size = input_width * input_height
        let output_size = output_width * output_height
        let kernel_size = kernel_width * kernel_height
        let input_size_c = input_size * channels
        let output_size_c = output_size * channels
        let kernel_size_c = kernel_size * channels

        for in_channels in range(channels):
            for out_channels in range(channels):
                for kernel_x in range(kernel_width):
                    let _x = kernel_x * x_diff
                    for kernel_y in range(kernel_height):
                        let _y = kernel_y * y_diff
                        var patch_sum: Float32 = 0.0
                        for batches in range(batches):
                            for dx in range(output_width):
                                for dy in range(output_height):
                                    let ix = _x + dx
                                    let iy = _y + dy
                                    if not (
                                        ix < 0
                                        or iy < 0
                                        or ix >= input_shape.load(2)
                                        or iy >= input_shape.load(3)
                                    ):
                                        let c_index = batches * (
                                            output_size_c
                                        ) + output_size_c + dy * output_width + dx
                                        let b_index = out_channels * (
                                            kernel_size_c
                                        ) + kernel_size_c + kernel_y * kernel_width + kernel_x
                                        let a_index = batches * (
                                            input_size_c
                                        ) + input_size_c + iy * input_width + ix
                                        patch_sum += (
                                            c.load_grad(c_index) * b.load_data(b_index)
                                        ).reduce_add()
                        let b_grad_index = out_channels * (
                            kernel_size_c
                        ) + kernel_size_c + kernel_y * kernel_width + kernel_x
                        b.store_grad(b_grad_index, patch_sum)

        for p in range(batches):
            for j in range(channels):
                for i in range(channels):
                    for x in range(input_width):
                        for y in range(input_height):
                            let a_grad_index = index(
                                p,
                                j,
                                x,
                                y,
                                channels,
                                input_width,
                                input_height,
                            )
                            var patch_sum: Float32 = a.load_grad(a_grad_index)
                            for dx in range(kernel_width):
                                for dy in range(kernel_height):
                                    let ix = x * stride_x - dx + padding_x
                                    let iy = y * stride_y - dy + padding_y
                                    if not (
                                        ix < 0
                                        or iy < 0
                                        or ix >= output_width
                                        or iy >= output_height
                                    ):
                                        let c_grad_index = index(
                                            p,
                                            i,
                                            ix,
                                            iy,
                                            channels,
                                            input_width,
                                            input_height,
                                        )
                                        let b_index = index(
                                            i,
                                            j,
                                            kernel_width - dx - 1,
                                            kernel_height - dy - 1,
                                            channels,
                                            kernel_width,
                                            kernel_height,
                                        )
                                        patch_sum += (
                                            c.load_grad(c_grad_index)
                                            * c.load_data(b_index)
                                        ).reduce_add()

                            a.store_grad(a_grad_index, patch_sum)


@always_inline
fn index(
    n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
) -> Int:
    return n * (num_channels * height * width) + c * (height * width) + h * width + w
