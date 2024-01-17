from algorithm import vectorize
from math import max
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
                                    input_off + kernel_y * input_width + kernel_x
                                )

                                let input = a.load_data(input_off)
                                let kernel = b.load_data(kernel_off)

                                output = output + input * kernel

                        c.store_data(output_off, output)

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

        for i in range(channels):
            for j in range(channels):
                for x in range(kernel_width):
                    let _x = x * x_diff
                    for y in range(kernel_height):
                        let _y = y * y_diff
                        var patch_sum: Float32 = 0.0
                        for b in range(batches):
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
                                        let a_index = index(
                                            b,
                                            i,
                                            ix,
                                            iy,
                                            channels,
                                            input_width,
                                            input_height,
                                        )
                                        let c_grad_index = index(
                                            b,
                                            j,
                                            dx,
                                            dy,
                                            channels,
                                            output_width,
                                            output_height,
                                        )
                                        patch_sum += (
                                            a.load_data(a_index)
                                            * c.load_grad(c_grad_index)
                                        ).reduce_add()
                        let b_grad_index = index(
                            i,
                            j,
                            x,
                            y,
                            channels,
                            kernel_width,
                            kernel_height,
                        )
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
