from algorithm import vectorize
from math import max
from voodoo import Node


struct Conv1D:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        let params = c.other_params_ptr.load()

        let padding_x = params.load(0)
        let stride_x = params.load(1)

        let input_shape = a.shape_ptr.load()
        let kernel_shape = b.shape_ptr.load()
        let output_shape = c.shape_ptr.load()

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)

        let kernel_width = kernel_shape.load(1)

        let output_width = output_shape.load(2)

        let im2col = DTypePointer[DType.float32].alloc(
            batches * output_width * kernel_width * channels
        )

        for batch in range(batches):
            for channel in range(channels):
                for output_x in range(output_width):
                    for kernel_x in range(kernel_width):
                        let input_x = output_x * stride_x + kernel_x - padding_x

                        if input_x < 0 or input_x >= input_width:
                            im2col.store(
                                batch * output_width * kernel_width * channels
                                + output_x * kernel_width * channels
                                + kernel_x * channels
                                + channel,
                                0.0,
                            )
                        else:
                            im2col.store(
                                batch * output_width * kernel_width * channels
                                + output_x * kernel_width * channels
                                + kernel_x * channels
                                + channel,
                                a.load_data(
                                    batch * input_width * channels
                                    + input_x * channels
                                    + channel
                                ),
                            )

        for batch in range(batches):
            for output_x in range(output_width):
                for kernel_x in range(kernel_width):
                    for channel in range(channels):
                        let kernel_value = b.load_data(
                            channel * kernel_width + kernel_x
                        )

                        let output_value = c.load_data(
                            batch * output_width * channels
                            + output_x * channels
                            + channel
                        )

                        let im2col_value = im2col.load(
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel
                        )

                        c.store_data(
                            batch * output_width * channels
                            + output_x * channels
                            + channel,
                            output_value + kernel_value * im2col_value,
                        )

        im2col.free()

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        let params = c.other_params_ptr.load()

        let padding_x = params.load(0)
        let stride_x = params.load(1)

        let input_shape = a.shape_ptr.load()
        let kernel_shape = b.shape_ptr.load()
        let output_shape = c.shape_ptr.load()

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)

        let kernel_width = kernel_shape.load(1)

        let output_width = output_shape.load(2)

        let im2col = DTypePointer[DType.float32].alloc(
            batches * output_width * kernel_width * channels
        )

        for batch in range(batches):
            for channel in range(channels):
                for output_x in range(output_width):
                    for kernel_x in range(kernel_width):
                        let input_x = output_x * stride_x + kernel_x - padding_x

                        if input_x < 0 or input_x >= input_width:
                            im2col.store(
                                batch * output_width * kernel_width * channels
                                + output_x * kernel_width * channels
                                + kernel_x * channels
                                + channel,
                                0.0,
                            )
                        else:
                            im2col.store(
                                batch * output_width * kernel_width * channels
                                + output_x * kernel_width * channels
                                + kernel_x * channels
                                + channel,
                                a.load_data(
                                    batch * input_width * channels
                                    + input_x * channels
                                    + channel
                                ),
                            )

        for batch in range(batches):
            for output_x in range(output_width):
                for kernel_x in range(kernel_width):
                    for channel in range(channels):
                        let kernel_value = b.load_data(
                            channel * kernel_width + kernel_x
                        )

                        let output_value = c.load_data(
                            batch * output_width * channels
                            + output_x * channels
                            + channel
                        )

                        let im2col_value = im2col.load(
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel
                        )

                        a.store_grad(
                            batch * input_width * channels
                            + (output_x * stride_x + kernel_x - padding_x) * channels
                            + channel,
                            a.load_grad(
                                batch * input_width * channels
                                + (output_x * stride_x + kernel_x - padding_x)
                                * channels
                                + channel
                            )
                            + kernel_value
                            * c.load_grad(
                                batch * output_width * channels
                                + output_x * channels
                                + channel
                            ),
                        )

                        b.store_grad(
                            channel * kernel_width + kernel_x,
                            b.load_grad(channel * kernel_width + kernel_x)
                            + output_value * im2col_value,
                        )

        im2col.free()


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

        let im2col = DTypePointer[DType.float32].alloc(
            batches
            * output_width
            * output_height
            * kernel_width
            * kernel_height
            * channels
        )

        for batch in range(batches):
            for channel in range(channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        for kernel_y in range(kernel_height):
                            for kernel_x in range(kernel_width):
                                let input_x = output_x * stride_x + kernel_x - padding_x
                                let input_y = output_y * stride_y + kernel_y - padding_y

                                if input_x < 0 or input_x >= input_width:
                                    im2col.store(
                                        batch
                                        * output_width
                                        * output_height
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_y
                                        * output_width
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_x
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + kernel_y * kernel_width * channels
                                        + kernel_x * channels
                                        + channel,
                                        0.0,
                                    )
                                elif input_y < 0 or input_y >= input_height:
                                    im2col.store(
                                        batch
                                        * output_width
                                        * output_height
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_y
                                        * output_width
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_x
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + kernel_y * kernel_width * channels
                                        + kernel_x * channels
                                        + channel,
                                        0.0,
                                    )
                                else:
                                    im2col.store(
                                        batch
                                        * output_width
                                        * output_height
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_y
                                        * output_width
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_x
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + kernel_y * kernel_width * channels
                                        + kernel_x * channels
                                        + channel,
                                        a.load_data(
                                            batch
                                            * input_width
                                            * input_height
                                            * channels
                                            + input_y * input_width * channels
                                            + input_x * channels
                                            + channel
                                        ),
                                    )

        for batch in range(batches):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    for kernel_y in range(kernel_height):
                        for kernel_x in range(kernel_width):
                            for channel in range(channels):
                                let kernel_value = b.load_data(
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let output_value = c.load_data(
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                let im2col_value = im2col.load(
                                    batch
                                    * output_width
                                    * output_height
                                    * kernel_width
                                    * kernel_height
                                    * channels
                                    + output_y
                                    * output_width
                                    * kernel_width
                                    * kernel_height
                                    * channels
                                    + output_x * kernel_width * kernel_height * channels
                                    + kernel_y * kernel_width * channels
                                    + kernel_x * channels
                                    + channel
                                )

                                c.store_data(
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel,
                                    output_value + kernel_value * im2col_value,
                                )

        im2col.free()

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

        let im2col = DTypePointer[DType.float32].alloc(
            batches
            * output_width
            * output_height
            * kernel_width
            * kernel_height
            * channels
        )

        for batch in range(batches):
            for channel in range(channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        for kernel_y in range(kernel_height):
                            for kernel_x in range(kernel_width):
                                let input_x = output_x * stride_x + kernel_x - padding_x
                                let input_y = output_y * stride_y + kernel_y - padding_y

                                if input_x < 0 or input_x >= input_width:
                                    im2col.store(
                                        batch
                                        * output_width
                                        * output_height
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_y
                                        * output_width
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_x
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + kernel_y * kernel_width * channels
                                        + kernel_x * channels
                                        + channel,
                                        0.0,
                                    )
                                elif input_y < 0 or input_y >= input_height:
                                    im2col.store(
                                        batch
                                        * output_width
                                        * output_height
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_y
                                        * output_width
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_x
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + kernel_y * kernel_width * channels
                                        + kernel_x * channels
                                        + channel,
                                        0.0,
                                    )
                                else:
                                    im2col.store(
                                        batch
                                        * output_width
                                        * output_height
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_y
                                        * output_width
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + output_x
                                        * kernel_width
                                        * kernel_height
                                        * channels
                                        + kernel_y * kernel_width * channels
                                        + kernel_x * channels
                                        + channel,
                                        a.load_data(
                                            batch
                                            * input_width
                                            * input_height
                                            * channels
                                            + input_y * input_width * channels
                                            + input_x * channels
                                            + channel
                                        ),
                                    )

        for batch in range(batches):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    for kernel_y in range(kernel_height):
                        for kernel_x in range(kernel_width):
                            for channel in range(channels):
                                let kernel_value = b.load_data(
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let output_value = c.load_data(
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                let im2col_value = im2col.load(
                                    batch
                                    * output_width
                                    * output_height
                                    * kernel_width
                                    * kernel_height
                                    * channels
                                    + output_y
                                    * output_width
                                    * kernel_width
                                    * kernel_height
                                    * channels
                                    + output_x * kernel_width * kernel_height * channels
                                    + kernel_y * kernel_width * channels
                                    + kernel_x * channels
                                    + channel
                                )

                                a.store_grad(
                                    batch * input_width * input_height * channels
                                    + (output_y * stride_y + kernel_y - padding_y)
                                    * input_width
                                    * channels
                                    + (output_x * stride_x + kernel_x - padding_x)
                                    * channels
                                    + channel,
                                    a.load_grad(
                                        batch * input_width * input_height * channels
                                        + (output_y * stride_y + kernel_y - padding_y)
                                        * input_width
                                        * channels
                                        + (output_x * stride_x + kernel_x - padding_x)
                                        * channels
                                        + channel
                                    )
                                    + kernel_value
                                    * c.load_grad(
                                        batch * output_width * output_height * channels
                                        + output_y * output_width * channels
                                        + output_x * channels
                                        + channel
                                    ),
                                )

                                b.store_grad(
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x,
                                    b.load_grad(
                                        channel * kernel_width * kernel_height
                                        + kernel_y * kernel_width
                                        + kernel_x
                                    )
                                    + output_value * im2col_value,
                                )

        im2col.free()
