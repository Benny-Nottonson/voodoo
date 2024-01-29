from algorithm import vectorize, tile
from math import max
from voodoo import Node
from ..constants import NELTS, PREFETCH_READ, PREFETCH_WRITE

alias tile_sizes = VariadicList[Int](32, 16, 8, 4, 2, 1)


struct Conv1D:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        let params = c.other_params

        let padding_x = params.load(0)
        let stride_x = params.load(1)

        let input_shape = a.shape
        let kernel_shape = b.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)

        let kernel_width = kernel_shape.load(1)

        let output_width = output_shape.load(2)

        let im2col = im2col2D(
            a.data.load(),
            input_shape,
            kernel_shape,
            output_shape,
            padding_x,
            stride_x,
        )

        for batch in range(batches):
            for output_x in range(output_width):
                for kernel_x in range(kernel_width):
                    for channel in range(channels):
                        let kernel_value = b.data.load().load(
                            channel * kernel_width + kernel_x
                        )

                        let output_value = c.data.load().load(
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

                        c.data.load().store(
                            batch * output_width * channels
                            + output_x * channels
                            + channel,
                            output_value + kernel_value * im2col_value,
                        )

        im2col.free()

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        let params = c.other_params

        let padding_x = params.load(0)
        let stride_x = params.load(1)

        let input_shape = a.shape
        let kernel_shape = b.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)

        let kernel_width = kernel_shape.load(1)

        let output_width = output_shape.load(2)

        let im2col = im2col2D(
            a.data.load(),
            input_shape,
            kernel_shape,
            output_shape,
            padding_x,
            stride_x,
        )

        for batch in range(batches):
            for output_x in range(output_width):
                for kernel_x in range(kernel_width):
                    for channel in range(channels):
                        let kernel_value = b.data.load().load(
                            channel * kernel_width + kernel_x
                        )

                        let output_value = c.data.load().load(
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

                        a.data.load(1).store(
                            batch * input_width * channels
                            + (output_x * stride_x + kernel_x - padding_x) * channels
                            + channel,
                            a.data.load(1).load(
                                batch * input_width * channels
                                + (output_x * stride_x + kernel_x - padding_x)
                                * channels
                                + channel
                            )
                            + kernel_value
                            * c.data.load(1).load(
                                batch * output_width * channels
                                + output_x * channels
                                + channel
                            ),
                        )

                        b.data.load(1).store(
                            channel * kernel_width + kernel_x,
                            b.data.load(1).load(channel * kernel_width + kernel_x)
                            + output_value * im2col_value,
                        )

        im2col.free()


struct Conv2D:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        let params = c.other_params

        let padding_x = params.load(0)
        let padding_y = params.load(1)
        let stride_x = params.load(2)
        let stride_y = params.load(3)

        let input_shape = a.shape
        let kernel_shape = b.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)
        let input_height = input_shape.load(3)

        let kernel_width = kernel_shape.load(1)
        let kernel_height = kernel_shape.load(2)

        let output_width = output_shape.load(2)
        let output_height = output_shape.load(3)

        let im2col = im2col3D(
            a.data.load(),
            input_shape,
            kernel_shape,
            output_shape,
            padding_x,
            padding_y,
            stride_x,
            stride_y,
        )

        let a_data = a.data.load(0)
        let b_data = b.data.load(0)
        let c_data = c.data.load(0)

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](a_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](c_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](im2col)

        for batch in range(batches):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    for kernel_y in range(kernel_height):

                        @parameter
                        fn fw_vec[NELTS: Int](kernel_x: Int):
                            for channel in range(channels):
                                let kernel_value = b_data.simd_load[NELTS](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let output_value = c_data.simd_load[NELTS](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                let im2col_value = im2col.simd_load[NELTS](
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

                                c_data.simd_store[NELTS](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel,
                                    output_value + kernel_value * im2col_value,
                                )

                        vectorize[NELTS, fw_vec](kernel_width)

        im2col.free()

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        let params = c.other_params

        let padding_x = params.load(0)
        let padding_y = params.load(1)
        let stride_x = params.load(2)
        let stride_y = params.load(3)

        let input_shape = a.shape
        let kernel_shape = b.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)
        let input_height = input_shape.load(3)

        let kernel_width = kernel_shape.load(1)
        let kernel_height = kernel_shape.load(2)

        let output_width = output_shape.load(2)
        let output_height = output_shape.load(3)

        let im2col = im2col3D(
            a.data.load(),
            input_shape,
            kernel_shape,
            output_shape,
            padding_x,
            padding_y,
            stride_x,
            stride_y,
        )

        let b_data = b.data.load(0)
        let c_data = c.data.load(0)
        let a_grad = a.data.load(1)
        let b_grad = b.data.load(1)
        let c_grad = c.data.load(1)

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](c_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](im2col)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](a_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](c_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](a_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](b_grad)

        for batch in range(batches):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    for kernel_y in range(kernel_height):

                        @parameter
                        fn bw_vec[NELTS: Int](kernel_x: Int):
                            for channel in range(channels):
                                let kernel_value = b_data.simd_load[NELTS](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let output_value = c_data.simd_load[NELTS](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                let im2col_value = im2col.simd_load[NELTS](
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

                                a_grad.simd_store[NELTS](
                                    batch * input_width * input_height * channels
                                    + (output_y * stride_y + kernel_y - padding_y)
                                    * input_width
                                    * channels
                                    + (output_x * stride_x + kernel_x - padding_x)
                                    * channels
                                    + channel,
                                    a_grad.simd_load[NELTS](
                                        batch * input_width * input_height * channels
                                        + (output_y * stride_y + kernel_y - padding_y)
                                        * input_width
                                        * channels
                                        + (output_x * stride_x + kernel_x - padding_x)
                                        * channels
                                        + channel
                                    )
                                    + kernel_value
                                    * c_grad.simd_load[NELTS](
                                        batch * output_width * output_height * channels
                                        + output_y * output_width * channels
                                        + output_x * channels
                                        + channel
                                    ),
                                )

                                b_grad.simd_store[NELTS](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x,
                                    b_grad.simd_load[NELTS](
                                        channel * kernel_width * kernel_height
                                        + kernel_y * kernel_width
                                        + kernel_x
                                    )
                                    + output_value * im2col_value,
                                )

        im2col.free()


fn im2col2D(
    input: DTypePointer[DType.float32],
    input_shape: Vector[Int],
    kernel_shape: Vector[Int],
    output_shape: Vector[Int],
    padding: Int,
    stride: Int,
) -> DTypePointer[DType.float32]:
    let batches = input_shape.load(0)
    let channels = input_shape.load(1)
    let input_width = input_shape.load(2)

    let kernel_width = kernel_shape.load(1)

    let output_width = output_shape.load(2)

    let im2col = DTypePointer[DType.float32].alloc(
        batches * output_width * kernel_width * channels
    )

    DTypePointer[DType.float32].prefetch[PREFETCH_READ](input)
    DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](im2col)

    for batch in range(batches):
        for channel in range(channels):

            @parameter
            fn workgroup_function[NELTS: Int](output_x: Int):
                @parameter
                fn fw_vec[NELTS: Int](kernel_x: Int):
                    let input_x = output_x * stride + kernel_x - padding

                    if input_x < 0 or input_x >= input_width:
                        im2col.simd_store[NELTS](
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel,
                            0.0,
                        )
                    else:
                        im2col.simd_store[NELTS](
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel,
                            input.simd_load[NELTS](
                                batch * input_width * channels
                                + input_x * channels
                                + channel
                            ),
                        )

                vectorize[NELTS, fw_vec](kernel_width)

            tile[workgroup_function, tile_sizes](0, output_width)

    return im2col


fn im2col3D(
    input: DTypePointer[DType.float32],
    input_shape: Vector[Int],
    kernel_shape: Vector[Int],
    output_shape: Vector[Int],
    padding_x: Int,
    padding_y: Int,
    stride_x: Int,
    stride_y: Int,
) -> DTypePointer[DType.float32]:
    let batches = input_shape.load(0)
    let channels = input_shape.load(1)
    let input_width = input_shape.load(2)
    let input_height = input_shape.load(3)

    let kernel_width = kernel_shape.load(1)
    let kernel_height = kernel_shape.load(2)

    let output_width = output_shape.load(2)
    let output_height = output_shape.load(3)

    let im2col = DTypePointer[DType.float32].alloc(
        batches * output_width * output_height * kernel_width * kernel_height * channels
    )

    DTypePointer[DType.float32].prefetch[PREFETCH_READ](input)
    DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](im2col)

    for batch in range(batches):
        for channel in range(channels):

            @parameter
            fn workgroup_function[NELTS: Int](output_y: Int):
                for output_x in range(output_width):
                    let base_index = batch * output_width * output_height * kernel_width * kernel_height * channels + output_y * output_width * kernel_width * kernel_height * channels + output_x * kernel_width * kernel_height * channels + channel
                    for kernel_y in range(kernel_height):
                        let input_y = output_y * stride_y + kernel_y - padding_y
                        let y_index = base_index + kernel_y * kernel_width * channels
                        if input_y < 0 or input_y >= input_height:

                            @parameter
                            fn fw_vec_zero[NELTS: Int](kernel_x: Int):
                                im2col.simd_store[NELTS](
                                    y_index + kernel_x * channels, 0.0
                                )

                            vectorize[NELTS, fw_vec_zero](kernel_width)
                        else:

                            @parameter
                            fn fw_vec_one[NELTS: Int](kernel_x: Int):
                                let input_x = output_x * stride_x + kernel_x - padding_x
                                if input_x < 0 or input_x >= input_width:
                                    im2col.simd_store[NELTS](
                                        y_index + kernel_x * channels, 0.0
                                    )
                                else:
                                    let input_index = batch * input_width * input_height * channels + input_y * input_width * channels + input_x * channels
                                    im2col.simd_store[NELTS](
                                        y_index + kernel_x * channels,
                                        input.simd_load[NELTS](input_index),
                                    )

                            vectorize[NELTS, fw_vec_one](kernel_width)

            tile[workgroup_function, tile_sizes](0, output_height)

    return im2col
