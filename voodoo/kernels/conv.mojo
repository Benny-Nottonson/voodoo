from algorithm import vectorize_unroll, tile
from math import max
from voodoo import Node
from ..constants import nelts, prefetch_read, prefetch_write


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

        DTypePointer[DType.float32].prefetch[prefetch_read](a_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](b_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](c_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](im2col)

        for batch in range(batches):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    for kernel_y in range(kernel_height):

                        @parameter
                        fn fw_vec[nelts: Int](kernel_x: Int):
                            for channel in range(channels):
                                let kernel_value = b_data.simd_load[nelts](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let output_value = c_data.simd_load[nelts](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                let im2col_value = im2col.simd_load[nelts](
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

                                c_data.simd_store[nelts](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel,
                                    output_value + kernel_value * im2col_value,
                                )

                        vectorize_unroll[nelts, 1, fw_vec](kernel_width)

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

        DTypePointer[DType.float32].prefetch[prefetch_read](b_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](c_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](im2col)
        DTypePointer[DType.float32].prefetch[prefetch_read](a_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](b_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](c_grad)
        DTypePointer[DType.float32].prefetch[prefetch_write](a_grad)
        DTypePointer[DType.float32].prefetch[prefetch_write](b_grad)

        for batch in range(batches):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    for kernel_y in range(kernel_height):

                        @parameter
                        fn bw_vec[nelts: Int](kernel_x: Int):
                            for channel in range(channels):
                                let kernel_value = b_data.simd_load[nelts](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                let output_value = c_data.simd_load[nelts](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                let im2col_value = im2col.simd_load[nelts](
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

                                a_grad.simd_store[nelts](
                                    batch * input_width * input_height * channels
                                    + (output_y * stride_y + kernel_y - padding_y)
                                    * input_width
                                    * channels
                                    + (output_x * stride_x + kernel_x - padding_x)
                                    * channels
                                    + channel,
                                    a_grad.simd_load[nelts](
                                        batch * input_width * input_height * channels
                                        + (output_y * stride_y + kernel_y - padding_y)
                                        * input_width
                                        * channels
                                        + (output_x * stride_x + kernel_x - padding_x)
                                        * channels
                                        + channel
                                    )
                                    + kernel_value
                                    * c_grad.simd_load[nelts](
                                        batch * output_width * output_height * channels
                                        + output_y * output_width * channels
                                        + output_x * channels
                                        + channel
                                    ),
                                )

                                b_grad.simd_store[nelts](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x,
                                    b_grad.simd_load[nelts](
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

    DTypePointer[DType.float32].prefetch[prefetch_read](input)
    DTypePointer[DType.float32].prefetch[prefetch_write](im2col)

    for batch in range(batches):
        for channel in range(channels):

            @parameter
            fn workgroup_function[nelts: Int](output_x: Int):
                @parameter
                fn fw_vec[nelts: Int](kernel_x: Int):
                    let input_x = output_x * stride + kernel_x - padding

                    if input_x < 0 or input_x >= input_width:
                        im2col.simd_store[nelts](
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel,
                            0.0,
                        )
                    else:
                        im2col.simd_store[nelts](
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel,
                            input.simd_load[nelts](
                                batch * input_width * channels
                                + input_x * channels
                                + channel
                            ),
                        )

                vectorize_unroll[nelts, 1, fw_vec](kernel_width)

            tile[workgroup_function, VariadicList[Int](32, 16, 8, 4, 2, 1)](0, output_width)

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

    DTypePointer[DType.float32].prefetch[prefetch_read](input)
    DTypePointer[DType.float32].prefetch[prefetch_write](im2col)

    for batch in range(batches):
        for channel in range(channels):
            @parameter
            fn workgroup_function[nelts: Int](output_y: Int):
                for output_x in range(output_width):
                    let base_index = batch * output_width * output_height * kernel_width * kernel_height * channels
                                    + output_y * output_width * kernel_width * kernel_height * channels
                                    + output_x * kernel_width * kernel_height * channels
                                    + channel
                    for kernel_y in range(kernel_height):
                        let input_y = output_y * stride_y + kernel_y - padding_y
                        let y_index = base_index + kernel_y * kernel_width * channels
                        if input_y < 0 or input_y >= input_height:
                            @parameter
                            fn fw_vec_zero[nelts: Int](kernel_x: Int):
                                im2col.simd_store[nelts](y_index + kernel_x * channels, 0.0)

                            vectorize_unroll[nelts, 1, fw_vec_zero](kernel_width)
                        else:
                            @parameter
                            fn fw_vec_one[nelts: Int](kernel_x: Int):
                                let input_x = output_x * stride_x + kernel_x - padding_x
                                if input_x < 0 or input_x >= input_width:
                                    im2col.simd_store[nelts](y_index + kernel_x * channels, 0.0)
                                else:
                                    let input_index = batch * input_width * input_height * channels
                                                    + input_y * input_width * channels
                                                    + input_x * channels
                                    im2col.simd_store[nelts](y_index + kernel_x * channels, input.simd_load[nelts](input_index))

                            vectorize_unroll[nelts, 1, fw_vec_one](kernel_width)

            tile[workgroup_function, VariadicList[Int](32, 16, 8, 4, 2, 1)](0, output_height)

    return im2col
