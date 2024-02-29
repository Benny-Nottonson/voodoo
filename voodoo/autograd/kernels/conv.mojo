from algorithm import vectorize, tile
from math import max

from voodoo.autograd import Node
from voodoo.utils import Vector
from voodoo.constants import NELTS, PREFETCH_READ, PREFETCH_WRITE

alias tile_sizes = VariadicList[Int](32, 16, 8, 4, 2, 1)


trait Conv:
    ...


@register_passable("trivial")
struct Conv1D(Conv):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        var params = c.get_other_params()

        var padding_x = params[0]
        var stride_x = params[1]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]

        var kernel_width = b.get_shape()[1]

        var output_width = c.get_shape()[2]

        var im2col = im2col2D(
            a.get_data(),
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            padding_x,
            stride_x,
        )

        for batch in range(batches):
            for output_x in range(output_width):
                for kernel_x in range(kernel_width):
                    for channel in range(channels):
                        var kernel_value = b.get_data().load(
                            channel * kernel_width + kernel_x
                        )

                        var output_value = c.get_data().load(
                            batch * output_width * channels
                            + output_x * channels
                            + channel
                        )

                        var im2col_value = im2col.load(
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel
                        )

                        c.get_data().store(
                            batch * output_width * channels
                            + output_x * channels
                            + channel,
                            output_value + kernel_value * im2col_value,
                        )

        im2col.free()

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        var params = c.get_other_params()

        var padding_x = params[0]
        var stride_x = params[1]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]

        var kernel_width = b.get_shape()[1]

        var output_width = c.get_shape()[2]

        var im2col = im2col2D(
            a.get_data(),
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            padding_x,
            stride_x,
        )

        for batch in range(batches):
            for output_x in range(output_width):
                for kernel_x in range(kernel_width):
                    for channel in range(channels):
                        var kernel_value = b.get_data().load(
                            channel * kernel_width + kernel_x
                        )

                        var output_value = c.get_data().load(
                            batch * output_width * channels
                            + output_x * channels
                            + channel
                        )

                        var im2col_value = im2col.load(
                            batch * output_width * kernel_width * channels
                            + output_x * kernel_width * channels
                            + kernel_x * channels
                            + channel
                        )

                        a.get_grad().store(
                            batch * input_width * channels
                            + (output_x * stride_x + kernel_x - padding_x) * channels
                            + channel,
                            a.get_grad().load(
                                batch * input_width * channels
                                + (output_x * stride_x + kernel_x - padding_x)
                                * channels
                                + channel
                            )
                            + kernel_value
                            * c.get_grad().load(
                                batch * output_width * channels
                                + output_x * channels
                                + channel
                            ),
                        )

                        b.get_grad().store(
                            channel * kernel_width + kernel_x,
                            b.get_grad()[channel * kernel_width + kernel_x]
                            + output_value * im2col_value,
                        )

        im2col.free()


@register_passable("trivial")
struct Conv2D(Conv):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        var params = c.get_other_params()

        var padding_x = params[0]
        var padding_y = params[1]
        var stride_x = params[2]
        var stride_y = params[3]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]
        var input_height = a.get_shape()[3]

        var kernel_width = b.get_shape()[1]
        var kernel_height = b.get_shape()[2]

        var output_width = c.get_shape()[2]
        var output_height = c.get_shape()[3]

        var im2col = im2col3D(
            a.get_data(),
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            padding_x,
            padding_y,
            stride_x,
            stride_y,
        )

        var a_data = a.get_data()
        var b_data = b.get_data()
        var c_data = c.get_data()

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
                                var kernel_value = b_data.simd_load[NELTS](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                var output_value = c_data.simd_load[NELTS](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                var im2col_value = im2col.simd_load[NELTS](
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

                        vectorize[fw_vec, NELTS](kernel_width)

        im2col.free()

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        var params = c.get_other_params()

        var padding_x = params[0]
        var padding_y = params[1]
        var stride_x = params[2]
        var stride_y = params[3]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]
        var input_height = a.get_shape()[3]

        var kernel_width = b.get_shape()[1]
        var kernel_height = b.get_shape()[2]

        var output_width = c.get_shape()[2]
        var output_height = c.get_shape()[3]

        var im2col = im2col3D(
            a.get_data(),
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            padding_x,
            padding_y,
            stride_x,
            stride_y,
        )

        var b_data = b.get_data()
        var c_data = c.get_data()
        var a_grad = a.get_grad()
        var b_grad = b.get_grad()
        var c_grad = c.get_grad()

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
                                var kernel_value = b_data.simd_load[NELTS](
                                    channel * kernel_width * kernel_height
                                    + kernel_y * kernel_width
                                    + kernel_x
                                )

                                var output_value = c_data.simd_load[NELTS](
                                    batch * output_width * output_height * channels
                                    + output_y * output_width * channels
                                    + output_x * channels
                                    + channel
                                )

                                var im2col_value = im2col.simd_load[NELTS](
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
    var batches = input_shape[0]
    var channels = input_shape[1]
    var input_width = input_shape[2]

    var kernel_width = kernel_shape[1]

    var output_width = output_shape[2]

    var im2col = DTypePointer[DType.float32].alloc(
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
                    var input_x = output_x * stride + kernel_x - padding

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

                vectorize[fw_vec, NELTS](kernel_width)

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
    var batches = input_shape[0]
    var channels = input_shape[1]
    var input_width = input_shape[2]
    var input_height = input_shape[3]

    var kernel_width = kernel_shape[1]
    var kernel_height = kernel_shape[2]

    var output_width = output_shape[2]
    var output_height = output_shape[3]

    var im2col = DTypePointer[DType.float32].alloc(
        batches * output_width * output_height * kernel_width * kernel_height * channels
    )

    DTypePointer[DType.float32].prefetch[PREFETCH_READ](input)
    DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](im2col)

    for batch in range(batches):
        for channel in range(channels):

            @parameter
            fn workgroup_function[NELTS: Int](output_y: Int):
                for output_x in range(output_width):
                    var base_index = batch * output_width * output_height * kernel_width * kernel_height * channels + output_y * output_width * kernel_width * kernel_height * channels + output_x * kernel_width * kernel_height * channels + channel
                    for kernel_y in range(kernel_height):
                        var input_y = output_y * stride_y + kernel_y - padding_y
                        var y_index = base_index + kernel_y * kernel_width * channels
                        if input_y < 0 or input_y >= input_height:

                            @parameter
                            fn fw_vec_zero[NELTS: Int](kernel_x: Int):
                                im2col.simd_store[NELTS](
                                    y_index + kernel_x * channels, 0.0
                                )

                            vectorize[fw_vec_zero, NELTS](kernel_width)
                        else:

                            @parameter
                            fn fw_vec_one[NELTS: Int](kernel_x: Int):
                                var input_x = output_x * stride_x + kernel_x - padding_x
                                if input_x < 0 or input_x >= input_width:
                                    im2col.simd_store[NELTS](
                                        y_index + kernel_x * channels, 0.0
                                    )
                                else:
                                    var input_index = batch * input_width * input_height * channels + input_y * input_width * channels + input_x * channels
                                    im2col.simd_store[NELTS](
                                        y_index + kernel_x * channels,
                                        input.simd_load[NELTS](input_index),
                                    )

                            vectorize[fw_vec_one, NELTS](kernel_width)

            tile[workgroup_function, tile_sizes](0, output_height)

    return im2col
