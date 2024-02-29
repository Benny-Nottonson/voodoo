from algorithm import vectorize
from math import max

from voodoo.autograd import Node
from voodoo.constants import PREFETCH_READ, PREFETCH_WRITE, F32_MAX, NELTS


trait MaxPool:
    ...


@register_passable("trivial")
struct MaxPool1D(MaxPool):
    @staticmethod
    fn fw(c: Node, a: Node):
        var params = c.get_other_params()

        var kernel_width = params[0]
        var stride = params[1]
        var padding = params[2]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]

        var output_width = c.get_shape()[2]

        DTypePointer.prefetch[PREFETCH_READ](a.get_data())
        DTypePointer.prefetch[PREFETCH_WRITE](c.get_data())

        for batch in range(batches):
            var batch_offset = batch * channels * input_width
            var output_batch_offset = batch * channels * output_width
            for channel in range(channels):
                var channel_offset = channel * input_width
                var output_channel_offset = channel * output_width
                for output_pos in range(output_width):
                    var input_pos = output_pos * stride - padding
                    var max_value = -F32_MAX

                    @parameter
                    fn fw_vec[NELTS: Int](kernel_pos: Int):
                        var input_index = channel_offset + input_pos + kernel_pos
                        if input_index >= 0 and input_index < input_width:
                            var value = a.get_data().simd_load[NELTS](
                                batch_offset + input_index
                            )
                            max_value = max(max_value, value.reduce_max())

                    vectorize[fw_vec, NELTS](kernel_width)
                    c.get_data().store(
                        output_batch_offset + output_channel_offset + output_pos,
                        max_value,
                    )

    @staticmethod
    fn bw(c: Node, a: Node):
        var params = c.get_other_params()

        var kernel_width = params[0]
        var stride = params[1]
        var padding = params[2]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_width = a.get_shape()[2]

        var output_width = c.get_shape()[2]

        DTypePointer.prefetch[PREFETCH_READ](a.get_data())
        DTypePointer.prefetch[PREFETCH_READ](c.get_data())
        DTypePointer.prefetch[PREFETCH_READ](c.get_grad())
        DTypePointer.prefetch[PREFETCH_WRITE](a.get_grad())

        for batch in range(batches):
            var batch_offset = batch * channels * input_width
            var output_batch_offset = batch * channels * output_width
            for channel in range(channels):
                var channel_offset = channel * input_width
                var output_channel_offset = channel * output_width
                for output_pos in range(output_width):
                    var input_pos = output_pos * stride - padding
                    var output_index = output_batch_offset + output_channel_offset + output_pos
                    var max_value = c.get_data()[output_index]

                    @parameter
                    fn bw_vec[NELTS: Int](kernel_pos: Int):
                        var input_index = channel_offset + input_pos + kernel_pos
                        if input_index >= 0 and input_index < input_width:
                            var value = a.get_data().simd_load[NELTS](
                                batch_offset + input_index
                            )
                            var grad = c.get_grad().simd_load[NELTS](output_index)
                            var grad_value = (value == max_value).select(grad, 0)
                            a.get_grad().simd_store[NELTS](
                                batch_offset + input_index, grad_value
                            )

                    vectorize[bw_vec, NELTS](kernel_width)

                    var grad = c.get_grad()[output_index]
                    a.get_grad().store(batch_offset + input_pos, grad.reduce_add())


@register_passable("trivial")
struct MaxPool2D(MaxPool):
    @staticmethod
    fn fw(c: Node, a: Node):
        var params = c.get_other_params()

        var kernel_width = params[0]
        var kernel_height = params[1]
        var stride = params[2]
        var padding = params[3]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_height = a.get_shape()[2]
        var input_width = a.get_shape()[3]

        var output_height = c.get_shape()[2]
        var output_width = c.get_shape()[3]

        DTypePointer.prefetch[PREFETCH_READ](a.get_data())
        DTypePointer.prefetch[PREFETCH_WRITE](c.get_data())

        for batch in range(batches):
            var batch_offset = batch * channels * input_height * input_width
            var output_batch_offset = batch * channels * output_height * output_width
            for channel in range(channels):
                var channel_offset = channel * input_height * input_width
                var output_channel_offset = channel * output_height * output_width
                for output_y in range(output_height):
                    var input_y = output_y * stride - padding
                    for output_x in range(output_width):
                        var input_x = output_x * stride - padding
                        var max_value = -F32_MAX

                        for kernel_y in range(kernel_height):

                            @parameter
                            fn fw_vec[NELTS: Int](kernel_x: Int):
                                var input_index = channel_offset + input_y + kernel_y * input_width + input_x + kernel_x
                                if (
                                    input_index >= 0
                                    and input_index < input_height * input_width
                                ):
                                    var value = a.get_data().simd_load[NELTS](
                                        batch_offset + input_index
                                    )
                                    max_value = max(max_value, value.reduce_max())

                            vectorize[fw_vec, NELTS](kernel_width)
                        c.get_data().store(
                            output_batch_offset
                            + output_channel_offset
                            + output_y * output_width
                            + output_x,
                            max_value,
                        )

    @staticmethod
    fn bw(c: Node, a: Node):
        var params = c.get_other_params()

        var kernel_width = params[0]
        var kernel_height = params[1]
        var stride = params[2]
        var padding = params[3]

        var batches = a.get_shape()[0]
        var channels = a.get_shape()[1]
        var input_height = a.get_shape()[2]
        var input_width = a.get_shape()[3]

        var output_height = c.get_shape()[2]
        var output_width = c.get_shape()[3]

        DTypePointer.prefetch[PREFETCH_READ](a.get_data())
        DTypePointer.prefetch[PREFETCH_READ](c.get_data())
        DTypePointer.prefetch[PREFETCH_READ](c.get_grad())
        DTypePointer.prefetch[PREFETCH_WRITE](a.get_grad())

        for batch in range(batches):
            var batch_offset = batch * channels * input_height * input_width
            var output_batch_offset = batch * channels * output_height * output_width
            for channel in range(channels):
                var channel_offset = channel * input_height * input_width
                var output_channel_offset = channel * output_height * output_width
                for output_y in range(output_height):
                    var input_y = output_y * stride - padding
                    for output_x in range(output_width):
                        var input_x = output_x * stride - padding
                        var output_index = (
                            output_batch_offset
                            + output_channel_offset
                            + output_y * output_width
                            + output_x
                        )
                        var max_value = c.get_data()[output_index]

                        for kernel_y in range(kernel_height):

                            @parameter
                            fn bw_vec[NELTS: Int](kernel_x: Int):
                                var input_index = channel_offset + input_y + kernel_y * input_width + input_x + kernel_x
                                if (
                                    input_index >= 0
                                    and input_index < input_height * input_width
                                ):
                                    var value = a.get_data().simd_load[NELTS](
                                        batch_offset + input_index
                                    )
                                    var grad = c.get_grad().simd_load[NELTS](
                                        output_index
                                    )
                                    var grad_value = (value == max_value).select(
                                        grad, 0
                                    )
                                    a.get_grad().simd_store[NELTS](
                                        batch_offset + input_index, grad_value
                                    )

                            vectorize[bw_vec, NELTS](kernel_width)

                        var grad = c.get_grad()[output_index]
                        a.get_grad().store(
                            batch_offset + input_y * input_width + input_x,
                            grad.reduce_add(),
                        )
