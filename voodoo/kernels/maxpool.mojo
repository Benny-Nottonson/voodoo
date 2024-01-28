from algorithm import vectorize
from math import max
from voodoo import Node
from ..constants import PREFETCH_READ, PREFETCH_WRITE, F32_MAX, NELTS


struct MaxPool1D:
    @staticmethod
    fn fw(c: Node, a: Node):
        let params = c.other_params

        let kernel_width = params.load(0)
        let stride = params.load(1)
        let padding = params.load(2)

        let input_shape = a.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)

        let output_width = output_shape.load(2)

        DTypePointer.prefetch[PREFETCH_READ](a.data.load())
        DTypePointer.prefetch[PREFETCH_WRITE](c.data.load())

        for batch in range(batches):
            let batch_offset = batch * channels * input_width
            let output_batch_offset = batch * channels * output_width
            for channel in range(channels):
                let channel_offset = channel * input_width
                let output_channel_offset = channel * output_width
                for output_pos in range(output_width):
                    let input_pos = output_pos * stride - padding
                    var max_value = -F32_MAX

                    @parameter
                    @always_inline("nodebug")
                    fn fw_vec[NELTS: Int](kernel_pos: Int):
                        let input_index = channel_offset + input_pos + kernel_pos
                        if input_index >= 0 and input_index < input_width:
                            let value = a.load_data[NELTS](batch_offset + input_index)
                            max_value = max(max_value, value.reduce_max())

                    vectorize[NELTS, fw_vec](kernel_width)
                    c.store_data(
                        output_batch_offset + output_channel_offset + output_pos,
                        max_value,
                    )

    @staticmethod
    fn bw(c: Node, a: Node):
        let params = c.other_params

        let kernel_width = params.load(0)
        let stride = params.load(1)
        let padding = params.load(2)

        let input_shape = a.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_width = input_shape.load(2)

        let output_width = output_shape.load(2)

        DTypePointer.prefetch[PREFETCH_READ](a.data.load())
        DTypePointer.prefetch[PREFETCH_READ](c.data.load())
        DTypePointer.prefetch[PREFETCH_READ](c.data.load(1))
        DTypePointer.prefetch[PREFETCH_WRITE](a.data.load(1))

        for batch in range(batches):
            let batch_offset = batch * channels * input_width
            let output_batch_offset = batch * channels * output_width
            for channel in range(channels):
                let channel_offset = channel * input_width
                let output_channel_offset = channel * output_width
                for output_pos in range(output_width):
                    let input_pos = output_pos * stride - padding
                    let output_index = output_batch_offset + output_channel_offset + output_pos
                    let max_value = c.load_data(output_index)

                    @parameter
                    @always_inline("nodebug")
                    fn bw_vec[NELTS: Int](kernel_pos: Int):
                        let input_index = channel_offset + input_pos + kernel_pos
                        if input_index >= 0 and input_index < input_width:
                            let value = a.load_data[NELTS](batch_offset + input_index)
                            let grad = c.load_grad[NELTS](output_index)
                            let grad_value = (value == max_value).select(grad, 0)
                            a.store_grad[NELTS](batch_offset + input_index, grad_value)

                    vectorize[NELTS, bw_vec](kernel_width)

                    let grad = c.load_grad(output_index)
                    a.store_grad(batch_offset + input_pos, grad.reduce_add())


struct MaxPool2D:
    @staticmethod
    fn fw(c: Node, a: Node):
        let params = c.other_params

        let kernel_width = params.load(0)
        let kernel_height = params.load(1)
        let stride = params.load(2)
        let padding = params.load(3)

        let input_shape = a.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_height = input_shape.load(2)
        let input_width = input_shape.load(3)

        let output_height = output_shape.load(2)
        let output_width = output_shape.load(3)

        DTypePointer.prefetch[PREFETCH_READ](a.data.load())
        DTypePointer.prefetch[PREFETCH_WRITE](c.data.load())

        for batch in range(batches):
            let batch_offset = batch * channels * input_height * input_width
            let output_batch_offset = batch * channels * output_height * output_width
            for channel in range(channels):
                let channel_offset = channel * input_height * input_width
                let output_channel_offset = channel * output_height * output_width
                for output_y in range(output_height):
                    let input_y = output_y * stride - padding
                    for output_x in range(output_width):
                        let input_x = output_x * stride - padding
                        var max_value = -F32_MAX

                        for kernel_y in range(kernel_height):

                            @parameter
                            @always_inline("nodebug")
                            fn fw_vec[NELTS: Int](kernel_x: Int):
                                let input_index = channel_offset + input_y + kernel_y * input_width + input_x + kernel_x
                                if (
                                    input_index >= 0
                                    and input_index < input_height * input_width
                                ):
                                    let value = a.load_data[NELTS](
                                        batch_offset + input_index
                                    )
                                    max_value = max(max_value, value.reduce_max())

                            vectorize[NELTS, fw_vec](kernel_width)
                        c.store_data(
                            output_batch_offset
                            + output_channel_offset
                            + output_y * output_width
                            + output_x,
                            max_value,
                        )

    @staticmethod
    fn bw(c: Node, a: Node):
        let params = c.other_params

        let kernel_width = params.load(0)
        let kernel_height = params.load(1)
        let stride = params.load(2)
        let padding = params.load(3)

        let input_shape = a.shape
        let output_shape = c.shape

        let batches = input_shape.load(0)
        let channels = input_shape.load(1)
        let input_height = input_shape.load(2)
        let input_width = input_shape.load(3)

        let output_height = output_shape.load(2)
        let output_width = output_shape.load(3)

        DTypePointer.prefetch[PREFETCH_READ](a.data.load())
        DTypePointer.prefetch[PREFETCH_READ](c.data.load())
        DTypePointer.prefetch[PREFETCH_READ](c.data.load(1))
        DTypePointer.prefetch[PREFETCH_WRITE](a.data.load(1))

        for batch in range(batches):
            let batch_offset = batch * channels * input_height * input_width
            let output_batch_offset = batch * channels * output_height * output_width
            for channel in range(channels):
                let channel_offset = channel * input_height * input_width
                let output_channel_offset = channel * output_height * output_width
                for output_y in range(output_height):
                    let input_y = output_y * stride - padding
                    for output_x in range(output_width):
                        let input_x = output_x * stride - padding
                        let output_index = (
                            output_batch_offset
                            + output_channel_offset
                            + output_y * output_width
                            + output_x
                        )
                        let max_value = c.load_data(output_index)

                        for kernel_y in range(kernel_height):

                            @parameter
                            @always_inline("nodebug")
                            fn bw_vec[NELTS: Int](kernel_x: Int):
                                let input_index = channel_offset + input_y + kernel_y * input_width + input_x + kernel_x
                                if (
                                    input_index >= 0
                                    and input_index < input_height * input_width
                                ):
                                    let value = a.load_data[NELTS](
                                        batch_offset + input_index
                                    )
                                    let grad = c.load_grad[NELTS](output_index)
                                    let grad_value = (value == max_value).select(
                                        grad, 0
                                    )
                                    a.store_grad[NELTS](
                                        batch_offset + input_index, grad_value
                                    )

                            vectorize[NELTS, bw_vec](kernel_width)

                        let grad = c.load_grad(output_index)
                        a.store_grad(
                            batch_offset + input_y * input_width + input_x,
                            grad.reduce_add(),
                        )
