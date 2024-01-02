from random import random_float64
from sys.param_env import env_get_int
from algorithm import vectorize, parallelize

from voodoo import Node

alias DType_F32 = DType.float32
alias nelts = simdwidthof[DType_F32]()
alias workers = env_get_int["WORKERS", 0]()


fn fw_copy(node: Node, parent1: Node):
    @parameter
    fn v_copy[_nelts: Int](i: Int):
        node.store_data[_nelts](i, parent1.load_data[_nelts](i))

    vectorize[nelts, v_copy](node.load_cap())


fn bw_copy(node: Node, parent1: Node):
    @parameter
    fn v_copy_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](i, parent1.load_grad[_nelts](i))

    vectorize[nelts, v_copy_bw](node.load_cap())


fn fw_sum(node: Node, parent1: Node):
    var sum: Float32 = 0.0

    @parameter
    fn v_sum[nelts: Int](i: Int):
        sum += parent1.load_data[nelts](i).reduce_add()

    vectorize[nelts, v_sum](parent1.load_cap())
    node.store_data(0, sum)


fn bw_sum(node: Node, parent1: Node):
    @parameter
    fn v_sum_bw[nelts: Int](i: Int):
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + node.load_grad(0))

    vectorize[nelts, v_sum_bw](parent1.load_cap())


fn fw_reshape(node: Node, parent1: Node):
    for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
        let offset = s * parent1.cap_ptr.load()

        @parameter
        fn v_reshape[nelts: Int](i: Int):
            node.store_data[nelts](i, parent1.load_data[nelts](i))

        vectorize[nelts, v_reshape](parent1.cap_ptr.load())


fn bw_reshape(node: Node, parent1: Node):
    for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
        let offset = s * parent1.cap_ptr.load()

        @parameter
        fn v_reshape[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i, parent1.load_grad[nelts](i) + node.load_grad[nelts](i)
            )

        vectorize[nelts, v_reshape](parent1.cap_ptr.load())


fn fw_transp(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let M = parent1.shape_ptr.load().load(num_dims - 2)
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transp[nelts: Int](j: Int):
                node.store_data[nelts](
                    offset + j * M + i, parent1.load_data[nelts](offset + i * N + j)
                )

            vectorize[nelts, v_transp](N)


fn bw_transp(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let M = parent1.shape_ptr.load().load(num_dims - 2)
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transp_bw[nelts: Int](j: Int):
                parent1.store_grad[nelts](
                    offset + j * M + i,
                    parent1.load_grad[nelts](offset + j * M + i)
                    + node.load_grad[nelts](offset + i * N + j),
                )

            vectorize[nelts, v_transp_bw](N)


fn index(
    n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
) -> Int:
    return n * (num_channels * height * width) + c * (height * width) + h * width + w


@always_inline
fn conv_2d(c: Node, a: Node, b: Node):
    let padding = c.other_params_ptr.load().load(0)
    let stride = c.other_params_ptr.load().load(1)

    @parameter
    fn batch_loop(i: Int):
        for j in range(b.shape_ptr.load().load(0)):
            for x in range(c.shape_ptr.load().load(2)):
                for y in range(c.shape_ptr.load().load(3)):
                    var patch_sum: Float32 = 0.0
                    for k in range(a.shape_ptr.load().load(1)):
                        for dx in range(b.shape_ptr.load().load(2)):

                            @parameter
                            fn inner_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    let a_index = index(
                                        i,
                                        k,
                                        ix,
                                        iy,
                                        a.shape_ptr.load().load(1),
                                        a.shape_ptr.load().load(2),
                                        a.shape_ptr.load().load(3),
                                    )
                                    let b_index = index(
                                        j,
                                        k,
                                        dx,
                                        dy,
                                        a.shape_ptr.load().load(1),
                                        b.shape_ptr.load().load(2),
                                        b.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        a.load_data[_nelts](a_index)
                                        * b.load_data[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, inner_loop](b.shape_ptr.load().load(3))
                    let c_index = index(
                        i,
                        j,
                        x,
                        y,
                        b.shape_ptr.load().load(0),
                        c.shape_ptr.load().load(2),
                        c.shape_ptr.load().load(3),
                    )
                    c.store_data(c_index, patch_sum)

    parallelize[batch_loop](
        a.shape_ptr.load().load(0),
        workers if workers > 0 else a.shape_ptr.load().load(0),
    )


fn bw_conv_2d(c: Node, a: Node, b: Node):
    let padding = c.other_params_ptr.load().load(0)
    let stride = c.other_params_ptr.load().load(1)

    for i in range(a.shape_ptr.load().load(1)):
        for j in range(b.shape_ptr.load().load(0)):
            for x in range(b.shape_ptr.load().load(2)):
                for y in range(b.shape_ptr.load().load(3)):
                    var patch_sum: Float32 = 0.0
                    for b in range(a.shape_ptr.load().load(0)):
                        for dx in range(c.shape_ptr.load().load(2)):
                            for dy in range(c.shape_ptr.load().load(3)):
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    let a_index = index(
                                        b,
                                        i,
                                        ix,
                                        iy,
                                        a.shape_ptr.load().load(1),
                                        a.shape_ptr.load().load(2),
                                        a.shape_ptr.load().load(3),
                                    )
                                    let c_grad_index = index(
                                        b,
                                        j,
                                        dx,
                                        dy,
                                        c.shape_ptr.load().load(1),
                                        c.shape_ptr.load().load(2),
                                        c.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        a.load_data(a_index) * c.load_grad(c_grad_index)
                                    ).reduce_add()
                    let b_grad_index = index(
                        i,
                        j,
                        x,
                        y,
                        b.shape_ptr.load().load(0),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    b.store_grad(b_grad_index, patch_sum)

    @parameter
    fn batch_loop(p: Int):
        for j in range(a.shape_ptr.load().load(1)):
            for i in range(b.shape_ptr.load().load(0)):
                for x in range(a.shape_ptr.load().load(2)):
                    for y in range(a.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        for dx in range(b.shape_ptr.load().load(2)):

                            @parameter
                            fn dy_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - dx + padding
                                let iy = y * stride - dy + padding
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= c.shape_ptr.load().load(2)
                                    or iy >= c.shape_ptr.load().load(3)
                                ):
                                    let c_grad_index = index(
                                        p,
                                        i,
                                        ix,
                                        iy,
                                        c.shape_ptr.load().load(1),
                                        c.shape_ptr.load().load(2),
                                        c.shape_ptr.load().load(3),
                                    )
                                    let b_index = index(
                                        i,
                                        j,
                                        b.shape_ptr.load().load(2) - dx - 1,
                                        b.shape_ptr.load().load(3) - dy - 1,
                                        b.shape_ptr.load().load(1),
                                        b.shape_ptr.load().load(2),
                                        b.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        c.load_grad[_nelts](c_grad_index)
                                        * c.load_data[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, dy_loop](b.shape_ptr.load().load(3))
                        let a_grad_index = index(
                            p,
                            j,
                            x,
                            y,
                            a.shape_ptr.load().load(1),
                            a.shape_ptr.load().load(2),
                            a.shape_ptr.load().load(3),
                        )
                        a.store_grad(
                            a_grad_index, a.load_grad(a_grad_index) + patch_sum
                        )

    parallelize[batch_loop](
        a.shape_ptr.load().load(0),
        workers if workers > 0 else a.shape_ptr.load().load(0),
    )


fn max_pool_2d(b: Node, a: Node):
    let padding = b.other_params_ptr.load().load(0)
    let stride = b.other_params_ptr.load().load(1)
    let kernel_width = b.other_params_ptr.load().load(2)
    let kernel_height = b.other_params_ptr.load().load(3)

    for p in range(a.shape_ptr.load().load(0)):
        for i in range(a.shape_ptr.load().load(1)):
            for x in range(
                0, a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding, stride
            ):
                for y in range(
                    0,
                    a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                    stride,
                ):
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= a.shape_ptr.load().load(2)
                                or iy >= a.shape_ptr.load().load(3)
                            ):
                                continue
                            let idx = index(
                                p,
                                i,
                                ix,
                                iy,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            let entry = a.load_data(idx)
                            if entry > max_val:
                                max_val = entry
                                arg_max = idx
                    let idx = index(
                        p,
                        i,
                        (x) // stride,
                        (y) // stride,
                        b.shape_ptr.load().load(1),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    b.store_data(idx, max_val)


fn bw_max_pool_2d(b: Node, a: Node):
    let padding = b.other_params_ptr.load().load(0)
    let stride = b.other_params_ptr.load().load(1)
    let kernel_width = b.other_params_ptr.load().load(2)
    let kernel_height = b.other_params_ptr.load().load(3)

    for p in range(a.shape_ptr.load().load(0)):
        for i in range(a.shape_ptr.load().load(1)):
            for x in range(
                0, a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding, stride
            ):
                for y in range(
                    0,
                    a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                    stride,
                ):
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= a.shape_ptr.load().load(2)
                                or iy >= a.shape_ptr.load().load(3)
                            ):
                                continue
                            let idx = index(
                                p,
                                i,
                                ix,
                                iy,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            let entry = a.load_data(idx)
                            if entry > max_val:
                                max_val = entry
                                arg_max = idx
                    let b_grad_idx = index(
                        p,
                        i,
                        (x) // stride,
                        (y) // stride,
                        b.shape_ptr.load().load(1),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    a.store_grad(
                        arg_max, a.load_grad(arg_max) + b.load_grad(b_grad_idx)
                    )


fn fw_dropout(node: Node, parent1: Node):
    let params = node.other_params_ptr.load()
    let keep_prob = params.load(0) / 1000000.0
    let scale = 1.0 / keep_prob
    # TODO: Implement mask shape

    @parameter
    fn v_dropout[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        for i in range(_nelts):
            let rand = random_float64()
            node.store_data[_nelts](
                i, (rand < keep_prob).cast[DType_F32]() * data * scale
            )

    vectorize[nelts, v_dropout](node.load_cap())


fn bw_dropout(node: Node, parent1: Node):
    let params = node.other_params_ptr.load()
    let keep_prob = params.load(0) / 1000000.0
    let scale = 1.0 / keep_prob

    @parameter
    fn v_dropout_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        for i in range(_nelts):
            let rand = random_float64()
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + (rand < keep_prob).cast[DType_F32]()
                * node.load_grad[_nelts](i)
                * scale,
            )

    vectorize[nelts, v_dropout_bw](node.load_cap())
