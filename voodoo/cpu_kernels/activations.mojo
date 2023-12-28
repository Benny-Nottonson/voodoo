from math import (
    abs,
    pow,
    exp,
    log,
    tanh,
    erfc,
    max,
)
from algorithm import vectorize

from voodoo import Node

alias DType_F32 = DType.float32
alias nelts = simdwidthof[DType_F32]()
alias epsilon = 1e-8


fn fw_elu(node: Node, parent1: Node):
    @parameter
    fn v_elu[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            (data > 0.0).cast[DType_F32]() * data
            + (data <= 0.0).cast[DType_F32]() * (exp(data) - 1.0),
        )

    vectorize[nelts, v_elu](node.load_cap())


fn bw_elu(node: Node, parent1: Node):
    @parameter
    fn v_elu_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        parent1.store_grad[_nelts](
            i,
            (data > 0.0).cast[DType_F32]() * node.load_grad[_nelts](i)
            + (data <= 0.0).cast[DType_F32]()
            * (node.load_grad[_nelts](i) * (data + 1.0)),
        )

    vectorize[nelts, v_elu_bw](node.load_cap())


fn fw_exp(node: Node, parent1: Node):
    @parameter
    fn v_exp[_nelts: Int](i: Int):
        node.store_data[_nelts](i, exp(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_exp](node.load_cap())


fn bw_exp(node: Node, parent1: Node):
    @parameter
    fn v_exp_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i, parent1.load_grad[_nelts](i) * node.load_data[_nelts](i)
        )

    vectorize[nelts, v_exp_bw](node.load_cap())


fn fw_gelu(node: Node, parent1: Node):
    @parameter
    fn v_gelu[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            0.5
            * data
            * (1.0 + tanh(0.7978845608028654 * (data + 0.044715 * data**3))),
        )

    vectorize[nelts, v_gelu](node.load_cap())


fn bw_gelu(node: Node, parent1: Node):
    @parameter
    fn v_gelu_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (
                0.5 * (1.0 + tanh(0.7978845608028654 * (data + 0.044715 * data**3)))
                + 0.3989422804014327
                * exp(-0.7978845608028654 * (data + 0.044715 * data**3))
                * (
                    0.5
                    * (1.0 + tanh(0.7978845608028654 * (data + 0.044715 * data**3)))
                    + 0.7978845608028654
                    * 0.1341640786499874
                    * (
                        1.0
                        + tanh(0.7978845608028654 * (data + 0.044715 * data**3)) ** 2
                    )
                    * (data + 0.044715 * data**3)
                )
            ),
        )

    vectorize[nelts, v_gelu_bw](node.load_cap())


fn fw_hard_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_hard_sigmoid[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            (data > 2.5).cast[DType_F32]()
            + (data > -2.5).cast[DType_F32]()
            * (data < 2.5).cast[DType_F32]()
            * (0.2 * data + 0.5),
        )

    vectorize[nelts, v_hard_sigmoid](node.load_cap())


fn bw_hard_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_hard_sigmoid_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        parent1.store_grad[_nelts](
            i,
            (data > 2.5).cast[DType_F32]()
            + (data > -2.5).cast[DType_F32]() * (data < 2.5).cast[DType_F32]() * 0.2,
        )

    vectorize[nelts, v_hard_sigmoid_bw](node.load_cap())


fn fw_linear(node: Node, parent1: Node):
    @parameter
    fn v_linear[_nelts: Int](i: Int):
        node.store_data[_nelts](i, parent1.load_data[_nelts](i))

    vectorize[nelts, v_linear](node.load_cap())


fn bw_linear(node: Node, parent1: Node):
    @parameter
    fn v_linear_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i, parent1.load_grad[_nelts](i) + node.load_grad[_nelts](i)
        )

    vectorize[nelts, v_linear_bw](node.load_cap())


fn fw_mish(node: Node, parent1: Node):
    @parameter
    fn v_mish[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            data * tanh(log(1.0 + exp(data))),
        )

    vectorize[nelts, v_mish](node.load_cap())


fn bw_mish(node: Node, parent1: Node):
    @parameter
    fn v_mish_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (
                tanh(log(1.0 + exp(data)))
                + data * (0.0 - pow(tanh(log(0.0 + exp(data))), 2))
            ),
        )

    vectorize[nelts, v_mish_bw](node.load_cap())


fn fw_relu(node: Node, parent1: Node):
    @parameter
    fn v_relu[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            (data > 0.0).cast[DType_F32]() * data,
        )

    vectorize[nelts, v_relu](node.load_cap())


fn bw_relu(node: Node, parent1: Node):
    @parameter
    fn v_relu_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            (parent1.load_data[_nelts](i) > 0.0).cast[DType_F32]()
            * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_relu_bw](node.load_cap())


fn fw_selu(node: Node, parent1: Node):
    @parameter
    fn v_selu[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            (data > 0.0).cast[DType_F32]() * 1.05070098 * data
            + (data <= 0.0).cast[DType_F32]() * 1.75809932607 * (exp(data) - 1.0),
        )

    vectorize[nelts, v_selu](node.load_cap())


fn bw_selu(node: Node, parent1: Node):
    @parameter
    fn v_selu_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_grad[_nelts](
            i,
            (data > 0.0).cast[DType_F32]() * 1.05070098 * node.load_grad[_nelts](i)
            + (data <= 0.0).cast[DType_F32]()
            * 1.75809932607
            * node.load_grad[_nelts](i)
            * exp(data),
        )

    vectorize[nelts, v_selu_bw](node.load_cap())


fn fw_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_sigmoid[_nelts: Int](i: Int):
        node.store_data[_nelts](i, 1.0 / (1.0 + exp(-parent1.load_data[_nelts](i))))

    vectorize[nelts, v_sigmoid](node.load_cap())


fn bw_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_sigmoid_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * node.load_data[_nelts](i)
            * (1.0 - node.load_data[_nelts](i)),
        )

    vectorize[nelts, v_sigmoid_bw](node.load_cap())


fn fw_softmax(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // N):
        let offset = s * N
        var max_el: Float32 = 0.0

        @parameter
        fn v_max[nelts: Int](i: Int):
            max_el = max(max_el, parent1.load_data[nelts](offset + i).reduce_max())

        vectorize[nelts, v_max](N)
        var sum: Float32 = 0.0

        @parameter
        fn v_exp[nelts: Int](i: Int):
            let temp = exp(parent1.load_data[nelts](offset + i) - max_el)
            node.store_data[nelts](offset + i, temp)
            sum += temp.reduce_add()

        vectorize[nelts, v_exp](N)

        @parameter
        fn v_div[nelts: Int](i: Int):
            node.store_data[nelts](offset + i, node.load_data[nelts](offset + i) / sum)

        vectorize[nelts, v_div](N)


fn bw_softmax(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // N):
        let offset = s * N

        @parameter
        fn v_softmax_bw_outer[nelts: Int](j: Int):
            var grad: Float32 = 0.0

            @parameter
            fn v_softmax_bw[nelts: Int](i: Int):
                if i == j:
                    let temp = node.load_data[nelts](offset + j)
                    grad += (
                        node.load_grad[nelts](offset + i) * (temp * (1.0 - temp))
                    ).reduce_add()
                else:
                    grad += (
                        node.load_grad[nelts](offset + i)
                        * node.load_data[nelts](offset + i)
                        * node.load_data[nelts](offset + j)
                        * -1
                    ).reduce_add()

            vectorize[nelts, v_softmax_bw](N)
            parent1.store_grad[nelts](
                offset + j, parent1.load_grad[nelts](offset + j) + grad
            )

        vectorize[nelts, v_softmax_bw_outer](N)


fn fw_softplus(node: Node, parent1: Node):
    @parameter
    fn v_softplus[_nelts: Int](i: Int):
        node.store_data[_nelts](i, log(1.0 + exp(parent1.load_data[_nelts](i))))

    vectorize[nelts, v_softplus](node.load_cap())


fn bw_softplus(node: Node, parent1: Node):
    @parameter
    fn v_softplus_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i) * (1.0 - exp(-parent1.load_data[_nelts](i))),
        )

    vectorize[nelts, v_softplus_bw](node.load_cap())


fn fw_softsign(node: Node, parent1: Node):
    @parameter
    fn v_softsign[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            data / (1.0 + abs(data)),
        )

    vectorize[nelts, v_softsign](node.load_cap())


fn bw_softsign(node: Node, parent1: Node):
    @parameter
    fn v_softsign_bw[_nelts: Int](i: Int):
        let ones = SIMD[DType_F32, _nelts]()
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            / ((1.0 + abs(parent1.load_data[_nelts](i))) ** 2),
        )

    vectorize[nelts, v_softsign_bw](node.load_cap())


fn fw_swish(node: Node, parent1: Node):
    @parameter
    fn v_swish[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            data / (1.0 + exp(-data)),
        )

    vectorize[nelts, v_swish](node.load_cap())


fn bw_swish(node: Node, parent1: Node):
    @parameter
    fn v_swish_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (
                1.0
                + (1.0 - node.load_data[_nelts](i)) * exp(-parent1.load_data[_nelts](i))
            ),
        )

    vectorize[nelts, v_swish_bw](node.load_cap())


fn fw_tanh(node: Node, parent1: Node):
    @parameter
    fn v_tanh[_nelts: Int](i: Int):
        node.store_data[_nelts](i, tanh(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_tanh](node.load_cap())


fn bw_tanh(node: Node, parent1: Node):
    @parameter
    fn v_tanh_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * (1.0 - (node.load_data[_nelts](i)) ** 2),
        )

    vectorize[nelts, v_tanh_bw](node.load_cap())
