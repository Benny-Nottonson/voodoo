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

alias nelts = simdwidthof[DType.float32]()
alias epsilon = Float32(1e-8)


fn fw_elu(node: Node, parent1: Node):
    @parameter
    fn v_elu[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(0.0)).cast[DType.float32]()
            * parent1.load_data[_nelts](i)
            + (parent1.load_data[_nelts](i) <= Float32(0.0)).cast[DType.float32]()
            * (exp(parent1.load_data[_nelts](i)) - Float32(1.0)),
        )

    vectorize[nelts, v_elu](node.load_cap())


fn bw_elu(node: Node, parent1: Node):
    @parameter
    fn v_elu_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(0.0)).cast[DType.float32]()
            * node.load_grad[_nelts](i)
            + (parent1.load_data[_nelts](i) <= Float32(0.0)).cast[DType.float32]()
            * (node.load_grad[_nelts](i) + node.load_data[_nelts](i)),
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
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * node.load_data[_nelts](i),
        )

    vectorize[nelts, v_exp_bw](node.load_cap())


fn fw_gelu(node: Node, parent1: Node):
    @parameter
    fn v_gelu[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            Float32(0.5)
            * parent1.load_data[_nelts](i)
            * (
                Float32(1.0)
                + tanh(
                    Float32(0.7978845608028654)
                    * (
                        parent1.load_data[_nelts](i)
                        + Float32(0.044715) * pow(parent1.load_data[_nelts](i), 3)
                    )
                )
            ),
        )

    vectorize[nelts, v_gelu](node.load_cap())


fn bw_gelu(node: Node, parent1: Node):
    @parameter
    fn v_gelu_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (
                Float32(0.5)
                * (
                    Float32(1.0)
                    + tanh(
                        Float32(0.7978845608028654)
                        * (
                            parent1.load_data[_nelts](i)
                            + Float32(0.044715) * pow(parent1.load_data[_nelts](i), 3)
                        )
                    )
                )
                + Float32(0.3989422804014327)
                * exp(
                    Float32(-0.7978845608028654)
                    * (
                        parent1.load_data[_nelts](i)
                        + Float32(0.044715) * pow(parent1.load_data[_nelts](i), 3)
                    )
                )
                * (
                    Float32(0.5)
                    * (
                        Float32(1.0)
                        + tanh(
                            Float32(0.7978845608028654)
                            * (
                                parent1.load_data[_nelts](i)
                                + Float32(0.044715)
                                * pow(parent1.load_data[_nelts](i), 3)
                            )
                        )
                    )
                    + Float32(0.7978845608028654)
                    * Float32(0.1341640786499874)
                    * pow(
                        Float32(1.0)
                        + tanh(
                            Float32(0.7978845608028654)
                            * (
                                parent1.load_data[_nelts](i)
                                + Float32(0.044715)
                                * pow(parent1.load_data[_nelts](i), 3)
                            )
                        ),
                        2,
                    )
                    * (
                        parent1.load_data[_nelts](i)
                        + Float32(0.044715) * pow(parent1.load_data[_nelts](i), 3)
                    )
                )
            ),
        )

    vectorize[nelts, v_gelu_bw](node.load_cap())


fn fw_hard_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_hard_sigmoid[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(2.5)).cast[DType.float32]()
            + (parent1.load_data[_nelts](i) <= Float32(-2.5)).cast[DType.float32]()
            * Float32(0.0)
            + (parent1.load_data[_nelts](i) > Float32(-2.5)).cast[DType.float32]()
            * (parent1.load_data[_nelts](i) < Float32(2.5)).cast[DType.float32]()
            * (Float32(0.2) * parent1.load_data[_nelts](i) + Float32(0.5)),
        )

    vectorize[nelts, v_hard_sigmoid](node.load_cap())


fn bw_hard_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_hard_sigmoid_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(2.5)).cast[DType.float32]()
            + (parent1.load_data[_nelts](i) <= Float32(-2.5)).cast[DType.float32]()
            * Float32(0.0)
            + (parent1.load_data[_nelts](i) > Float32(-2.5)).cast[DType.float32]()
            * (parent1.load_data[_nelts](i) < Float32(2.5)).cast[DType.float32]()
            * Float32(0.2),
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
        node.store_data[_nelts](
            i,
            parent1.load_data[_nelts](i)
            * tanh(log(Float32(1.0) + exp(parent1.load_data[_nelts](i)))),
        )

    vectorize[nelts, v_mish](node.load_cap())


fn bw_mish(node: Node, parent1: Node):
    @parameter
    fn v_mish_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (
                tanh(log(Float32(1.0) + exp(parent1.load_data[_nelts](i))))
                + parent1.load_data[_nelts](i)
                * (
                    Float32(1.0)
                    - pow(
                        tanh(log(Float32(1.0) + exp(parent1.load_data[_nelts](i)))), 2
                    )
                )
            ),
        )

    vectorize[nelts, v_mish_bw](node.load_cap())


fn fw_relu(node: Node, parent1: Node):
    @parameter
    fn v_relu[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(0.0)).cast[DType.float32]()
            * parent1.load_data[_nelts](i),
        )

    vectorize[nelts, v_relu](node.load_cap())


fn bw_relu(node: Node, parent1: Node):
    @parameter
    fn v_relu_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(0.0)).cast[DType.float32]()
            * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_relu_bw](node.load_cap())


fn fw_selu(node: Node, parent1: Node):
    @parameter
    fn v_selu[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(0.0)).cast[DType.float32]()
            * Float32(1.05070098)
            * parent1.load_data[_nelts](i)
            + (parent1.load_data[_nelts](i) <= Float32(0.0)).cast[DType.float32]()
            * Float32(1.05070098)
            * Float32(1.67326324)
            * (exp(parent1.load_data[_nelts](i)) - Float32(1.0)),
        )

    vectorize[nelts, v_selu](node.load_cap())


fn bw_selu(node: Node, parent1: Node):
    @parameter
    fn v_selu_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            (parent1.load_data[_nelts](i) > Float32(0.0)).cast[DType.float32]()
            * Float32(1.05070098)
            * node.load_grad[_nelts](i)
            + (parent1.load_data[_nelts](i) <= Float32(0.0)).cast[DType.float32]()
            * Float32(1.05070098)
            * Float32(1.67326324)
            * node.load_grad[_nelts](i)
            * exp(parent1.load_data[_nelts](i)),
        )

    vectorize[nelts, v_selu_bw](node.load_cap())


fn fw_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_sigmoid[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i, Float32(1.0) / (Float32(1.0) + exp(-parent1.load_data[_nelts](i)))
        )

    vectorize[nelts, v_sigmoid](node.load_cap())


fn bw_sigmoid(node: Node, parent1: Node):
    @parameter
    fn v_sigmoid_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * node.load_data[_nelts](i)
            * (Float32(1.0) - node.load_data[_nelts](i)),
        )

    vectorize[nelts, v_sigmoid_bw](node.load_cap())


fn fw_softmax(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // N):
        let offset = s * N
        var max_el = Float32(0.0)

        @parameter
        fn v_max[nelts: Int](i: Int):
            max_el = max(max_el, parent1.load_data[nelts](offset + i).reduce_max())

        vectorize[nelts, v_max](N)
        var sum = Float32(0.0)

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
            var grad = Float32(0.0)
            var grad2 = Float32(0.0)

            @parameter
            fn v_softmax_bw[nelts: Int](i: Int):
                if i == j:
                    let temp = node.load_data[nelts](offset + j)
                    grad += (
                        node.load_grad[nelts](offset + i)
                        * (temp * (Float32(1.0) - temp))
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
        node.store_data[_nelts](
            i, log(Float32(1.0) + exp(parent1.load_data[_nelts](i)))
        )

    vectorize[nelts, v_softplus](node.load_cap())


fn bw_softplus(node: Node, parent1: Node):
    @parameter
    fn v_softplus_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (Float32(1.0) - exp(-parent1.load_data[_nelts](i))),
        )

    vectorize[nelts, v_softplus_bw](node.load_cap())


fn fw_softsign(node: Node, parent1: Node):
    @parameter
    fn v_softsign[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            parent1.load_data[_nelts](i)
            / (Float32(1.0) + abs(parent1.load_data[_nelts](i))),
        )

    vectorize[nelts, v_softsign](node.load_cap())


fn bw_softsign(node: Node, parent1: Node):
    @parameter
    fn v_softsign_bw[_nelts: Int](i: Int):
        let ones = SIMD[DType.float32, _nelts]()
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            / ((Float32(1.0) + abs(parent1.load_data[_nelts](i))) ** 2),
        )

    vectorize[nelts, v_softsign_bw](node.load_cap())


fn fw_swish(node: Node, parent1: Node):
    @parameter
    fn v_swish[_nelts: Int](i: Int):
        node.store_data[_nelts](
            i,
            parent1.load_data[_nelts](i)
            / (Float32(1.0) + exp(-parent1.load_data[_nelts](i))),
        )

    vectorize[nelts, v_swish](node.load_cap())


fn bw_swish(node: Node, parent1: Node):
    @parameter
    fn v_swish_bw[_nelts: Int](i: Int):
        node.store_grad[_nelts](
            i,
            node.load_grad[_nelts](i)
            * (
                Float32(1.0)
                + (Float32(1.0) - node.load_data[_nelts](i))
                * exp(-parent1.load_data[_nelts](i))
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
            + node.load_grad[_nelts](i)
            * (Float32(1.0) - pow(node.load_data[_nelts](i), 2)),
        )

    vectorize[nelts, v_tanh_bw](node.load_cap())
