from math import abs, exp, log, tanh, max, erf
from algorithm import vectorize
from voodoo import Node
from .constants import DType_F32, nelts

alias generic_vectorized = fn[_nelts: Int] (SIMD[DType_F32, _nelts]) -> SIMD[
    DType_F32, _nelts
]

# TODO: Rewrite when lambda functions are supported
# TODO: Rewrite leaky relu and softmax to use GenericActivation

struct GenericActivation[fw_vec: generic_vectorized, bw_vec: generic_vectorized]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn generic_vectorized_fw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                fw_vec[_nelts](x),
            )

        vectorize[nelts, generic_vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn generic_vectorized_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * bw_vec[_nelts](x),
            )

        vectorize[nelts, generic_vectorized_bw](node.load_cap())


fn elu_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 0.0).cast[DType_F32]() * x + (x <= 0.0).cast[DType_F32]() * (
        exp(x) - 1.0
    )


fn elu_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[DType_F32]() * exp(x)


alias Elu = GenericActivation[elu_fw_vec, elu_bw_vec]


fn exp_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return exp(x)


fn exp_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return exp(x)


alias Exp = GenericActivation[exp_fw_vec, exp_bw_vec]


fn gelu_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 0.5 * x * (1.0 + erf(x / 1.4142135623730951))


fn gelu_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x / x) + x * 0.56418958354 * exp(-0.5 * (x / 1.4142135623730951) ** 2)


alias Gelu = GenericActivation[gelu_fw_vec, gelu_bw_vec]


fn hard_sigmoid_fw_vec[
    _nelts: Int
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 2.5).cast[DType_F32]() + (x > -2.5).cast[DType_F32]() * (x < 2.5).cast[
        DType_F32
    ]() * (0.2 * x + 0.5)


fn hard_sigmoid_bw_vec[
    _nelts: Int
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > -2.5).cast[DType_F32]() * (x < 2.5).cast[DType_F32]() * 0.2


alias HardSigmoid = GenericActivation[hard_sigmoid_fw_vec, hard_sigmoid_bw_vec]


fn linear_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x


fn linear_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1


alias Linear = GenericActivation[linear_fw_vec, linear_bw_vec]


fn mish_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x * tanh(log(1.0 + exp(x)))


fn mish_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x / x) + x * (1.0 - tanh(log(1.0 + exp(x))) ** 2) * (1.0 / (1.0 + exp(-x)))


alias Mish = GenericActivation[mish_fw_vec, mish_bw_vec]


fn relu_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 0.0).cast[DType_F32]() * x


fn relu_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 0.0).cast[DType_F32]()


alias ReLu = GenericActivation[relu_fw_vec, relu_bw_vec]


fn selu_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 0.0).cast[DType_F32]() * 1.05070098 * x + (x <= 0.0).cast[
        DType_F32
    ]() * 1.05070098 * 1.67326324 * (exp(x) - 1.0)


fn selu_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[
        DType_F32
    ]() * 1.75809932607 * exp(x)


alias Selu = GenericActivation[selu_fw_vec, selu_bw_vec]


fn sigmoid_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 / (1.0 + exp(-x))


fn sigmoid_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return exp(x) / (exp(x) + 1.0) ** 2


alias Sigmoid = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec]


fn softplus_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return log(1.0 + exp(x))


fn softplus_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return exp(x) / (1.0 + exp(x))


alias Softplus = GenericActivation[softplus_fw_vec, softplus_bw_vec]


fn softsign_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x / (1.0 + abs(x))


fn softsign_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x**2 / (1.0 + abs(x)) ** 2


alias Softsign = GenericActivation[softsign_fw_vec, softsign_bw_vec]


fn swish_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return x / (1.0 + exp(-x))


fn swish_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return (x / x) + x * (1.0 - x / (1.0 + exp(-x))) * (1.0 / (1.0 + exp(-x)))


alias Swish = GenericActivation[swish_fw_vec, swish_bw_vec]


fn tanh_fw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return tanh(x)


fn tanh_bw_vec[_nelts: Int](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    return 1.0 - tanh(x) ** 2


alias Tanh = GenericActivation[tanh_fw_vec, tanh_bw_vec]


struct LeakyReLu:
    # f(x) = x > 0 ? x : alpha * x
    # f'(x) = x > 0 ? 1 : alpha
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_leaky_relu[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let alpha = node.other_params_ptr.load().data.load().load() / 1000000.0
            node.store_data[_nelts](
                i,
                (x > 0.0).cast[DType_F32]() * x
                + (x <= 0.0).cast[DType_F32]() * alpha * x,
            )

        vectorize[nelts, vectorized_leaky_relu](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_leaky_relu_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let alpha = node.other_params_ptr.load().data.load().load() / 1000000.0
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * ((x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[DType_F32]() * alpha),
            )

        vectorize[nelts, vectorized_leaky_relu_bw](node.load_cap())


struct Softmax:
    # f(x) = e^wx_i / sum(e^wx_i)
    # f'x(x) = f(x) * (1 - f(x))
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N
            var max_el: Float32 = 0.0

            @parameter
            fn vectorized_max[nelts: Int](i: Int):
                max_el = max(max_el, parent1.load_data[nelts](offset + i).reduce_max())

            vectorize[nelts, vectorized_max](N)
            var sum: Float32 = 0.0

            @parameter
            fn vectorized_exp[nelts: Int](i: Int):
                let temp = exp(parent1.load_data[nelts](offset + i) - max_el)
                node.store_data[nelts](offset + i, temp)
                sum += temp.reduce_add()

            vectorize[nelts, vectorized_exp](N)

            @parameter
            fn vectorized_div[nelts: Int](i: Int):
                node.store_data[nelts](
                    offset + i, node.load_data[nelts](offset + i) / sum
                )

            vectorize[nelts, vectorized_div](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N

            @parameter
            fn vectorized_softmax_bw_outer[nelts: Int](j: Int):
                var grad: Float32 = 0.0

                @parameter
                fn vectorized_softmax_bw[nelts: Int](i: Int):
                    if i == j:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * node.load_data[nelts](offset + i)
                            * (1.0 - node.load_data[nelts](offset + i))
                        ).reduce_add()
                    else:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * node.load_data[nelts](offset + i)
                            * node.load_data[nelts](offset + j)
                            * -1
                        ).reduce_add()

                vectorize[nelts, vectorized_softmax_bw](N)
                parent1.store_grad[nelts](
                    offset + j, parent1.load_grad[nelts](offset + j) + grad
                )

            vectorize[nelts, vectorized_softmax_bw_outer](N)
