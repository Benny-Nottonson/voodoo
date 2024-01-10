from math import abs, exp, log, tanh, max, erf
from algorithm import vectorize
from voodoo import Node
from .constants import DType_F32, nelts, f32_max

alias generic_vectorized = fn[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
] (SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]

# TODO: Rewrite when lambda functions are supported

struct GenericActivation[
    fw_vec: generic_vectorized,
    bw_vec: generic_vectorized,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn generic_vectorized_fw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                fw_vec[_nelts, arg1, arg2, arg3](x),
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
                + node.load_grad[_nelts](i) * bw_vec[_nelts, arg1, arg2, arg3](x),
            )

        vectorize[nelts, generic_vectorized_bw](node.load_cap())


struct Relu[arg1: Float32 = 0.0, arg2: Float32 = f32_max, arg3: Float32 = 0.0]:
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, arg1, arg2, arg3].bw


struct Sigmoid:
    alias fw = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[sigmoid_fw_vec, sigmoid_bw_vec, 0.0, 0.0, 0.0].bw


struct Softmax:
    alias fw = _Softmax.fw
    alias bw = _Softmax.bw


struct Softplus:
    alias fw = GenericActivation[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[softplus_fw_vec, softplus_bw_vec, 0.0, 0.0, 0.0].bw


struct Softsign:
    alias fw = GenericActivation[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[softsign_fw_vec, softsign_bw_vec, 0.0, 0.0, 0.0].bw


struct Tanh:
    alias fw = GenericActivation[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[tanh_fw_vec, tanh_bw_vec, 0.0, 0.0, 0.0].bw


struct Selu:
    alias fw = GenericActivation[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[selu_fw_vec, selu_bw_vec, 0.0, 0.0, 0.0].bw


struct Elu[alpha: Float32 = 1.0]:
    alias fw = GenericActivation[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].fw
    alias bw = GenericActivation[elu_fw_vec, elu_bw_vec, 0.0, 0.0, alpha].bw


struct Exp:
    alias fw = GenericActivation[exp_fw_vec, exp_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[exp_fw_vec, exp_bw_vec, 0.0, 0.0, 0.0].bw


struct LeakyRelu[alpha: Float32 = 0.0]:
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, alpha, f32_max, 0.0].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, alpha, f32_max, 0.0].bw


struct Relu6:
    alias fw = GenericActivation[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].fw
    alias bw = GenericActivation[relu_fw_vec, relu_bw_vec, 0.0, 6.0, 0.0].bw


struct Silu:
    alias fw = GenericActivation[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[silu_fw_vec, silu_bw_vec, 0.0, 0.0, 0.0].bw


alias Swish = Silu


struct Gelu[approximate: Float32 = 0.0]:
    alias fw = GenericActivation[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].fw
    alias bw = GenericActivation[gelu_fw_vec, gelu_bw_vec, approximate, 0.0, 0.0].bw


struct HardSigmoid:
    alias fw = GenericActivation[
        hard_sigmoid_fw_vec, hard_sigmoid_bw_vec, 0.0, 0.0, 0.0
    ].fw
    alias bw = GenericActivation[
        hard_sigmoid_fw_vec, hard_sigmoid_bw_vec, 0.0, 0.0, 0.0
    ].bw


struct Linear:
    alias fw = GenericActivation[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[linear_fw_vec, linear_bw_vec, 0.0, 0.0, 0.0].bw


struct Mish:
    alias fw = GenericActivation[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].fw
    alias bw = GenericActivation[mish_fw_vec, mish_bw_vec, 0.0, 0.0, 0.0].bw


struct LogSoftmax:
    alias fw = _LogSoftmax.fw
    alias bw = _LogSoftmax.bw


fn relu_fw_vec[
    _nelts: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = f32_max,
    threshold: Float32 = 0.0,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x > threshold ? (x > max_value ? max_value : x) : negative_slope * x
    @parameter
    if negative_slope == 0.0 and max_value == f32_max and threshold == 0.0:
        return (x > 0.0).cast[DType_F32]() * x
    return (
        (x > threshold).cast[DType_F32]()
        * (x > max_value).cast[DType_F32]()
        * max_value
        + (x > threshold).cast[DType_F32]() * (x <= max_value).cast[DType_F32]() * x
        + (x <= threshold).cast[DType_F32]() * negative_slope * x
    )


fn relu_bw_vec[
    _nelts: Int,
    negative_slope: Float32 = 0.0,
    max_value: Float32 = f32_max,
    threshold: Float32 = 0.0,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = x > threshold ? (x > max_value ? 0 : 1) : negative_slope
    @parameter
    if negative_slope == 0.0 and max_value == f32_max and threshold == 0.0:
        return (x > 0.0).cast[DType_F32]()
    return (
        (x > threshold).cast[DType_F32]() * (x > max_value).cast[DType_F32]() * 0.0
        + (x > threshold).cast[DType_F32]() * (x <= max_value).cast[DType_F32]() * 1.0
        + (x <= threshold).cast[DType_F32]() * negative_slope
    )


fn sigmoid_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = 1 / (1 + e^-x)
    return 1.0 / (1.0 + exp(-x))


fn sigmoid_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = f(x)(1-f(x))
    # simplifies to e^x / (e^x + 1)^2
    return exp(x) / (exp(x) + 1.0) ** 2


fn softplus_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = log(1 + e^x)
    return log(1.0 + exp(x))


fn softplus_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = e^x / (1 + e^x)
    return exp(x) / (1.0 + exp(x))


fn softsign_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x / (1 + |x|)
    return x / (1.0 + abs(x))


fn softsign_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = 1 / (1 + |x|)^2
    return 1.0 / (1.0 + abs(x)) ** 2


fn tanh_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = tanh(x)
    return tanh(x)


fn tanh_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = 1 - tanh(x)^2
    return 1.0 - tanh(x) ** 2


fn selu_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x > 0 ? 1.05070098 * x : 1.05070098 * 1.67326324 * (e^x - 1)
    return (x > 0.0).cast[DType_F32]() * 1.05070098 * x + (x <= 0.0).cast[
        DType_F32
    ]() * 1.05070098 * 1.67326324 * (exp(x) - 1.0)


fn selu_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = x > 0 ? 1.05070098 : 1.05070098 * 1.67326324 * e^x
    return (x > 0.0).cast[DType_F32]() * 1.05070098 + (x <= 0.0).cast[
        DType_F32
    ]() * 1.05070098 * 1.75809932607 * exp(x)


fn elu_fw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    alpha: Float32 = 1.0,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x > 0 ? x : alpha * (e^x - 1)
    @parameter
    if alpha == 1.0:
        return (x > 0.0).cast[DType_F32]() * x + (x <= 0.0).cast[DType_F32]() * (
            exp(x) - 1.0
        )
    return (x > 0.0).cast[DType_F32]() * x + (x <= 0.0).cast[DType_F32]() * alpha * (
        exp(x) - 1.0
    )


fn elu_bw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    alpha: Float32 = 1.0,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = x > 0 ? 1 : alpha * e^x
    @parameter
    if alpha == 1.0:
        return (x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[DType_F32]() * exp(x)
    return (x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[DType_F32]() * alpha * exp(x)


fn exp_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = e^x
    return exp(x)


fn exp_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = e^x
    return exp(x)


fn silu_fw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x / (1 + e^-x)
    return x / (1.0 + exp(-x))


fn silu_bw_vec[
    _nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = (e^x * x + e^x + e^2x) / (e^x + 1)^2
    return (exp(x) * x + exp(x) + exp(2.0 * x)) / (exp(x) + 1.0) ** 2


fn gelu_fw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    approximate: Float32 = 0.0,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) when approximate == 0.0 = 0.5 * x * (1 + erf(x / sqrt(2)))
    # f(x) when approximate != 0.0 = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    @parameter
    if approximate == 0.0:
        return 0.5 * x * (1.0 + erf(x / 1.4142135623730951))
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


fn gelu_bw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    approximate: Float32 = 0.0,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) when approximate == 0.0 = 0.5 * (erf(0.7071067811865475 * x) + 1) + 0.3989422804014327 * x * exp(-0.5 * x^2)
    # f'(x) when approximate != 0.0 = 0.5 * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3))^2) + 0.7978845608028654 * x * (1 - tanh(0.7978845608028654 * (x + 0.044715 * x^3))^2)
    @parameter
    if approximate == 0.0:
        return 0.5 * (erf(0.7071067811865475 * x) + 1.0) + 0.3989422804014327 * x * exp(
            -0.5 * x**2
        )
    return 0.5 * (
        1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x**3)) ** 2
    ) + 0.7978845608028654 * x * (
        1.0 - tanh(0.7978845608028654 * (x + 0.044715 * x**3)) ** 2
    )


fn hard_sigmoid_fw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x > 2.5 ? 1 : x < -2.5 ? 0 : 0.2 * x + 0.5
    return (x > 2.5).cast[DType_F32]() + (x > -2.5).cast[DType_F32]() * (x < 2.5).cast[
        DType_F32
    ]() * (0.2 * x + 0.5)


fn hard_sigmoid_bw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = x > -2.5 ? x < 2.5 ? 0.2 : 0 : 0
    return (x > -2.5).cast[DType_F32]() * (x < 2.5).cast[DType_F32]() * 0.2


fn linear_fw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x
    return x


fn linear_bw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = 1
    return 1.0


fn mish_fw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f(x) = x * tanh(log(1 + e^x))
    return x * tanh(log(1.0 + exp(x)))


fn mish_bw_vec[
    _nelts: Int,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](x: SIMD[DType_F32, _nelts]) -> SIMD[DType_F32, _nelts]:
    # f'(x) = (e^x (4 e^x x + 4 x + 6 e^x + 4 e^(2 x) + e^(3 x) + 4))/(2 e^x + e^(2 x) + 2)^2
    return (
        exp(x)
        * (
            4.0 * exp(x) * x
            + 4.0 * x
            + 6.0 * exp(x)
            + 4.0 * exp(2.0 * x)
            + exp(3.0 * x)
            + 4.0
        )
        / (2.0 * exp(x) + exp(2.0 * x) + 2.0) ** 2
    )


struct _Softmax:
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


struct _LogSoftmax:
    # f(x) = log(e^wx_i / sum(e^wx_i))
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

            @parameter
            fn vectorized_log[nelts: Int](i: Int):
                node.store_data[nelts](
                    offset + i, log(node.load_data[nelts](offset + i))
                )

            vectorize[nelts, vectorized_log](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // N):
            let offset = s * N

            @parameter
            fn vectorized_log_softmax_bw_outer[nelts: Int](j: Int):
                var grad: Float32 = 0.0

                @parameter
                fn vectorized_log_softmax_bw[nelts: Int](i: Int):
                    if i == j:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * (1.0 - node.load_data[nelts](offset + i))
                        ).reduce_add()
                    else:
                        grad += (
                            node.load_grad[nelts](offset + i)
                            * node.load_data[nelts](offset + j)
                            * -1
                        ).reduce_add()

                vectorize[nelts, vectorized_log_softmax_bw](N)
                parent1.store_grad[nelts](
                    offset + j, parent1.load_grad[nelts](offset + j) + grad
                )

            vectorize[nelts, vectorized_log_softmax_bw_outer](N)
