from math import abs, pow, exp, log, tanh, max, erf
from algorithm import vectorize

from voodoo import Node

alias DType_F32 = DType.float32
alias nelts = simdwidthof[DType_F32]()
alias epsilon = 1e-8


trait Activation:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        ...

    @staticmethod
    fn bw(node: Node, parent1: Node):
        ...


struct Elu(Activation):
    # f(x) = x > 0 ? x : e^x - 1
    # f'(x) = x > 0 ? 1 : e^x
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_fw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                (x > 0.0).cast[DType_F32]() * x
                + (x <= 0.0).cast[DType_F32]() * (exp(x) - 1.0),
            )

        vectorize[nelts, v_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * ((x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[DType_F32]() * exp(x)),
            )

        vectorize[nelts, v_bw](node.load_cap())


struct Exp(Activation):
    # f(x) = e^x
    # f'(x) = e^x
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_exp[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](i, exp(x))

        vectorize[nelts, v_exp](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_exp_bw[_nelts: Int](i: Int):
            let f_x = node.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i) + node.load_grad[_nelts](i) * f_x,
            )

        vectorize[nelts, v_exp_bw](node.load_cap())


struct Gelu(Activation):
    # f(x) = .5x * (1 + erf(x / sqrt(2)))
    # f'(x) = .5 * (1 + erf(x / sqrt(2))) + x * .56418958354 * exp(-.5 * (x / sqrt(2)) ** 2)
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_gelu[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                0.5 * x * (1.0 + erf(x / 1.4142135623730951)),
            )

        vectorize[nelts, v_gelu](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_gelu_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let f_x = node.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * (
                    (f_x / x)
                    + x * 0.56418958354 * exp(-0.5 * (x / 1.4142135623730951) ** 2)
                ),
            )

        vectorize[nelts, v_gelu_bw](node.load_cap())


struct HardSigmoid(Activation):
    # f(x) = x < -2.5 ? 0 : x > 2.5 ? 1 : .2x + .5
    # f'(x) = x < -2.5 ? 0 : x > 2.5 ? 0 : .2
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_h_sig[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                (x > 2.5).cast[DType_F32]()
                + (x > -2.5).cast[DType_F32]()
                * (x < 2.5).cast[DType_F32]()
                * (0.2 * x + 0.5),
            )

        vectorize[nelts, v_h_sig](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_h_sig_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * (x > -2.5).cast[DType_F32]()
                * (x < 2.5).cast[DType_F32]()
                * 0.2,
            )

        vectorize[nelts, v_h_sig_bw](node.load_cap())


struct Linear(Activation):
    # f(x) = x
    # f'(x) = 1
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_linear[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](i, x)

        vectorize[nelts, v_linear](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_linear_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i) + node.load_grad[_nelts](i),
            )

        vectorize[nelts, v_linear_bw](node.load_cap())


struct Mish(Activation):
    # f(x) = x * tanh(ln(1 + e^x))
    # f'(x) = tanh(ln(1 + e^x)) + x * (1 - tanh(ln(1 + e^x)) ** 2) * (1 / (1 + e^-x))
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_mish[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                x * tanh(log(1.0 + exp(x))),
            )

        vectorize[nelts, v_mish](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_mish_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let f_x = node.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * (
                    (f_x / x)
                    + x * (1.0 - tanh(log(1.0 + exp(x))) ** 2) * (1.0 / (1.0 + exp(-x)))
                ),
            )

        vectorize[nelts, v_mish_bw](node.load_cap())


struct ReLu(Activation):
    # f(x) = x > 0 ? x : 0
    # f'(x) = x > 0 ? 1 : 0
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_relu[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                (x > 0.0).cast[DType_F32]() * x,
            )

        vectorize[nelts, v_relu](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_relu_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * (x > 0.0).cast[DType_F32](),
            )

        vectorize[nelts, v_relu_bw](node.load_cap())


struct Selu(Activation):
    # f(x) = x > 0 ? 1.05070098 * x : 1.05070098 * 1.67326324 * e^x - 1
    # f'(x) = x > 0 ? 1 : 1.75809932607 * e^x - 1
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_selu[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                (x > 0.0).cast[DType_F32]() * 1.05070098 * x
                + (x <= 0.0).cast[DType_F32]()
                * 1.05070098
                * 1.67326324
                * (exp(x) - 1.0),
            )

        vectorize[nelts, v_selu](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_selu_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * (
                    (x > 0.0).cast[DType_F32]()
                    + (x <= 0.0).cast[DType_F32]() * 1.75809932607 * exp(x)
                ),
            )

        vectorize[nelts, v_selu_bw](node.load_cap())


struct Sigmoid(Activation):
    # f(x) = 1 / (1 + e^-x)
    # f'(x) = e^x / (e^x + 1) ** 2
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_sig[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](i, 1.0 / (1.0 + exp(-x)))

        vectorize[nelts, v_sig](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_sig_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * (exp(x) / (exp(x) + 1.0) ** 2),
            )

        vectorize[nelts, v_sig_bw](node.load_cap())


struct Softmax(Activation):
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
                node.store_data[nelts](
                    offset + i, node.load_data[nelts](offset + i) / sum
                )

            vectorize[nelts, v_div](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
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

                vectorize[nelts, v_softmax_bw](N)
                parent1.store_grad[nelts](
                    offset + j, parent1.load_grad[nelts](offset + j) + grad
                )

            vectorize[nelts, v_softmax_bw_outer](N)


struct Softplus(Activation):
    # f(x) = ln(1 + e^x)
    # f'(x) = e^x / (1 + e^x)
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_softplus[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](i, log(1.0 + exp(x)))

        vectorize[nelts, v_softplus](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_softplus_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * (exp(x) / (1.0 + exp(x))),
            )

        vectorize[nelts, v_softplus_bw](node.load_cap())


struct Softsign(Activation):
    # f(x) = x / (1 + abs(x))
    # f'(x) = 1 / (1 + abs(x)) ** 2
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_softsign[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                x / (1.0 + abs(x)),
            )

        vectorize[nelts, v_softsign](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_softsign_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let f_x = node.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i) + node.load_grad[_nelts](i) * f_x**2 / x,
            )

        vectorize[nelts, v_softsign_bw](node.load_cap())


struct Swish(Activation):
    # f(x) = x / (1 + e^-x)
    # f'(x) = (1 + e^-x + e^-x * x) / (1 + e^-x) ** 2
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_swish[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                x / (1.0 + exp(-x)),
            )

        vectorize[nelts, v_swish](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_swish_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let f_x = node.load_data[_nelts](i)
            node.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * (f_x**2)
                / x
                * (1 + exp(-x) + exp(-x) * x),
            )

        vectorize[nelts, v_swish_bw](node.load_cap())


struct Tanh(Activation):
    # f(x) = tanh(x)
    # f'(x) = 1 - tanh(x) ** 2
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_tanh[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            node.store_data[_nelts](i, tanh(x))

        vectorize[nelts, v_tanh](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_tanh_bw[_nelts: Int](i: Int):
            let f_x = node.load_data[_nelts](i)
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * (1.0 - f_x**2),
            )

        vectorize[nelts, v_tanh_bw](node.load_cap())


struct LeakyReLu(Activation):
    # f(x) = x > 0 ? x : alpha * x
    # f'(x) = x > 0 ? 1 : alpha
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn v_leaky_relu[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let alpha = node.other_params_ptr.load().data.load().load() / 1000000.0
            node.store_data[_nelts](
                i,
                (x > 0.0).cast[DType_F32]() * x
                + (x <= 0.0).cast[DType_F32]() * alpha * x,
            )

        vectorize[nelts, v_leaky_relu](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn v_leaky_relu_bw[_nelts: Int](i: Int):
            let x = parent1.load_data[_nelts](i)
            let alpha = node.other_params_ptr.load().data.load().load() / 1000000.0
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                * ((x > 0.0).cast[DType_F32]() + (x <= 0.0).cast[DType_F32]() * alpha),
            )

        vectorize[nelts, v_leaky_relu_bw](node.load_cap())
