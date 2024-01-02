from math import (
    sqrt,
    exp2,
    log2,
    log,
    cos,
    sin,
    tan,
    asin,
    acos,
    atan,
    cosh,
    sinh,
)
from algorithm import vectorize
from voodoo import Node
from .shared import DType_F32, nelts


trait UnaryArithmetic:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        ...

    @staticmethod
    fn bw(node: Node, parent1: Node):
        ...


struct Sqrt(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sqrt[_nelts: Int](i: Int):
            node.store_data[_nelts](i, sqrt(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_sqrt](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sqrt_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                / (2.0 * sqrt(parent1.load_data[_nelts](i))),
            )

        vectorize[nelts, vectorized_sqrt_bw](node.load_cap())


struct Abs(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_abs[_nelts: Int](i: Int):
            let data = parent1.load_data[_nelts](i)
            node.store_data[_nelts](
                i,
                (data >= 0.0).cast[DType_F32]() * data
                + (data < 0.0).cast[DType_F32]() * (-data),
            )

        vectorize[nelts, vectorized_abs](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_abs_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + (2.0 * (parent1.load_data[_nelts](i) >= 0.0).cast[DType_F32]() - 1.0)
                * node.load_grad[_nelts](i),
            )

        vectorize[nelts, vectorized_abs_bw](node.load_cap())


struct Exp2(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_exp2[_nelts: Int](i: Int):
            node.store_data[_nelts](i, exp2(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_exp2](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_exp2_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * node.load_data[_nelts](i) * 0.69314718056,
            )

        vectorize[nelts, vectorized_exp2_bw](node.load_cap())


struct Log2(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_log2[_nelts: Int](i: Int):
            node.store_data[_nelts](i, log2(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_log2](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_log2_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                / (parent1.load_data[_nelts](i) * 0.69314718056),
            )

        vectorize[nelts, vectorized_log2_bw](node.load_cap())


struct Log(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_log[_nelts: Int](i: Int):
            node.store_data[_nelts](i, log(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_log](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_log_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) / parent1.load_data[_nelts](i),
            )

        vectorize[nelts, vectorized_log_bw](node.load_cap())


struct Sin(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sin[_nelts: Int](i: Int):
            node.store_data[_nelts](i, sin(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_sin](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sin_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + cos(parent1.load_data[_nelts](i)) * node.load_grad[_nelts](i),
            )

        vectorize[nelts, vectorized_sin_bw](node.load_cap())


struct Cos(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_cos[_nelts: Int](i: Int):
            node.store_data[_nelts](i, cos(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_cos](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_cos_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                - sin(parent1.load_data[_nelts](i)) * node.load_grad[_nelts](i),
            )

        vectorize[nelts, vectorized_cos_bw](node.load_cap())


struct Tan(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_tan[_nelts: Int](i: Int):
            node.store_data[_nelts](i, tan(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_tan](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_tan_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) / (cos(parent1.load_data[_nelts](i))) ** 2,
            )

        vectorize[nelts, vectorized_tan_bw](node.load_cap())


struct Asin(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_asin[_nelts: Int](i: Int):
            node.store_data[_nelts](i, asin(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_asin](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_asin_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                / sqrt(1.0 - (parent1.load_data[_nelts](i)) ** 2),
            )

        vectorize[nelts, vectorized_asin_bw](node.load_cap())


struct Acos(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_acos[_nelts: Int](i: Int):
            node.store_data[_nelts](i, acos(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_acos](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_acos_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                - node.load_grad[_nelts](i)
                / sqrt(1.0 - (parent1.load_data[_nelts](i)) ** 2),
            )

        vectorize[nelts, vectorized_acos_bw](node.load_cap())


struct Atan(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_atan[_nelts: Int](i: Int):
            node.store_data[_nelts](i, atan(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_atan](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_atan_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i)
                / (1.0 + (parent1.load_data[_nelts](i)) ** 2),
            )

        vectorize[nelts, vectorized_atan_bw](node.load_cap())


struct Sinh(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sinh[_nelts: Int](i: Int):
            node.store_data[_nelts](i, sinh(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_sinh](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sinh_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * cosh(parent1.load_data[_nelts](i)),
            )

        vectorize[nelts, vectorized_sinh_bw](node.load_cap())


struct Cosh(UnaryArithmetic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_cosh[_nelts: Int](i: Int):
            node.store_data[_nelts](i, cosh(parent1.load_data[_nelts](i)))

        vectorize[nelts, vectorized_cosh](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_cosh_bw[_nelts: Int](i: Int):
            parent1.store_grad[_nelts](
                i,
                parent1.load_grad[_nelts](i)
                + node.load_grad[_nelts](i) * sinh(parent1.load_data[_nelts](i)),
            )

        vectorize[nelts, vectorized_cosh_bw](node.load_cap())
