from random import random_float64
from algorithm import vectorize

from voodoo.autograd import Node
from voodoo.constants import NELTS


trait Operation:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        ...

    @staticmethod
    fn bw(node: Node, parent1: Node):
        ...


@register_passable("trivial")
struct Copy(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy[NELTS: Int](i: Int):
            node.get_data().simd_store[NELTS](i, parent1.get_data().simd_load[NELTS](i))

        vectorize[NELTS, vectorized_copy](node.get_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy_bw[NELTS: Int](i: Int):
            parent1.get_grad().simd_store[NELTS](
                i, parent1.get_grad().simd_load[NELTS](i)
            )

        vectorize[NELTS, vectorized_copy_bw](node.get_cap())


@register_passable("trivial")
struct Sum(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_sum[NELTS: Int](i: Int):
            sum += parent1.get_data().simd_load[NELTS](i).reduce_add()

        vectorize[NELTS, vectorized_sum](parent1.get_cap())
        node.get_data().store(0, sum)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sum_bw[NELTS: Int](i: Int):
            parent1.get_grad().simd_store[NELTS](
                i,
                parent1.get_grad().simd_load[NELTS](i) + node.get_grad()[0],
            )

        vectorize[NELTS, vectorized_sum_bw](parent1.get_cap())


@register_passable("trivial")
struct Reshape(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        for s in range(node.get_cap() // parent1.get_cap()):
            let offset = s * parent1.get_cap()

            @parameter
            fn vectorized_reshape[NELTS: Int](i: Int):
                node.get_data().simd_store[NELTS](
                    i, parent1.get_data().simd_load[NELTS](i)
                )

            vectorize[NELTS, vectorized_reshape](parent1.get_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        for s in range(node.get_cap() // parent1.get_cap()):
            let offset = s * parent1.get_cap()

            @parameter
            fn vectorized_reshape[NELTS: Int](i: Int):
                parent1.get_grad().simd_store[NELTS](
                    i,
                    parent1.get_grad().simd_load[NELTS](i)
                    + node.get_grad().simd_load[NELTS](i),
                )

            vectorize[NELTS, vectorized_reshape](parent1.get_cap())


@register_passable("trivial")
struct Transpose(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.get_num_dims()
        let M = parent1.get_shape()[num_dims - 2]
        let N = parent1.get_shape()[num_dims - 1]
        for s in range(node.get_cap() // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp[NELTS: Int](j: Int):
                    node.get_data().simd_store[NELTS](
                        offset + j * M + i,
                        parent1.get_data().simd_load[NELTS](offset + i * N + j),
                    )

                vectorize[NELTS, vectorized_transp](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.get_num_dims()
        let M = parent1.get_shape()[num_dims - 2]
        let N = parent1.get_shape()[num_dims - 1]
        for s in range(node.get_cap() // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp_bw[NELTS: Int](j: Int):
                    parent1.get_grad().simd_store[NELTS](
                        offset + j * M + i,
                        parent1.get_grad().simd_load[NELTS](offset + j * M + i)
                        + node.get_grad().simd_load[NELTS](offset + i * N + j),
                    )

                vectorize[NELTS, vectorized_transp_bw](N)


@register_passable("trivial")
struct Dropout(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let params = node.get_other_params()
        let keep_prob = 1 - params[0] / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout[NELTS: Int](i: Int):
            let rand = random_float64()
            node.get_data().simd_store[NELTS](
                i,
                (rand < keep_prob).select[DType.float32](1.0, 0.0)
                * parent1.get_data().simd_load[NELTS](i),
            )

        vectorize[NELTS, vectorized_dropout](node.get_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let params = node.get_other_params()
        let keep_prob = 1 - params[0] / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout_bw[NELTS: Int](i: Int):
            let previous = node.get_data().simd_load[NELTS](i)
            node.get_grad().simd_store[NELTS](
                i,
                (previous == 0.0).select[DType.float32](
                    parent1.get_grad().simd_load[NELTS](i) * scale, 0.0
                ),
            )

        vectorize[NELTS, vectorized_dropout_bw](node.get_cap())
