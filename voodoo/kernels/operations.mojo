from random import random_float64
from algorithm import vectorize
from voodoo import Node
from ..constants import NELTS
from algorithm import vectorize


trait Operation:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        ...

    @staticmethod
    fn bw(node: Node, parent1: Node):
        ...


struct Copy(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy[NELTS: Int](i: Int):
            node.data.load().simd_store[NELTS](
                i, parent1.data.load().simd_load[NELTS](i)
            )

        vectorize[NELTS, vectorized_copy](node.cap)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy_bw[NELTS: Int](i: Int):
            parent1.data.load(1).simd_store[NELTS](
                i, parent1.data.load(1).simd_load[NELTS](i)
            )

        vectorize[NELTS, vectorized_copy_bw](node.cap)


struct Sum(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_sum[NELTS: Int](i: Int):
            sum += parent1.data.load().simd_load[NELTS](i).reduce_add()

        vectorize[NELTS, vectorized_sum](parent1.cap)
        node.data.load().store(0, sum)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sum_bw[NELTS: Int](i: Int):
            parent1.data.load(1).simd_store[NELTS](
                i, parent1.data.load(1).simd_load[NELTS](i) + node.data.load(1).load(0)
            )

        vectorize[NELTS, vectorized_sum_bw](parent1.cap)


struct Reshape(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        for s in range(node.cap // parent1.cap):
            let offset = s * parent1.cap

            @parameter
            fn vectorized_reshape[NELTS: Int](i: Int):
                node.data.load().simd_store[NELTS](
                    i, parent1.data.load().simd_load[NELTS](i)
                )

            vectorize[NELTS, vectorized_reshape](parent1.cap)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        for s in range(node.cap // parent1.cap):
            let offset = s * parent1.cap

            @parameter
            fn vectorized_reshape[NELTS: Int](i: Int):
                parent1.data.load(1).simd_store[NELTS](
                    i,
                    parent1.data.load(1).simd_load[NELTS](i)
                    + node.data.load(1).simd_load[NELTS](i),
                )

            vectorize[NELTS, vectorized_reshape](parent1.cap)


struct Transpose(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims
        let M = parent1.shape.load(num_dims - 2)
        let N = parent1.shape.load(num_dims - 1)
        for s in range(node.cap // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp[NELTS: Int](j: Int):
                    node.data.load().simd_store[NELTS](
                        offset + j * M + i,
                        parent1.data.load().simd_load[NELTS](offset + i * N + j),
                    )

                vectorize[NELTS, vectorized_transp](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims
        let M = parent1.shape.load(num_dims - 2)
        let N = parent1.shape.load(num_dims - 1)
        for s in range(node.cap // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp_bw[NELTS: Int](j: Int):
                    parent1.data.load(1).simd_store[NELTS](
                        offset + j * M + i,
                        parent1.data.load(1).simd_load[NELTS](offset + j * M + i)
                        + node.data.load(1).simd_load[NELTS](offset + i * N + j),
                    )

                vectorize[NELTS, vectorized_transp_bw](N)


struct Dropout(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let params = node.other_params
        let keep_prob = 1 - params.load(0) / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout[NELTS: Int](i: Int):
            let rand = random_float64()
            node.data.load().simd_store[NELTS](
                i,
                (rand < keep_prob).cast[DType.float32]()
                * parent1.data.load().simd_load[NELTS](i),
            )

        vectorize[NELTS, vectorized_dropout](node.cap)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let params = node.other_params
        let keep_prob = 1 - params.load(0) / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout_bw[NELTS: Int](i: Int):
            let previous = node.data.load().simd_load[NELTS](i)
            node.data.load(1).simd_store[NELTS](
                i,
                (previous == 0.0).cast[DType.float32]()
                * parent1.data.load(1).simd_load[NELTS](i)
                * scale,
            )

        vectorize[NELTS, vectorized_dropout_bw](node.cap)
