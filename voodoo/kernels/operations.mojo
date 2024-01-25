from random import random_float64
from algorithm import vectorize
from voodoo import Node
from ..constants import nelts
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
        fn vectorized_copy[nelts: Int](i: Int):
            node.store_data[nelts](i, parent1.load_data[nelts](i))

        vectorize[nelts, vectorized_copy](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_copy_bw[nelts: Int](i: Int):
            parent1.store_grad[nelts](i, parent1.load_grad[nelts](i))

        vectorize[nelts, vectorized_copy_bw](node.load_cap())


struct Sum(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_sum[nelts: Int](i: Int):
            sum += parent1.load_data[nelts](i).reduce_add()

        vectorize[nelts, vectorized_sum](parent1.load_cap())
        node.store_data(0, sum)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_sum_bw[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i, parent1.load_grad[nelts](i) + node.load_grad(0)
            )

        vectorize[nelts, vectorized_sum_bw](parent1.load_cap())


struct Reshape(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
            let offset = s * parent1.cap_ptr.load()

            @parameter
            fn vectorized_reshape[nelts: Int](i: Int):
                node.store_data[nelts](i, parent1.load_data[nelts](i))

            vectorize[nelts, vectorized_reshape](parent1.cap_ptr.load())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
            let offset = s * parent1.cap_ptr.load()

            @parameter
            fn vectorized_reshape[nelts: Int](i: Int):
                parent1.store_grad[nelts](
                    i, parent1.load_grad[nelts](i) + node.load_grad[nelts](i)
                )

            vectorize[nelts, vectorized_reshape](parent1.cap_ptr.load())


struct Transpose(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let M = parent1.shape_ptr.load().load(num_dims - 2)
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp[nelts: Int](j: Int):
                    node.store_data[nelts](
                        offset + j * M + i, parent1.load_data[nelts](offset + i * N + j)
                    )

                vectorize[nelts, vectorized_transp](N)

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let num_dims = parent1.num_dims_ptr.load()
        let M = parent1.shape_ptr.load().load(num_dims - 2)
        let N = parent1.shape_ptr.load().load(num_dims - 1)
        for s in range(node.cap_ptr.load() // (M * N)):
            let offset = s * M * N
            for i in range(M):

                @parameter
                fn vectorized_transp_bw[nelts: Int](j: Int):
                    parent1.store_grad[nelts](
                        offset + j * M + i,
                        parent1.load_grad[nelts](offset + j * M + i)
                        + node.load_grad[nelts](offset + i * N + j),
                    )

                vectorize[nelts, vectorized_transp_bw](N)


struct Dropout(Operation):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let params = node.other_params_ptr.load()
        let keep_prob = 1 - params.load(0) / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout[nelts: Int](i: Int):
            let rand = random_float64()
            node.store_data[nelts](
                i,
                (rand < keep_prob).cast[DType.float32]() * parent1.load_data[nelts](i),
            )

        vectorize[nelts, vectorized_dropout](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let params = node.other_params_ptr.load()
        let keep_prob = 1 - params.load(0) / 1000000.0
        let scale = 1.0 / keep_prob

        @parameter
        fn vectorized_dropout_bw[nelts: Int](i: Int):
            let previous = node.load_data[nelts](i)
            node.store_grad[nelts](
                i,
                (previous == 0.0).cast[DType.float32]()
                * parent1.load_grad[nelts](i)
                * scale,
            )

        vectorize[nelts, vectorized_dropout_bw](node.load_cap())
