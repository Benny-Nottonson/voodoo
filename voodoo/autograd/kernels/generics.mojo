from algorithm import vectorize

from voodoo.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    Vector,
)
from voodoo.constants import NELTS, PREFETCH_READ, PREFETCH_WRITE


trait Generic:
    ...


alias generic_activation_vectorized = fn[
    NELTS: Int, arg1: Float32, arg2: Float32, arg3: Float32
] (SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]

alias generic_arithmetic_vectorized = fn[NELTS: Int] (
    SIMD[DType.float32, NELTS]
) -> SIMD[DType.float32, NELTS]

alias generic_binary_arithmetic_vectorized = fn[NELTS: Int] (
    SIMD[DType.float32, NELTS], SIMD[DType.float32, NELTS]
) -> SIMD[DType.float32, NELTS]

alias generic_loss_vectorized_fw = generic_binary_arithmetic_vectorized

alias generic_loss_vectorized_bw = fn[NELTS: Int] (
    SIMD[DType.float32, NELTS], SIMD[DType.float32, NELTS], Float32, Int
) -> SIMD[DType.float32, NELTS]

alias generic_optimizer_vectorized = fn[NELTS: Int, learning_rate: Float32] (
    SIMD[DType.float32, NELTS]
) -> SIMD[DType.float32, NELTS]


@register_passable("trivial")
struct GenericActivation[
    fw_vec: generic_activation_vectorized,
    bw_vec: generic_activation_vectorized,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
](Generic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let node_data = node.get_data()
        let parent1_data = parent1.get_data()

        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](node_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](parent1_data)

        @parameter
        fn vectorized_fw[NELTS: Int](i: Int):
            node_data.simd_store[NELTS](
                i,
                fw_vec[NELTS, arg1, arg2, arg3](parent1_data.simd_load[NELTS](i)),
            )

        vectorize[NELTS, vectorized_fw](node.get_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let node_data = node.get_data()
        let node_grad = node.get_grad()
        let parent1_data = parent1.get_data()
        let parent1_grad = parent1.get_grad()

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](parent1_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](node_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](parent1_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](parent1_grad)

        @parameter
        fn vectorized_bw[NELTS: Int](i: Int):
            parent1_grad.simd_store[NELTS](
                i,
                parent1_grad.simd_load[NELTS](i)
                + node_grad.simd_load[NELTS](i)
                * bw_vec[NELTS, arg1, arg2, arg3](parent1_data.simd_load[NELTS](i)),
            )

        vectorize[NELTS, vectorized_bw](node.get_cap())


@register_passable("trivial")
struct GenericArithmetic[
    fw_vec: generic_arithmetic_vectorized, bw_vec: generic_arithmetic_vectorized
](Generic):
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let node_data = node.get_data()
        let parent1_data = parent1.get_data()

        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](node_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](parent1_data)

        @parameter
        fn vectorized_fw[NELTS: Int](i: Int):
            node_data.simd_store[NELTS](
                i,
                fw_vec[NELTS](parent1_data.simd_load[NELTS](i)),
            )

        vectorize[NELTS, vectorized_fw](node.get_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let node_data = node.get_data()
        let node_grad = node.get_grad()
        let parent1_data = parent1.get_data()
        let parent1_grad = parent1.get_grad()

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](parent1_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](node_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](parent1_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](parent1_grad)

        @parameter
        fn vectorized_bw[NELTS: Int](i: Int):
            parent1_grad.simd_store[NELTS](
                i,
                parent1_grad.simd_load[NELTS](i)
                + node_grad.simd_load[NELTS](i)
                * bw_vec[NELTS](parent1_data.simd_load[NELTS](i)),
            )

        vectorize[NELTS, vectorized_bw](node.get_cap())


@register_passable("trivial")
struct GenericBinaryArithmetic[
    fw_vec: generic_binary_arithmetic_vectorized,
    bw_a_vec: generic_binary_arithmetic_vectorized,
    bw_b_vec: generic_binary_arithmetic_vectorized,
](Generic):
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_fw[fw_vec], True](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.get_is_single():
            recursive_broadcast[Self.kernel_bw[bw_a_vec, True], True](c, a, b)
        if not b.get_is_single():
            recursive_broadcast[Self.kernel_bw[bw_b_vec, False], True](c, a, b)

    @staticmethod
    fn kernel_fw[
        generic_func: generic_binary_arithmetic_vectorized
    ](
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.get_shape()[depth] * c.get_strides()[depth]
        let offset_c = c_index * c_rest

        let a_data = a.get_data()
        let b_data = b.get_data()
        let c_data = c.get_data()

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](a_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](c_data)

        @parameter
        fn vectorized_fw[NELTS: Int](i: Int):
            c_data.simd_store[NELTS](
                offset_c + i,
                generic_func(
                    a_data.simd_load[NELTS](offset_a + i),
                    b_data.simd_load[NELTS](offset_b + i),
                ),
            )

        vectorize[NELTS, vectorized_fw](c_rest)

    @staticmethod
    fn kernel_bw[
        generic_func: generic_binary_arithmetic_vectorized,
        is_a: Bool,
    ](
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let offset_c = c_index * c.get_shape()[depth] * c.get_strides()[depth]
        let c_rest = c.get_shape()[depth] * c.get_strides()[depth]

        @parameter
        if is_a:
            let a_data = a.get_data()
            let b_data = b.get_data()
            let a_grad = a.get_grad()
            let c_grad = c.get_grad()

            DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](a_grad)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](a_grad)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](a_data)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_data)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](c_grad)

            @parameter
            fn vectorized_bw_a[NELTS: Int](i: Int):
                a_grad.simd_store[NELTS](
                    offset_a + i,
                    a_grad.simd_load[NELTS](offset_a + i)
                    + generic_func(
                        a_data.simd_load[NELTS](offset_a + i),
                        b_data.simd_load[NELTS](offset_b + i),
                    )
                    * c_grad.simd_load[NELTS](offset_c + i),
                )

            vectorize[NELTS, vectorized_bw_a](c_rest)
        else:
            let a_data = a.get_data()
            let b_data = b.get_data()
            let b_grad = b.get_grad()
            let c_grad = c.get_grad()

            DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](b_grad)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_grad)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](a_data)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](b_data)
            DTypePointer[DType.float32].prefetch[PREFETCH_READ](c_grad)

            @parameter
            fn vectorized_bw_b[NELTS: Int](i: Int):
                b_grad.simd_store[NELTS](
                    offset_b + i,
                    b_grad.simd_load[NELTS](offset_b + i)
                    + generic_func(
                        a_data.simd_load[NELTS](offset_a + i),
                        b_data.simd_load[NELTS](offset_b + i),
                    )
                    * c_grad.simd_load[NELTS](offset_c + i),
                )

            vectorize[NELTS, vectorized_bw_b](c_rest)


@register_passable("trivial")
struct GenericLoss[
    fw_vec: generic_loss_vectorized_fw,
    bw_vec: generic_loss_vectorized_bw,
](Generic):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = len(y_pred.get_shape())
        let N = y_pred.get_shape()[num_dims - 1]
        let cap = y_pred.get_cap()
        var e: Float32 = 0.0

        let y_pred_data = y_pred.get_data()
        let y_true_data = y_true.get_data()

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](y_pred_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](y_true_data)

        @parameter
        fn vectorized_fw[NELTS: Int](i: Int):
            node.get_data().store(
                0,
                node.get_data()[0]
                + fw_vec[NELTS](
                    y_true.get_data().simd_load[NELTS](i),
                    y_pred.get_data().simd_load[NELTS](i),
                ).reduce_add(),
            )

        vectorize[NELTS, vectorized_fw](cap)
        node.get_data().store(0, node.get_data()[0] / cap / Float32(N))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = len(y_pred.get_shape())
        let N = y_pred.get_shape()[num_dims - 1]
        let cap = y_pred.get_cap()
        let scalar = cap / Float32(N)

        let y_pred_data = y_pred.get_data()
        let y_pred_grad = y_pred.get_grad()
        let y_true_data = y_true.get_data()
        let y_true_grad = y_true.get_grad()

        DTypePointer[DType.float32].prefetch[PREFETCH_READ](y_pred_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](y_pred_grad)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](y_true_data)
        DTypePointer[DType.float32].prefetch[PREFETCH_READ](y_true_grad)

        @parameter
        fn vectorized_mae_bw[NELTS: Int](i: Int):
            let grad = bw_vec[NELTS](
                y_true_data.simd_load[NELTS](i), y_pred_data.simd_load[NELTS](i), cap, N
            ) / scalar

            y_pred_grad.simd_store[NELTS](i, y_pred_grad.simd_load[NELTS](i) + grad)
            y_true_grad.simd_store[NELTS](i, y_true_grad.simd_load[NELTS](i) - grad)

        vectorize[NELTS, vectorized_mae_bw](cap)


@register_passable("trivial")
struct GenericOptimizer[fw_vec: generic_optimizer_vectorized](Generic):
    @staticmethod
    fn step[learning_rate: Float32](x: Vector[Node]) raises:
        for i in range(len(x)):
            let node = x[i]
            if node.get_is_static() and node.get_grad_computed():
                let node_data = node.get_data()
                let node_grad = node.get_grad()

                DTypePointer[DType.float32].prefetch[PREFETCH_READ](node_data)
                DTypePointer[DType.float32].prefetch[PREFETCH_READ](node_grad)
                DTypePointer[DType.float32].prefetch[PREFETCH_WRITE](node_data)

                @parameter
                fn vectorized_update[NELTS: Int](i: Int):
                    node_data.simd_store[NELTS](
                        i,
                        node_data.simd_load[NELTS](i)
                        - fw_vec[NELTS, learning_rate](node_grad.simd_load[NELTS](i)),
                    )

                vectorize[NELTS, vectorized_update](node.get_cap())
