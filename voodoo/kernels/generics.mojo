from algorithm import vectorize
from voodoo.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import nelts

alias generic_activation_vectorized = fn[
    nelts: Int, arg1: Float32, arg2: Float32, arg3: Float32
] (SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]

alias generic_arithmetic_vectorized = fn[nelts: Int] (
    SIMD[DType.float32, nelts]
) -> SIMD[DType.float32, nelts]

alias generic_binary_arithmetic_vectorized = fn[nelts: Int] (
    SIMD[DType.float32, nelts], SIMD[DType.float32, nelts]
) -> SIMD[DType.float32, nelts]

alias generic_loss_vectorized_fw = generic_binary_arithmetic_vectorized

alias generic_loss_vectorized_bw = fn[nelts: Int] (
    SIMD[DType.float32, nelts], SIMD[DType.float32, nelts], Float32, Int
) -> SIMD[DType.float32, nelts]

alias generic_optimizer_vectorized = fn[nelts: Int, learning_rate: Float32] (
    SIMD[DType.float32, nelts]
) -> SIMD[DType.float32, nelts]


struct GenericActivation[
    fw_vec: generic_activation_vectorized,
    bw_vec: generic_activation_vectorized,
    arg1: Float32,
    arg2: Float32,
    arg3: Float32,
]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            node.store_data[nelts](
                i,
                fw_vec[nelts, arg1, arg2, arg3](parent1.load_data[nelts](i)),
            )

        vectorize[nelts, vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        @always_inline
        fn vectorized_bw[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i,
                parent1.load_grad[nelts](i)
                + node.load_grad[nelts](i)
                * bw_vec[nelts, arg1, arg2, arg3](parent1.load_data[nelts](i)),
            )

        vectorize[nelts, vectorized_bw](node.load_cap())


struct GenericArithmetic[
    fw_vec: generic_arithmetic_vectorized, bw_vec: generic_arithmetic_vectorized
]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        @parameter
        fn vectorized_fw[nelts: Int](i: Int):
            let x = parent1.load_data[nelts](i)
            node.store_data[nelts](
                i,
                fw_vec[nelts](x),
            )

        vectorize[nelts, vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        @parameter
        fn vectorized_bw[nelts: Int](i: Int):
            let x = parent1.load_data[nelts](i)
            parent1.store_grad[nelts](
                i,
                parent1.load_grad[nelts](i)
                + node.load_grad[nelts](i) * bw_vec[nelts](x),
            )

        vectorize[nelts, vectorized_bw](node.load_cap())


struct GenericBinaryArithmetic[
    fw_vec: generic_binary_arithmetic_vectorized,
    bw_a_vec: generic_binary_arithmetic_vectorized,
    bw_b_vec: generic_binary_arithmetic_vectorized,
]:
    @staticmethod
    fn fw(c: Node, a: Node, b: Node):
        recursive_broadcast[Self.kernel_fw[fw_vec], Self.base_case](c, a, b)

    @staticmethod
    fn bw(c: Node, a: Node, b: Node):
        if not a.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_bw[bw_a_vec, True], Self.base_case](
                c, a, b
            )
        if not b.is_single_ptr.load():
            recursive_broadcast_bw[Self.kernel_bw[bw_b_vec, False], Self.base_case](
                c, a, b
            )

    @parameter
    @staticmethod
    fn kernel_fw[
        generic_func: generic_binary_arithmetic_vectorized
    ](
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            c.store_data[nelts](
                offset_c + i,
                generic_func(
                    a.load_data[nelts](offset_a + i), b.load_data[nelts](offset_b + i)
                ),
            )

        vectorize[nelts, vectorized_fw](c_rest)

    @parameter
    @staticmethod
    fn kernel_bw[
        generic_func: generic_binary_arithmetic_vectorized,
        is_a: Bool,
    ](
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None:
        let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
        let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
        let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
        let offset_c = c_index * c_rest

        @parameter
        @always_inline
        fn vectorized_bw_a[nelts: Int](i: Int):
            a.store_grad[nelts](
                offset_a + i,
                a.load_grad[nelts](offset_a + i)
                + generic_func(
                    a.load_data[nelts](offset_a + i),
                    b.load_data[nelts](offset_b + i),
                )
                * c.load_grad[nelts](offset_c + i),
            )

        @parameter
        @always_inline
        fn vectorized_bw_b[nelts: Int](i: Int):
            b.store_grad[nelts](
                offset_b + i,
                b.load_grad[nelts](offset_b + i)
                + generic_func(
                    a.load_data[nelts](offset_a + i),
                    b.load_data[nelts](offset_b + i),
                )
                * c.load_grad[nelts](offset_c + i),
            )

        @parameter
        if is_a:
            vectorize[nelts, vectorized_bw_a](c_rest)
        else:
            vectorize[nelts, vectorized_bw_b](c_rest)

    @parameter
    @always_inline
    @staticmethod
    fn base_case(depth: Int, a: Node, b: Node) -> Bool:
        return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
            depth, a, b
        ) * shape_b(depth, a, b)


struct GenericLoss[
    fw_vec: generic_loss_vectorized_fw,
    bw_vec: generic_loss_vectorized_bw,
]:
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = y_pred.shape_ptr.load().len.load()
        let N = y_pred.shape_ptr.load().load(num_dims - 1)
        let cap = Float32(y_pred.load_cap())
        var e: Float32 = 0.0

        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            let error = fw_vec[nelts](
                y_true.load_data[nelts](i), y_pred.load_data[nelts](i)
            )
            e += error.reduce_add()

        vectorize[nelts, vectorized_fw](cap.to_int())
        node.store_data(0, e / cap / Float32(N))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = y_pred.shape_ptr.load().len.load()
        let N = y_pred.shape_ptr.load().load(num_dims - 1)
        let cap = y_pred.load_cap()
        let scalar = cap / Float32(N)

        @parameter
        @always_inline
        fn vectorized_mae_bw[nelts: Int](i: Int):
            let grad = bw_vec[nelts](
                y_true.load_data[nelts](i), y_pred.load_data[nelts](i), cap, N
            ) / scalar

            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_mae_bw](y_pred.load_cap())


struct GenericOptimizer[fw_vec: generic_optimizer_vectorized]:
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                @always_inline
                fn vectorized_update[nelts: Int](i: Int):
                    node.store_data[nelts](
                        i,
                        node.load_data[nelts](i)
                        - fw_vec[nelts, learning_rate](node.load_grad[nelts](i)),
                    )

                vectorize[nelts, vectorized_update](node.load_cap())
