from algorithm import vectorize_unroll
from voodoo.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)
from ..constants import nelts, prefetch_read, prefetch_write

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
        let node_data = node.data.load(0)
        let parent1_data = parent1.data.load(0)

        DTypePointer[DType.float32].prefetch[prefetch_write](node_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](parent1_data)

        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            node_data.simd_store[nelts](
                i,
                fw_vec[nelts, arg1, arg2, arg3](parent1_data.simd_load[nelts](i)),
            )

        vectorize_unroll[nelts, 1, vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let node_data = node.data.load(0)
        let node_grad = node.data.load(1)
        let parent1_data = parent1.data.load(0)
        let parent1_grad = parent1.data.load(1)

        DTypePointer[DType.float32].prefetch[prefetch_read](parent1_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](node_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](parent1_data)
        DTypePointer[DType.float32].prefetch[prefetch_write](parent1_grad)

        @parameter
        @always_inline
        fn vectorized_bw[nelts: Int](i: Int):
            parent1_grad.simd_store[nelts](
                i,
                parent1_grad.simd_load[nelts](i)
                + node_grad.simd_load[nelts](i)
                * bw_vec[nelts, arg1, arg2, arg3](parent1_data.simd_load[nelts](i)),
            )

        vectorize_unroll[nelts, 1, vectorized_bw](node.load_cap())


struct GenericArithmetic[
    fw_vec: generic_arithmetic_vectorized, bw_vec: generic_arithmetic_vectorized
]:
    @staticmethod
    fn fw(node: Node, parent1: Node):
        let node_data = node.data.load(0)
        let parent1_data = parent1.data.load(0)

        DTypePointer[DType.float32].prefetch[prefetch_write](node_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](parent1_data)

        @parameter
        fn vectorized_fw[nelts: Int](i: Int):
            node_data.simd_store[nelts](
                i,
                fw_vec[nelts](parent1_data.simd_load[nelts](i)),
            )

        vectorize_unroll[nelts, 1, vectorized_fw](node.load_cap())

    @staticmethod
    fn bw(node: Node, parent1: Node):
        let node_data = node.data.load(0)
        let node_grad = node.data.load(1)
        let parent1_data = parent1.data.load(0)
        let parent1_grad = parent1.data.load(1)

        DTypePointer[DType.float32].prefetch[prefetch_read](parent1_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](node_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](parent1_data)
        DTypePointer[DType.float32].prefetch[prefetch_write](parent1_grad)

        @parameter
        fn vectorized_bw[nelts: Int](i: Int):
            parent1_grad.simd_store[nelts](
                i,
                parent1_grad.simd_load[nelts](i)
                + node_grad.simd_load[nelts](i)
                * bw_vec[nelts](parent1_data.simd_load[nelts](i)),
            )

        vectorize_unroll[nelts, 1, vectorized_bw](node.load_cap())


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

        let a_data = a.data.load(0)
        let b_data = b.data.load(0)
        let c_data = c.data.load(0)

        DTypePointer[DType.float32].prefetch[prefetch_read](a_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](b_data)
        DTypePointer[DType.float32].prefetch[prefetch_write](c_data)

        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            c_data.simd_store[nelts](
                offset_c + i,
                generic_func(
                    a_data.simd_load[nelts](offset_a + i),
                    b_data.simd_load[nelts](offset_b + i),
                ),
            )

        vectorize_unroll[nelts, 1, vectorized_fw](c_rest)

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

        let a_data = a.data.load(0)
        let b_data = b.data.load(0)
        let a_grad = a.data.load(1)
        let b_grad = b.data.load(1)
        let c_grad = c.data.load(1)

        DTypePointer[DType.float32].prefetch[prefetch_read](a_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](b_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](c_grad)
        DTypePointer[DType.float32].prefetch[prefetch_write](a_grad)
        DTypePointer[DType.float32].prefetch[prefetch_write](b_grad)

        @parameter
        @always_inline
        fn vectorized_bw_a[nelts: Int](i: Int):
            a_grad.simd_store[nelts](
                offset_a + i,
                a_grad.simd_load[nelts](offset_a + i)
                + generic_func(
                    a_data.simd_load[nelts](offset_a + i),
                    b_data.simd_load[nelts](offset_b + i),
                )
                * c_grad.simd_load[nelts](offset_c + i),
            )

        @parameter
        @always_inline
        fn vectorized_bw_b[nelts: Int](i: Int):
            b_grad.simd_store[nelts](
                offset_b + i,
                b_grad.simd_load[nelts](offset_b + i)
                + generic_func(
                    a_data.simd_load[nelts](offset_a + i),
                    b_data.simd_load[nelts](offset_b + i),
                )
                * c_grad.simd_load[nelts](offset_c + i),
            )

        @parameter
        if is_a:
            vectorize_unroll[nelts, 1, vectorized_bw_a](c_rest)
        else:
            vectorize_unroll[nelts, 1, vectorized_bw_b](c_rest)

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
        let cap = y_pred.load_cap()
        var e: Float32 = 0.0

        let y_pred_data = y_pred.data.load(0)
        let y_true_data = y_true.data.load(0)

        DTypePointer[DType.float32].prefetch[prefetch_read](y_pred_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](y_true_data)

        @parameter
        @always_inline
        fn vectorized_fw[nelts: Int](i: Int):
            node.store_data(
                0,
                node.load_data(0)
                + fw_vec[nelts](
                    y_true.load_data[nelts](i), y_pred.load_data[nelts](i)
                ).reduce_add(),
            )

        vectorize_unroll[nelts, 1, vectorized_fw](cap)
        node.store_data(0, node.load_data(0) / cap / Float32(N))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = y_pred.shape_ptr.load().len.load()
        let N = y_pred.shape_ptr.load().load(num_dims - 1)
        let cap = y_pred.load_cap()
        let scalar = cap / Float32(N)

        let y_pred_data = y_pred.data.load(0)
        let y_pred_grad = y_pred.data.load(1)
        let y_true_data = y_true.data.load(0)
        let y_true_grad = y_true.data.load(1)

        DTypePointer[DType.float32].prefetch[prefetch_read](y_pred_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](y_pred_grad)
        DTypePointer[DType.float32].prefetch[prefetch_read](y_true_data)
        DTypePointer[DType.float32].prefetch[prefetch_read](y_true_grad)

        @parameter
        @always_inline
        fn vectorized_mae_bw[nelts: Int](i: Int):
            let grad = bw_vec[nelts](
                y_true_data.simd_load[nelts](i), y_pred_data.simd_load[nelts](i), cap, N
            ) / scalar

            y_pred_grad.simd_store[nelts](i, y_pred_grad.simd_load[nelts](i) + grad)
            y_true_grad.simd_store[nelts](i, y_true_grad.simd_load[nelts](i) - grad)

        vectorize_unroll[nelts, 1, vectorized_mae_bw](cap)


struct GenericOptimizer[fw_vec: generic_optimizer_vectorized]:
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():
                let node_data = node.data.load(0)
                let node_grad = node.data.load(1)

                DTypePointer[DType.float32].prefetch[prefetch_read](node_data)
                DTypePointer[DType.float32].prefetch[prefetch_read](node_grad)
                DTypePointer[DType.float32].prefetch[prefetch_write](node_data)

                @parameter
                @always_inline
                fn vectorized_update[nelts: Int](i: Int):
                    node_data.simd_store[nelts](
                        i,
                        node_data.simd_load[nelts](i)
                        - fw_vec[nelts, learning_rate](node_grad.simd_load[nelts](i)),
                    )

                vectorize_unroll[nelts, 1, vectorized_update](node.load_cap())
