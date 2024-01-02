from math import abs, log, max
from algorithm import vectorize
from voodoo import Node
from .shared import DType_F32, nelts, epsilon


trait Loss:
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        ...

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        ...


struct MAE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_mae[nelts: Int](i: Int):
            let error = abs(y_pred.load_data[nelts](i) - y_true.load_data[nelts](i))
            sum += error.reduce_add()

        vectorize[nelts, vectorized_mae](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_mae_bw[nelts: Int](i: Int):
            let grad = (y_pred.load_data[nelts](i) - y_true.load_data[nelts](i)) / (
                y_pred.load_data[nelts](i) - y_true.load_data[nelts](i)
            )
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_mae_bw](y_pred.load_cap())


struct MAPE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_mape[nelts: Int](i: Int):
            let error = abs(
                abs(y_true.load_data[nelts](i) - y_pred.load_data[nelts](i))
                / (abs(y_true.load_data[nelts](i)) + epsilon)
            ) * 10
            sum += error.reduce_add()

        vectorize[nelts, vectorized_mape](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_mape_bw[nelts: Int](i: Int):
            let grad = (
                abs(y_true.load_data[nelts](i) - y_pred.load_data[nelts](i))
                / (abs(y_true.load_data[nelts](i)) + epsilon)
            ) * 10
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_mape_bw](y_pred.load_cap())


struct MSE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_mse[nelts: Int](i: Int):
            let error = (y_pred.load_data[nelts](i) - y_true.load_data[nelts](i)) ** 2
            sum += error.reduce_add()

        vectorize[nelts, vectorized_mse](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_mse_bw[nelts: Int](i: Int):
            let grad = -2.0 * (
                y_true.load_data[nelts](i) - y_pred.load_data[nelts](i)
            ) / Float32(y_pred.load_cap())
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_mse_bw](y_pred.load_cap())


struct MSLE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_msle[nelts: Int](i: Int):
            let diff = log(max(y_true.load_data[nelts](i), 0) + 1.0) - log(
                max(y_pred.load_data[nelts](i), 0) + 1.0
            )
            let error = diff * diff
            sum += error.reduce_add()

        vectorize[nelts, vectorized_msle](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_msle_bw[nelts: Int](i: Int):
            let diff = log(max(y_true.load_data[nelts](i), 0) + 1.0) - log(
                max(y_pred.load_data[nelts](i), 0) + 1.0
            )
            let grad = 2.0 * diff / Float32(y_pred.load_cap())
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_msle_bw](y_pred.load_cap())


struct BCE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_bce[nelts: Int](i: Int):
            let error = -y_true.load_data[nelts](i) * log(
                abs(y_pred.load_data[nelts](i))
            )
            sum += error.reduce_add()

        vectorize[nelts, vectorized_bce](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_bce_bw[nelts: Int](i: Int):
            let grad = -y_true.load_data[nelts](i) / y_pred.load_data[nelts](i)
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_bce_bw](y_pred.load_cap())


struct CCE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_cce[nelts: Int](i: Int):
            let error = -y_true.load_data[nelts](i) * log(
                abs(y_pred.load_data[nelts](i))
            )
            sum += error.reduce_add()

        vectorize[nelts, vectorized_cce](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_cce_bw[nelts: Int](i: Int):
            let grad = -y_true.load_data[nelts](i) / y_pred.load_data[nelts](i)
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_cce_bw](y_pred.load_cap())


struct CFCE(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_cfce[nelts: Int](i: Int):
            let error = (
                -y_true.load_data[nelts](i)
                * log(abs(y_pred.load_data[nelts](i)))
                * (1.0 - y_pred.load_data[nelts](i))
                * (1.0 - y_pred.load_data[nelts](i))
            ) / y_pred.load_data[nelts](i)
            sum += error.reduce_add()

        vectorize[nelts, vectorized_cfce](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_cfce_bw[nelts: Int](i: Int):
            let grad = (
                -y_true.load_data[nelts](i)
                * log(abs(y_pred.load_data[nelts](i)))
                * (1.0 - y_pred.load_data[nelts](i))
                * (1.0 - y_pred.load_data[nelts](i))
            ) / y_pred.load_data[nelts](i)
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_cfce_bw](y_pred.load_cap())


struct CS(Loss):
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        var sum: Float32 = 0.0

        @parameter
        fn vectorized_cs[nelts: Int](i: Int):
            let error = (
                y_pred.load_data[nelts](i) * y_true.load_data[nelts](i)
            ).reduce_add()
            sum += error.reduce_add()

        vectorize[nelts, vectorized_cs](y_pred.load_cap())
        node.store_data(0, sum / Float32(y_pred.load_cap()))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        @parameter
        fn vectorized_cs_bw[nelts: Int](i: Int):
            let grad = (
                y_true.load_data[nelts](i) * y_pred.load_data[nelts](i)
            ).reduce_add()
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) + grad)

        vectorize[nelts, vectorized_cs_bw](y_pred.load_cap())
