from math import (
    abs,
    exp,
    log,
    tanh,
    cosh,
)
from algorithm import vectorize

from voodoo import Node

alias nelts = simdwidthof[DType.float32]()


fn fw_kld(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_kld[nelts: Int](i: Int):
        let error = parent1.load_data[nelts](i) * log(
            parent1.load_data[nelts](i) / parent2.load_data[nelts](i)
        )
        sum += error.reduce_add()

    vectorize[nelts, v_kld](parent1.load_cap())
    node.store_data(0, sum)


fn bw_kld(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_kld_bw[nelts: Int](i: Int):
        let grad = parent1.load_data[nelts](i) * (
            Float32(1.0)
            + log(parent1.load_data[nelts](i) / parent2.load_data[nelts](i))
        )
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_kld_bw](parent1.load_cap())


fn fw_mae(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_mae[nelts: Int](i: Int):
        let error = abs(parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
        sum += error.reduce_add()

    vectorize[nelts, v_mae](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_mae(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_mae_bw[nelts: Int](i: Int):
        let grad = (parent1.load_data[nelts](i) - parent2.load_data[nelts](i)) / (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        )
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_mae_bw](parent1.load_cap())


fn fw_mape(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_mape[nelts: Int](i: Int):
        let error = abs(
            (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
            / parent2.load_data[nelts](i)
        )
        sum += error.reduce_add()

    vectorize[nelts, v_mape](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_mape(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_mape_bw[nelts: Int](i: Int):
        let grad = (parent1.load_data[nelts](i) - parent2.load_data[nelts](i)) / (
            parent2.load_data[nelts](i) * parent2.load_data[nelts](i)
        )
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_mape_bw](parent1.load_cap())


fn fw_mse(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_mse[nelts: Int](i: Int):
        let error = (parent1.load_data[nelts](i) - parent2.load_data[nelts](i)) * (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        )
        sum += error.reduce_add()

    vectorize[nelts, v_mse](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_mse(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_mse_bw[nelts: Int](i: Int):
        let grad = -Float32(2.0) * (
            parent2.load_data[nelts](i) - parent1.load_data[nelts](i)
        ) / Float32(parent1.load_cap())
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_mse_bw](parent1.load_cap())


fn fw_msle(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_msle[nelts: Int](i: Int):
        let error = log(
            Float32(1.0)
            + (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
            * (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
        )
        sum += error.reduce_add()

    vectorize[nelts, v_msle](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_msle(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_msle_bw[nelts: Int](i: Int):
        let grad = (
            Float32(1.0)
            + (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
            * (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
        ) / (
            Float32(1.0)
            + (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
            * (parent1.load_data[nelts](i) - parent2.load_data[nelts](i))
        )
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_msle_bw](parent1.load_cap())


fn fw_bce(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_bce[nelts: Int](i: Int):
        let error = -(
            parent2.load_data[nelts](i) * log(parent1.load_data[nelts](i))
            + (Float32(1.0) - parent2.load_data[nelts](i))
            * log(Float32(1.0) - parent1.load_data[nelts](i))
        )
        sum += error.reduce_add()

    vectorize[nelts, v_bce](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_bce(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_bce_bw[nelts: Int](i: Int):
        let grad = (
            (Float32(1.0) - parent2.load_data[nelts](i))
            / (Float32(1.0) - parent1.load_data[nelts](i))
            - parent2.load_data[nelts](i) / parent1.load_data[nelts](i)
        )
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_bce_bw](parent1.load_cap())


fn fw_cce(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_cce[nelts: Int](i: Int):
        let error = -parent2.load_data[nelts](i) * log(parent1.load_data[nelts](i))
        sum += error.reduce_add()

    vectorize[nelts, v_cce](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_cce(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_cce_bw[nelts: Int](i: Int):
        let grad = -parent2.load_data[nelts](i) / parent1.load_data[nelts](i)
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_cce_bw](parent1.load_cap())


fn fw_cfce(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_cfce[nelts: Int](i: Int):
        let error = -parent2.load_data[nelts](i) * (
            Float32(1.0) - parent1.load_data[nelts](i)
        ) ** Float32(2.0) * log(parent1.load_data[nelts](i))
        sum += error.reduce_add()

    vectorize[nelts, v_cfce](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_cfce(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_cfce_bw[nelts: Int](i: Int):
        let grad = (
            -parent2.load_data[nelts](i)
            * (Float32(1.0) - parent1.load_data[nelts](i))
            * (Float32(1.0) - Float32(2.0) * parent1.load_data[nelts](i))
        ) / parent1.load_data[nelts](i)
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_cfce_bw](parent1.load_cap())


fn fw_cs(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_cs[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) * parent2.load_data[nelts](i)
        ).reduce_add()
        sum += error.reduce_add()

    vectorize[nelts, v_cs](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_cs(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_cs_bw[nelts: Int](i: Int):
        let grad = (
            parent2.load_data[nelts](i) * parent1.load_data[nelts](i)
        ).reduce_add()
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) + grad)

    vectorize[nelts, v_cs_bw](parent1.load_cap())


fn fw_huber(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_huber[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        ).reduce_add()
        if error < Float32(1.0):
            sum += error * error
        else:
            sum += Float32(2.0) * error - Float32(1.0)

    vectorize[nelts, v_huber](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_huber(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_huber_bw[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        ).reduce_add()
        let grad = Float32(0.0)
        if error < Float32(1.0):
            let grad = Float32(2.0) * error
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_huber_bw](parent1.load_cap())


fn fw_logcosh(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_logcosh[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        ).reduce_add()
        sum += log(cosh(error))

    vectorize[nelts, v_logcosh](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_logcosh(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_logcosh_bw[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        ).reduce_add()
        let grad = tanh(error)
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_logcosh_bw](parent1.load_cap())


fn fw_poisson(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_poisson[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        ).reduce_add()
        sum += exp(error)

    vectorize[nelts, v_poisson](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_poisson(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_poisson_bw[nelts: Int](i: Int):
        let error = (
            parent1.load_data[nelts](i) - parent2.load_data[nelts](i)
        ).reduce_add()
        let grad = exp(error)
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_poisson_bw](parent1.load_cap())


fn fw_scce(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_scce[nelts: Int](i: Int):
        let error = -parent2.load_data[nelts](i) * log(parent1.load_data[nelts](i))
        sum += error.reduce_add()

    vectorize[nelts, v_scce](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_scce(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_scce_bw[nelts: Int](i: Int):
        let grad = -parent2.load_data[nelts](i) / parent1.load_data[nelts](i)
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_scce_bw](parent1.load_cap())


fn fw_sce(node: Node, parent1: Node, parent2: Node):
    var sum = Float32(0.0)

    @parameter
    fn v_sce[nelts: Int](i: Int):
        let error = -parent2.load_data[nelts](i) * log(parent1.load_data[nelts](i))
        sum += error.reduce_add()

    vectorize[nelts, v_sce](parent1.load_cap())
    node.store_data(0, sum / Float32(parent1.load_cap()))


fn bw_sce(node: Node, parent1: Node, parent2: Node):
    @parameter
    fn v_sce_bw[nelts: Int](i: Int):
        let grad = -parent2.load_data[nelts](i) / parent1.load_data[nelts](i)
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + grad)
        parent2.store_grad[nelts](i, parent2.load_grad[nelts](i) - grad)

    vectorize[nelts, v_sce_bw](parent1.load_cap())
