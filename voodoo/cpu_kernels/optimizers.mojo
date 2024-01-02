from math import sqrt
from algorithm import vectorize
from voodoo import Node, Vector
from .shared import DType_F32, nelts


trait Optimizer:
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        ...


struct SGD(Optimizer):
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn v_sgd_update[_nelts: Int](i: Int):
                    node.store_data[_nelts](
                        i,
                        node.load_data[_nelts](i)
                        - node.load_grad[_nelts](i) * learning_rate,
                    )

                vectorize[nelts, v_sgd_update](node.load_cap())


struct Adafactor(Optimizer):
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn v_adafactor_update[_nelts: Int](i: Int):
                    let grad = node.load_grad[_nelts](i)
                    let data = node.load_data[_nelts](i)
                    let grad_sq = grad * grad
                    let data_sq = data * data
                    let new_grad_sq = grad_sq + 1e-30
                    let new_data_sq = data_sq + 1e-30
                    let decay_rate = 0.8
                    let lr = learning_rate
                    let param_scale = 1.0
                    let grad_scale = 1.0
                    let old_scale = sqrt(new_data_sq) / sqrt(new_grad_sq)
                    let new_scale = old_scale * param_scale
                    let update_scale = old_scale * grad_scale
                    let new_data = data - lr * grad * update_scale
                    let new_grad_sq_2 = new_grad_sq * decay_rate + new_data_sq * (
                        1.0 - decay_rate
                    )
                    let new_grad = sqrt(new_grad_sq_2) / param_scale * grad
                    node.store_data[_nelts](i, new_data)
                    node.store_grad[_nelts](i, new_grad)

                vectorize[nelts, v_adafactor_update](node.load_cap())


struct Adam(Optimizer):
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()

            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn v_adam_update[_nelts: Int](i: Int):
                    let grad = node.load_grad[_nelts](i)
                    let data = node.load_data[_nelts](i)
                    let grad_sq = grad * grad
                    let data_sq = data * data
                    let new_grad_sq = grad_sq + 1e-30
                    let new_data_sq = data_sq + 1e-30
                    let decay_rate = 0.8
                    let lr = learning_rate
                    let param_scale = 1.0
                    let grad_scale = 1.0
                    let old_scale = sqrt(new_data_sq) / sqrt(new_grad_sq)
                    let new_scale = old_scale * param_scale
                    let update_scale = old_scale * grad_scale
                    let new_data = data - lr * grad * update_scale
                    let new_grad_sq_2 = new_grad_sq * decay_rate + new_data_sq * (
                        1.0 - decay_rate
                    )
                    let new_grad = sqrt(new_grad_sq_2) / param_scale * grad
                    node.store_data[_nelts](i, new_data)
                    node.store_grad[_nelts](i, new_grad)

                vectorize[nelts, v_adam_update](node.load_cap())
