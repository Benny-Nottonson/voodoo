from math import sqrt
from algorithm import vectorize
from voodoo import Node, Vector
from ..constants import DType_F32, nelts


trait Optimizer:
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        ...


# TODO: Rewrite to use generic functions where possible


struct SGD(Optimizer):
    @staticmethod
    @always_inline
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn vectorized_sgd_update[nelts: Int](i: Int):
                    node.store_data[nelts](
                        i,
                        node.load_data[nelts](i)
                        - node.load_grad[nelts](i) * learning_rate,
                    )

                vectorize[nelts, vectorized_sgd_update](node.load_cap())


struct Adafactor(Optimizer):
    @staticmethod
    @always_inline
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn vectorized_adafactor_update[nelts: Int](i: Int):
                    let grad = node.load_grad[nelts](i)
                    let data = node.load_data[nelts](i)
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
                    node.store_data[nelts](i, new_data)
                    node.store_grad[nelts](i, new_grad)

                vectorize[nelts, vectorized_adafactor_update](node.load_cap())


struct Adam(Optimizer):
    @staticmethod
    @always_inline
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()

            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn vectorized_adam_update[nelts: Int](i: Int):
                    let grad = node.load_grad[nelts](i)
                    let data = node.load_data[nelts](i)
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
                    node.store_data[nelts](i, new_data)
                    node.store_grad[nelts](i, new_grad)

                vectorize[nelts, vectorized_adam_update](node.load_cap())
