from math import sqrt
from algorithm import vectorize
from voodoo import Node, Vector
from ..constants import DType_F32, nelts

alias generic_vectorized = fn[nelts: Int, learning_rate: Float32] (SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]

struct Generic[fw_vec: generic_vectorized]:
    @staticmethod
    fn step[learning_rate: Float32](x_ptr: Pointer[Vector[Pointer[Node, 0]], 0]) raises:
        let x = x_ptr.load()
        for i in range(x.len.load()):
            let node = x.load(i).load()
            if node.requires_grad_ptr.load() and node.grad_computed_ptr.load():

                @parameter
                fn vectorized_update[nelts: Int](i: Int):
                    node.store_data[nelts](
                        i,
                        node.load_data[nelts](i) - 
                        fw_vec[nelts, learning_rate](node.load_grad[nelts](i)),
                    )

                vectorize[nelts, vectorized_update](node.load_cap())

alias SGD = Generic[sgd]

@parameter
@always_inline
fn sgd[nelts: Int, learning_rate: Float32] (grad: SIMD[DType_F32, nelts]) -> SIMD[DType_F32, nelts]:
    return grad * learning_rate