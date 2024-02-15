from algorithm import vectorize

from voodoo.constants import NELTS, PREFETCH_READ, PREFETCH_WRITE
from voodoo.autograd import Node
from voodoo.utils import Vector


trait Optimizer(CollectionElement):
    @staticmethod
    fn step(x: Vector[Node]):
        ...

    @staticmethod
    fn key() -> String:
        ...


@register_passable("trivial")
struct SGD[learning_rate: Float32](Optimizer):
    @staticmethod
    fn step(x: Vector[Node]):
        for i in range(len(x)):
            let node = x[i]
            let node_data = node.get_data()
            let node_grad = node.get_grad()
            if node.get_is_static() and node.get_grad_id() != 0:
                DTypePointer.prefetch[PREFETCH_READ](node_data)
                DTypePointer.prefetch[PREFETCH_READ](node.get_grad())
                DTypePointer.prefetch[PREFETCH_WRITE](node_data)

                @parameter
                fn vectorized_update[NELTS: Int](i: Int):
                    node_data.simd_store[NELTS](
                        i,
                        node_data.simd_load[NELTS](i)
                        - (node_grad.simd_load[NELTS](i) * learning_rate),
                    )

                vectorize[NELTS, vectorized_update](node.get_cap())

    @staticmethod
    fn key() -> String:
        return "SGD"
