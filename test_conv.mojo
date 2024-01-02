from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    let pool = Layer[type="maxpool2d", in_neurons=4, out_neurons=4]()

    let input = Tensor(shape(4, 4)).initialize["random_normal"]()
    
    input.print()
    let x = pool.forward(input)
    x.print()
