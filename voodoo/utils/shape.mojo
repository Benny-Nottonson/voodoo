from voodoo import Node
from math import max


@always_inline("nodebug")
fn shape(*shapes: Int) -> DynamicVector[Int]:
    var shape = DynamicVector[Int]()
    for i in range(len(shapes)):
        shape.push_back(shapes[i])
    return shape


fn get_broadcasted_shape_for_ew_op(parent1: Node, parent2: Node) -> DynamicVector[Int]:
    var shape = DynamicVector[Int]()
    let target = parent1 if parent1.num_dims - parent2.num_dims > 0 else parent2
    for i in range(max(parent1.num_dims, parent2.num_dims)):
        shape.push_back(target.shape.load(i))
    return shape
