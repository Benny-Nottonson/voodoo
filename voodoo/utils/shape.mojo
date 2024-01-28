from voodoo import Node
from math import max, abs


@always_inline
fn shape(*shapes: Int) -> DynamicVector[Int]:
    var shape = DynamicVector[Int]()
    for i in range(len(shapes)):
        shape.push_back(shapes[i])
    return shape


fn get_broadcasted_shape_for_ew_op(parent1: Node, parent2: Node) -> DynamicVector[Int]:
    let new_num_dims = max(parent1.num_dims, parent2.num_dims)
    var shape = DynamicVector[Int]()
    let diff = parent1.num_dims - parent2.num_dims
    for i in range(new_num_dims):
        if diff > 0 and i < abs(diff):
            shape.push_back(parent1.shape.load(i))
        elif diff < 0 and i < abs(diff):
            shape.push_back(parent2.shape.load(i))
        else:
            if diff > 0:
                shape.push_back(parent1.shape.load(i))
            else:
                shape.push_back(parent2.shape.load(i))
    return shape
