from math import max
from voodoo import Node


@always_inline("nodebug")
fn shape_a(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(b.get_num_dims() - a.get_num_dims(), 0)
    return a.get_shape()[depth - diff] if depth >= diff else 1


@always_inline("nodebug")
fn shape_b(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(a.get_num_dims() - b.get_num_dims(), 0)
    return b.get_shape()[depth - diff] if depth >= diff else 1


@always_inline("nodebug")
fn strides_a(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(b.get_num_dims() - a.get_num_dims(), 0)
    return a.get_strides()[depth - diff] if depth >= diff else a.get_strides()[0]


@always_inline("nodebug")
fn strides_b(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(a.get_num_dims() - b.get_num_dims(), 0)
    return b.get_strides()[depth - diff] if depth >= diff else b.get_strides()[0]


@always_inline("nodebug")
fn get_broadcasted_shape_for_ew_op(parent1: Node, parent2: Node) -> DynamicVector[Int]:
    var shape = DynamicVector[Int]()
    let target = parent1 if parent1.get_num_dims() - parent2.get_num_dims() > 0 else parent2
    for i in range(target.get_num_dims()):
        shape.push_back(target.get_shape()[i])
    return shape


@always_inline("nodebug")
fn base_case[use_strides: Bool](depth: Int, a: Node, b: Node) -> Bool:
    @parameter
    if use_strides:
        return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
            depth, a, b
        ) * shape_b(depth, a, b)
    else:
        return depth == max(a.get_num_dims(), b.get_num_dims()) - 2


fn recursive_broadcast[
    kernel: fn (
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None,
    use_strides: Bool,
](
    c: Node,
    a: Node,
    b: Node,
    a_index: Int = 0,
    b_index: Int = 0,
    c_index: Int = 0,
    depth: Int = 0,
):
    var stack = Vector[StaticIntTuple[4]]()
    stack.push_back(StaticIntTuple[4](a_index, b_index, c_index, depth))

    while len(stack) > 0:
        let item = stack.pop_back()

        let item_a_index = item[0]
        let item_b_index = item[1]
        let item_c_index = item[2]
        let item_depth = item[3]

        if base_case[use_strides](item_depth, a, b):
            kernel(c, a, b, item_a_index, item_b_index, item_c_index, item_depth)
            continue

        let a_shape = shape_a(item_depth, a, b)
        let b_shape = shape_b(item_depth, a, b)
        let c_shape_indexed = c.get_shape()[item_depth] * item_c_index

        let scaled_a_index = item_a_index * a_shape
        let scaled_b_index = item_b_index * b_shape
        let max_shape = max(a_shape, b_shape)

        let a_step = 0 if a_shape == 1 else 1
        let b_step = 0 if b_shape == 1 else 1
        let new_depth = item_depth + 1

        for s in range(max_shape):
            stack.push_back(
                StaticIntTuple[4](
                    scaled_a_index + s * a_step,
                    scaled_b_index + s * b_step,
                    c_shape_indexed + s,
                    new_depth,
                )
            )
