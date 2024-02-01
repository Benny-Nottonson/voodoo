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


fn recursive_broadcast[
    kernel: fn (
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None,
    base_case: fn (depth: Int, a: Node, b: Node) -> Bool,
](
    c: Node,
    a: Node,
    b: Node,
    a_index: Int = 0,
    b_index: Int = 0,
    c_index: Int = 0,
    depth: Int = 0,
):
    if base_case(depth, a, b):
        kernel(c, a, b, a_index, b_index, c_index, depth)
        return

    let a_shape = shape_a(depth, a, b)
    let b_shape = shape_b(depth, a, b)
    let c_shape = c.get_shape()[depth]

    let scaled_a_index = a_index * a_shape
    let scaled_b_index = b_index * b_shape

    for s in range(max(a_shape, b_shape)):
        recursive_broadcast[kernel, base_case](
            c,
            a,
            b,
            scaled_a_index + s if a_shape != 1 else a_index,
            scaled_b_index + s if b_shape != 1 else b_index,
            c_shape * c_index + s,
            depth + 1,
        )


fn recursive_broadcast_bw[
    kernel: fn (
        c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
    ) -> None,
    base_case: fn (depth: Int, a: Node, b: Node) -> Bool,
](
    c: Node,
    a: Node,
    b: Node,
    a_index: Int = 0,
    b_index: Int = 0,
    c_index: Int = 0,
    depth: Int = 0,
):
    if base_case(depth, a, b):
        kernel(c, a, b, a_index, b_index, c_index, depth)
        return

    let a_shape = shape_a(depth, a, b)
    let b_shape = shape_b(depth, a, b)
    let c_shape = c.get_shape()[depth]

    let scaled_a_index = a_index * a_shape
    let scaled_b_index = b_index * b_shape

    for s in range(max(a_shape, b_shape)):
        recursive_broadcast_bw[kernel, base_case](
            c,
            a,
            b,
            scaled_a_index + s if a_shape != 1 else a_index,
            scaled_b_index + s if b_shape != 1 else b_index,
            c_shape * c_index + s,
            depth + 1,
        )
