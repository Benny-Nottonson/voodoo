from math import max
from voodoo import Node


@always_inline("nodebug")
fn shape_a(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(b.num_dims_ptr.load() - a.num_dims_ptr.load(), 0)
    return a.shape.load(depth - diff) if depth >= diff else 1


@always_inline("nodebug")
fn shape_b(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(a.num_dims_ptr.load() - b.num_dims_ptr.load(), 0)
    return b.shape.load(depth - diff) if depth >= diff else 1


@always_inline("nodebug")
fn strides_a(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(b.num_dims_ptr.load() - a.num_dims_ptr.load(), 0)
    return a.strides.load(depth - diff) if depth >= diff else a.strides.load(0)


@always_inline("nodebug")
fn strides_b(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(a.num_dims_ptr.load() - b.num_dims_ptr.load(), 0)
    return b.strides.load(depth - diff) if depth >= diff else b.strides.load(0)


@always_inline("nodebug")
fn get_broadcasted_shape_for_ew_op(parent1: Node, parent2: Node) -> DynamicVector[Int]:
    var shape = DynamicVector[Int]()
    let target = parent1 if parent1.num_dims_ptr.load() - parent2.num_dims_ptr.load() > 0 else parent2
    for i in range(max(parent1.num_dims_ptr.load(), parent2.num_dims_ptr.load())):
        shape.push_back(target.shape.load(i))
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
    let c_shape = c.shape.load(depth)

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
    let c_shape = c.shape.load(depth)

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
