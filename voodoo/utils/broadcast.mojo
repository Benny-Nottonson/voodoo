from math import max
from voodoo import Node


@always_inline
fn shape_a(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(b.num_dims_ptr.load() - a.num_dims_ptr.load(), 0)
    if depth < diff:
        return 1
    return a.shape_ptr.load().load(depth - diff)


@always_inline
fn shape_b(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(a.num_dims_ptr.load() - b.num_dims_ptr.load(), 0)
    if depth < diff:
        return 1
    return b.shape_ptr.load().load(depth - diff)


@always_inline
fn strides_a(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(b.num_dims_ptr.load() - a.num_dims_ptr.load(), 0)
    if depth < diff:
        return a.strides_ptr.load().load(0)
    return a.strides_ptr.load().load(depth - diff)


@always_inline
fn strides_b(depth: Int, a: Node, b: Node) -> Int:
    let diff = max(a.num_dims_ptr.load() - b.num_dims_ptr.load(), 0)
    if depth < diff:
        return b.strides_ptr.load().load(0)
    return b.strides_ptr.load().load(depth - diff)


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
    let c_shape = c.shape_ptr.load().load(depth)
    if a_shape != 1 and b_shape == 1:
        for s in range(a_shape):
            recursive_broadcast[kernel, base_case](
                c,
                a,
                b,
                a_shape * a_index + s,
                b_shape * b_index,
                c_shape * c_index + s,
                depth + 1,
            )
    elif a_shape == 1 and b_shape != 1:
        for s in range(b_shape):
            recursive_broadcast[kernel, base_case](
                c,
                a,
                b,
                a_shape * a_index,
                b_shape * b_index + s,
                c_shape * c_index + s,
                depth + 1,
            )
    else:
        for s in range(a_shape):
            recursive_broadcast[kernel, base_case](
                c,
                a,
                b,
                a_shape * a_index + s,
                b_shape * b_index + s,
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
    let c_shape = c.shape_ptr.load().load(depth)
    if a_shape != 1 and b_shape == 1:
        for s in range(a_shape):
            recursive_broadcast_bw[kernel, base_case](
                c,
                a,
                b,
                a_shape * a_index + s,
                b_shape * b_index,
                c_shape * c_index + s,
                depth + 1,
            )
    elif a_shape == 1 and b_shape != 1:
        for s in range(b_shape):
            recursive_broadcast_bw[kernel, base_case](
                c,
                a,
                b,
                a_shape * a_index,
                b_shape * b_index + s,
                c_shape * c_index + s,
                depth + 1,
            )
    else:
        for s in range(a_shape):
            recursive_broadcast_bw[kernel, base_case](
                c,
                a,
                b,
                a_shape * a_index + s,
                b_shape * b_index + s,
                c_shape * c_index + s,
                depth + 1,
            )
