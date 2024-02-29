from math import max

from voodoo.autograd import Node


fn shape_a(depth: Int, a: Node, b: Node) -> Int:
    var diff = max(b.get_num_dims() - a.get_num_dims(), 0)
    return a.get_shape()[depth - diff] if depth >= diff else 1


fn shape_b(depth: Int, a: Node, b: Node) -> Int:
    var diff = max(a.get_num_dims() - b.get_num_dims(), 0)
    return b.get_shape()[depth - diff] if depth >= diff else 1


fn strides_a(depth: Int, a: Node, b: Node) -> Int:
    var diff = max(b.get_num_dims() - a.get_num_dims(), 0)
    return a.get_strides()[depth - diff] if depth >= diff else a.get_strides()[0]


fn strides_b(depth: Int, a: Node, b: Node) -> Int:
    var diff = max(a.get_num_dims() - b.get_num_dims(), 0)
    return b.get_strides()[depth - diff] if depth >= diff else b.get_strides()[0]


fn get_broadcasted_shape_for_ew_op(parent1: Node, parent2: Node) -> Vector[Int]:
    var shape = Vector[Int]()
    var target = parent1 if parent1.get_num_dims() - parent2.get_num_dims() > 0 else parent2
    for i in range(target.get_num_dims()):
        shape.push_back(target.get_shape()[i])
    return shape


fn base_case[
    use_strides: Bool
](depth: Int, a: Node, b: Node, a_b_diff: Int, b_a_diff: Int) -> Bool:
    @parameter
    if use_strides:
        return (
            a.get_strides()[depth - b_a_diff] if depth
            >= b_a_diff else a.get_strides()[0]
        ) * (a.get_shape()[depth - b_a_diff] if depth >= b_a_diff else 1) == (
            b.get_strides()[depth - a_b_diff] if depth
            >= a_b_diff else b.get_strides()[0]
        ) * (
            b.get_shape()[depth - a_b_diff] if depth >= a_b_diff else 1
        )
    else:
        return depth == max(a.get_num_dims(), b.get_num_dims()) - 2


fn precompute_broadcasted_shape(
    diff: Int, shape: Pointer[Int], num_dims: Int
) -> Pointer[Int]:
    var precomputed_shape = Pointer[Int].alloc(num_dims)
    for i in range(num_dims):
        precomputed_shape[i] = 1 if i < diff else shape[i - diff]
    return precomputed_shape


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
    var stack = Vector[Tuple[Int, Int, Int, Int]]()
    stack.push_back((a_index, b_index, c_index, depth))

    var a_b_diff = max(a.get_num_dims() - b.get_num_dims(), 0)
    var b_a_diff = max(b.get_num_dims() - a.get_num_dims(), 0)
    var a_shape = a.get_shape()._data
    var b_shape = b.get_shape()._data
    var c_shape = c.get_shape()._data

    var a_shape_precomputed = precompute_broadcasted_shape(
        b_a_diff, a_shape, a.get_num_dims()
    )
    var b_shape_precomputed = precompute_broadcasted_shape(
        a_b_diff, b_shape, b.get_num_dims()
    )

    while len(stack) > 0:
        var item = stack.pop_back()

        var item_a_index = item.get[0, Int]()
        var item_b_index = item.get[1, Int]()
        var item_c_index = item.get[2, Int]()
        var item_depth = item.get[3, Int]()

        if base_case[use_strides](item_depth, a, b, a_b_diff, b_a_diff):
            kernel(c, a, b, item_a_index, item_b_index, item_c_index, item_depth)
            continue

        var a_shape = a_shape_precomputed[item_depth]
        var b_shape = b_shape_precomputed[item_depth]
        var c_shape_indexed = c_shape[item_depth] * item_c_index

        var scaled_a_index = item_a_index * a_shape
        var scaled_b_index = item_b_index * b_shape
        var max_shape = max(a_shape, b_shape)

        var a_step = 0 if a_shape == 1 else 1
        var b_step = 0 if b_shape == 1 else 1
        var new_depth = item_depth + 1

        for s in range(max_shape):
            stack.push_back(
                (
                    scaled_a_index + s * a_step,
                    scaled_b_index + s * b_step,
                    c_shape_indexed + s,
                    new_depth,
                )
            )

    a_shape_precomputed.free()
    b_shape_precomputed.free()
    stack.free()
