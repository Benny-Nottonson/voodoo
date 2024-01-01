from math import (
    max,
    sqrt,
    exp2,
    log2,
    log,
    cos,
    sin,
    tan,
    asin,
    acos,
    atan,
    cosh,
    sinh,
)
from random import random_float64, rand
from sys.param_env import env_get_int
from algorithm import vectorize, parallelize
from voodoo.utils import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)

from voodoo import Node

alias DType_F32 = DType.float32
alias nelts = simdwidthof[DType_F32]()
alias workers = env_get_int["WORKERS", 0]()


@parameter
fn base_case_add_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_add_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) + b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_add](c_rest)


fn fw_add(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_add_fw, base_case_add_fw](c, a, b)


@parameter
fn base_case_add_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_add_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_add_grad_a](c_rest)


@parameter
fn kernel_add_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i) + c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_add_grad_b](c_rest)


fn bw_add(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_add_bw_a, base_case_add_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_add_bw_b, base_case_add_bw](c, a, b)


@parameter
fn base_case_mul_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_mul_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) * b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_mul](c_rest)


fn fw_mul(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_mul_fw, base_case_mul_fw](c, a, b)


@parameter
fn base_case_mul_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_mul_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + b.load_data[nelts](offset_b + i) * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_mul_grad_a](c_rest)


@parameter
fn kernel_mul_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            + a.load_data[nelts](offset_a + i) * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_mul_grad_b](c_rest)


fn bw_mul(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_mul_bw_a, base_case_mul_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_mul_bw_b, base_case_mul_bw](c, a, b)


@parameter
fn base_case_sub_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_sub_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) - b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_sub](c_rest)


fn fw_sub(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_sub_fw, base_case_sub_fw](c, a, b)


@parameter
fn base_case_sub_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_sub_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i) + c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_sub_grad_a](c_rest)


@parameter
fn kernel_sub_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i) - c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_sub_grad_b](c_rest)


fn bw_sub(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_sub_bw_a, base_case_sub_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_sub_bw_b, base_case_sub_bw](c, a, b)


@parameter
fn base_case_div_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_div_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) / b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_div](c_rest)


fn fw_div(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_div_fw, base_case_div_fw](c, a, b)


@parameter
fn base_case_div_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_div_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div_grad_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + c.load_grad[nelts](offset_c + i) / b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_div_grad_a](c_rest)


@parameter
fn kernel_div_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div_grad_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            - a.load_data[nelts](offset_a + i)
            * c.load_grad[nelts](offset_c + i)
            / (b.load_data[nelts](offset_b + i)) ** 2,
        )

    vectorize[nelts, v_div_grad_b](c_rest)


fn bw_div(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_div_bw_a, base_case_div_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_div_bw_b, base_case_div_bw](c, a, b)


@parameter
fn base_case_matmul_fw(depth: Int, a: Node, b: Node) -> Bool:
    return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2


@parameter
fn kernel_matmul_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * a.shape_ptr.load().load(
        a.num_dims_ptr.load() - 2
    ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
    let offset_b = b_index * b.shape_ptr.load().load(
        b.num_dims_ptr.load() - 2
    ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
    let offset_c = c_index * c.shape_ptr.load().load(
        c.num_dims_ptr.load() - 2
    ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

    let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
    let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
    let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

    @parameter
    fn calc_row_fw(m: Int):
        for k in range(K):

            @parameter
            fn dot_fw[_nelts: Int](n: Int):
                c.store_data[_nelts](
                    offset_c + m * N + n,
                    c.load_data[_nelts](offset_c + m * N + n)
                    + a.load_data(offset_a + m * K + k)
                    * b.load_data[_nelts](offset_b + k * N + n),
                )

            vectorize[nelts, dot_fw](N)

    parallelize[calc_row_fw](M, workers if workers > 0 else M)


fn fw_mmul(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_matmul_fw, base_case_matmul_fw](c, a, b)


@parameter
fn base_case_matmul_bw(depth: Int, a: Node, b: Node) -> Bool:
    return depth == max(a.num_dims_ptr.load(), b.num_dims_ptr.load()) - 2


@parameter
fn kernel_matmul_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * a.shape_ptr.load().load(
        a.num_dims_ptr.load() - 2
    ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
    let offset_b = b_index * b.shape_ptr.load().load(
        b.num_dims_ptr.load() - 2
    ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
    let offset_c = c_index * c.shape_ptr.load().load(
        c.num_dims_ptr.load() - 2
    ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

    let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
    let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
    let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

    @parameter
    fn calc_row_1(m: Int):
        for n in range(N):

            @parameter
            fn dot_bw_a[nelts: Int](k: Int):
                let val = a.load_grad(offset_a + m * K + k) + c.load_grad(
                    offset_c + m * N + n
                ) * b.load_data(offset_b + k * N + n)
                a.store_grad(offset_a + m * K + k, val)

            vectorize[1, dot_bw_a](K)

    parallelize[calc_row_1](M, workers if workers > 0 else M)


@parameter
fn kernel_matmul_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * a.shape_ptr.load().load(
        a.num_dims_ptr.load() - 2
    ) * a.shape_ptr.load().load(a.num_dims_ptr.load() - 1)
    let offset_b = b_index * b.shape_ptr.load().load(
        b.num_dims_ptr.load() - 2
    ) * b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)
    let offset_c = c_index * c.shape_ptr.load().load(
        c.num_dims_ptr.load() - 2
    ) * c.shape_ptr.load().load(c.num_dims_ptr.load() - 1)

    let M = a.shape_ptr.load().load(a.num_dims_ptr.load() - 2)
    let K = b.shape_ptr.load().load(b.num_dims_ptr.load() - 2)
    let N = b.shape_ptr.load().load(b.num_dims_ptr.load() - 1)

    @parameter
    fn calc_row_2(k: Int):
        for m in range(M):

            @parameter
            fn dot_bw_b[nelts: Int](n: Int):
                let val = b.load_grad(offset_b + k * N + n) + a.load_data(
                    offset_a + m * K + k
                ) * c.load_grad(offset_c + m * N + n)
                b.store_grad(offset_b + k * N + n, val)

            vectorize[1, dot_bw_b](N)

    parallelize[calc_row_2](K, workers if workers > 0 else K)


fn bw_mmul(c: Node, a: Node, b: Node):
    if not a.is_single_ptr.load():
        recursive_broadcast_bw[kernel_matmul_bw_a, base_case_matmul_bw](c, a, b)
    if not b.is_single_ptr.load():
        recursive_broadcast_bw[kernel_matmul_bw_b, base_case_matmul_bw](c, a, b)


fn fw_sqrt(node: Node, parent1: Node):
    @parameter
    fn v_sqrt[_nelts: Int](i: Int):
        node.store_data[_nelts](i, sqrt(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_sqrt](node.load_cap())


fn bw_sqrt(node: Node, parent1: Node):
    @parameter
    fn v_sqrt_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) / (2.0 * sqrt(parent1.load_data[_nelts](i))),
        )

    vectorize[nelts, v_sqrt_bw](node.load_cap())


fn fw_abs(node: Node, parent1: Node):
    @parameter
    fn v_abs[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        node.store_data[_nelts](
            i,
            (data >= 0.0).cast[DType_F32]() * data
            + (data < 0.0).cast[DType_F32]() * (-data),
        )

    vectorize[nelts, v_abs](node.load_cap())


fn bw_abs(node: Node, parent1: Node):
    @parameter
    fn v_abs_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + (2.0 * (parent1.load_data[_nelts](i) >= 0.0).cast[DType_F32]() - 1.0)
            * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_abs_bw](node.load_cap())


fn fw_exp2(node: Node, parent1: Node):
    @parameter
    fn v_exp2[_nelts: Int](i: Int):
        node.store_data[_nelts](i, exp2(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_exp2](node.load_cap())


fn bw_exp2(node: Node, parent1: Node):
    @parameter
    fn v_exp2_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * node.load_data[_nelts](i) * 0.69314718056,
        )

    vectorize[nelts, v_exp2_bw](node.load_cap())


fn fw_log2(node: Node, parent1: Node):
    @parameter
    fn v_log2[_nelts: Int](i: Int):
        node.store_data[_nelts](i, log2(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_log2](node.load_cap())


fn bw_log2(node: Node, parent1: Node):
    @parameter
    fn v_log2_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            / (parent1.load_data[_nelts](i) * 0.69314718056),
        )

    vectorize[nelts, v_log2_bw](node.load_cap())


fn fw_log(node: Node, parent1: Node):
    @parameter
    fn v_log[_nelts: Int](i: Int):
        node.store_data[_nelts](i, log(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_log](node.load_cap())


fn bw_log(node: Node, parent1: Node):
    @parameter
    fn v_log_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) / parent1.load_data[_nelts](i),
        )

    vectorize[nelts, v_log_bw](node.load_cap())


@parameter
fn base_case_pow_fw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_pow_fw(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow[nelts: Int](i: Int):
        c.store_data[nelts](
            offset_c + i,
            a.load_data[nelts](offset_a + i) ** b.load_data[nelts](offset_b + i),
        )

    vectorize[nelts, v_pow](c_rest)


fn fw_pow(c: Node, a: Node, b: Node):
    recursive_broadcast[kernel_pow_fw, base_case_pow_fw](c, a, b)


@parameter
fn base_case_pow_bw(depth: Int, a: Node, b: Node) -> Bool:
    return strides_a(depth, a, b) * shape_a(depth, a, b) == strides_b(
        depth, a, b
    ) * shape_b(depth, a, b)


@parameter
fn kernel_pow_bw_a(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow_bw_a[nelts: Int](i: Int):
        a.store_grad[nelts](
            offset_a + i,
            a.load_grad[nelts](offset_a + i)
            + b.load_data[nelts](offset_b + i)
            * (
                a.load_data[nelts](offset_a + i)
                ** (b.load_data[nelts](offset_b + i) - 1.0)
            )
            * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_pow_bw_a](c_rest)


@parameter
fn kernel_pow_bw_b(
    c: Node, a: Node, b: Node, a_index: Int, b_index: Int, c_index: Int, depth: Int
) -> None:
    let offset_a = a_index * shape_a(depth, a, b) * strides_a(depth, a, b)
    let offset_b = b_index * shape_b(depth, a, b) * strides_b(depth, a, b)
    let c_rest = c.shape_ptr.load().load(depth) * c.strides_ptr.load().load(depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow_bw_b[nelts: Int](i: Int):
        b.store_grad[nelts](
            offset_b + i,
            b.load_grad[nelts](offset_b + i)
            + c.load_data[nelts](offset_c + i)
            * log(a.load_data[nelts](offset_a + i))
            * c.load_grad[nelts](offset_c + i),
        )

    vectorize[nelts, v_pow_bw_b](c_rest)


fn bw_pow(c: Node, a: Node, b: Node):
    recursive_broadcast_bw[kernel_pow_bw_a, base_case_pow_bw](c, a, b)
    recursive_broadcast_bw[kernel_pow_bw_b, base_case_pow_bw](c, a, b)


fn fw_sin(node: Node, parent1: Node):
    @parameter
    fn v_sin[_nelts: Int](i: Int):
        node.store_data[_nelts](i, sin(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_sin](node.load_cap())


fn bw_sin(node: Node, parent1: Node):
    @parameter
    fn v_sin_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + cos(parent1.load_data[_nelts](i)) * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_sin_bw](node.load_cap())


fn fw_cos(node: Node, parent1: Node):
    @parameter
    fn v_cos[_nelts: Int](i: Int):
        node.store_data[_nelts](i, cos(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_cos](node.load_cap())


fn bw_cos(node: Node, parent1: Node):
    @parameter
    fn v_cos_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            - sin(parent1.load_data[_nelts](i)) * node.load_grad[_nelts](i),
        )

    vectorize[nelts, v_cos_bw](node.load_cap())


fn fw_tan(node: Node, parent1: Node):
    @parameter
    fn v_tan[_nelts: Int](i: Int):
        node.store_data[_nelts](i, tan(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_tan](node.load_cap())


fn bw_tan(node: Node, parent1: Node):
    @parameter
    fn v_tan_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) / (cos(parent1.load_data[_nelts](i))) ** 2,
        )

    vectorize[nelts, v_tan_bw](node.load_cap())


fn fw_asin(node: Node, parent1: Node):
    @parameter
    fn v_asin[_nelts: Int](i: Int):
        node.store_data[_nelts](i, asin(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_asin](node.load_cap())


fn bw_asin(node: Node, parent1: Node):
    @parameter
    fn v_asin_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i)
            / sqrt(1.0 - (parent1.load_data[_nelts](i)) ** 2),
        )

    vectorize[nelts, v_asin_bw](node.load_cap())


fn fw_acos(node: Node, parent1: Node):
    @parameter
    fn v_acos[_nelts: Int](i: Int):
        node.store_data[_nelts](i, acos(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_acos](node.load_cap())


fn bw_acos(node: Node, parent1: Node):
    @parameter
    fn v_acos_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            - node.load_grad[_nelts](i)
            / sqrt(1.0 - (parent1.load_data[_nelts](i)) ** 2),
        )

    vectorize[nelts, v_acos_bw](node.load_cap())


fn fw_atan(node: Node, parent1: Node):
    @parameter
    fn v_atan[_nelts: Int](i: Int):
        node.store_data[_nelts](i, atan(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_atan](node.load_cap())


fn bw_atan(node: Node, parent1: Node):
    @parameter
    fn v_atan_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) / (1.0 + (parent1.load_data[_nelts](i)) ** 2),
        )

    vectorize[nelts, v_atan_bw](node.load_cap())


fn fw_sinh(node: Node, parent1: Node):
    @parameter
    fn v_sinh[_nelts: Int](i: Int):
        node.store_data[_nelts](i, sinh(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_sinh](node.load_cap())


fn bw_sinh(node: Node, parent1: Node):
    @parameter
    fn v_sinh_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * cosh(parent1.load_data[_nelts](i)),
        )

    vectorize[nelts, v_sinh_bw](node.load_cap())


fn fw_cosh(node: Node, parent1: Node):
    @parameter
    fn v_cosh[_nelts: Int](i: Int):
        node.store_data[_nelts](i, cosh(parent1.load_data[_nelts](i)))

    vectorize[nelts, v_cosh](node.load_cap())


fn bw_cosh(node: Node, parent1: Node):
    @parameter
    fn v_cosh_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](
            i,
            parent1.load_grad[_nelts](i)
            + node.load_grad[_nelts](i) * sinh(parent1.load_data[_nelts](i)),
        )

    vectorize[nelts, v_cosh_bw](node.load_cap())


fn fw_copy(node: Node, parent1: Node):
    @parameter
    fn v_copy[_nelts: Int](i: Int):
        node.store_data[_nelts](i, parent1.load_data[_nelts](i))

    vectorize[nelts, v_copy](node.load_cap())


fn bw_copy(node: Node, parent1: Node):
    @parameter
    fn v_copy_bw[_nelts: Int](i: Int):
        parent1.store_grad[_nelts](i, parent1.load_grad[_nelts](i))

    vectorize[nelts, v_copy_bw](node.load_cap())


fn fw_sum(node: Node, parent1: Node):
    var sum: Float32 = 0.0

    @parameter
    fn v_sum[nelts: Int](i: Int):
        sum += parent1.load_data[nelts](i).reduce_add()

    vectorize[nelts, v_sum](parent1.load_cap())
    node.store_data(0, sum)


fn bw_sum(node: Node, parent1: Node):
    @parameter
    fn v_sum_bw[nelts: Int](i: Int):
        parent1.store_grad[nelts](i, parent1.load_grad[nelts](i) + node.load_grad(0))

    vectorize[nelts, v_sum_bw](parent1.load_cap())


fn fw_reshape(node: Node, parent1: Node):
    for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
        let offset = s * parent1.cap_ptr.load()

        @parameter
        fn v_reshape[nelts: Int](i: Int):
            node.store_data[nelts](i, parent1.load_data[nelts](i))

        vectorize[nelts, v_reshape](parent1.cap_ptr.load())


fn bw_reshape(node: Node, parent1: Node):
    for s in range(node.cap_ptr.load() // parent1.cap_ptr.load()):
        let offset = s * parent1.cap_ptr.load()

        @parameter
        fn v_reshape[nelts: Int](i: Int):
            parent1.store_grad[nelts](
                i, parent1.load_grad[nelts](i) + node.load_grad[nelts](i)
            )

        vectorize[nelts, v_reshape](parent1.cap_ptr.load())


fn fw_transp(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let M = parent1.shape_ptr.load().load(num_dims - 2)
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transp[nelts: Int](j: Int):
                node.store_data[nelts](
                    offset + j * M + i, parent1.load_data[nelts](offset + i * N + j)
                )

            vectorize[nelts, v_transp](N)


fn bw_transp(node: Node, parent1: Node):
    let num_dims = parent1.num_dims_ptr.load()
    let M = parent1.shape_ptr.load().load(num_dims - 2)
    let N = parent1.shape_ptr.load().load(num_dims - 1)
    for s in range(node.cap_ptr.load() // (M * N)):
        let offset = s * M * N
        for i in range(M):

            @parameter
            fn v_transp_bw[nelts: Int](j: Int):
                parent1.store_grad[nelts](
                    offset + j * M + i,
                    parent1.load_grad[nelts](offset + j * M + i)
                    + node.load_grad[nelts](offset + i * N + j),
                )

            vectorize[nelts, v_transp_bw](N)


fn index(
    n: Int, c: Int, h: Int, w: Int, num_channels: Int, width: Int, height: Int
) -> Int:
    return n * (num_channels * height * width) + c * (height * width) + h * width + w


@always_inline
fn conv_2d(c: Node, a: Node, b: Node):
    let padding = c.other_params_ptr.load().load(0)
    let stride = c.other_params_ptr.load().load(1)

    @parameter
    fn batch_loop(i: Int):
        for j in range(b.shape_ptr.load().load(0)):
            for x in range(c.shape_ptr.load().load(2)):
                for y in range(c.shape_ptr.load().load(3)):
                    var patch_sum: Float32 = 0.0
                    for k in range(a.shape_ptr.load().load(1)):
                        for dx in range(b.shape_ptr.load().load(2)):

                            @parameter
                            fn inner_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    let a_index = index(
                                        i,
                                        k,
                                        ix,
                                        iy,
                                        a.shape_ptr.load().load(1),
                                        a.shape_ptr.load().load(2),
                                        a.shape_ptr.load().load(3),
                                    )
                                    let b_index = index(
                                        j,
                                        k,
                                        dx,
                                        dy,
                                        a.shape_ptr.load().load(1),
                                        b.shape_ptr.load().load(2),
                                        b.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        a.load_data[_nelts](a_index)
                                        * b.load_data[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, inner_loop](b.shape_ptr.load().load(3))
                    let c_index = index(
                        i,
                        j,
                        x,
                        y,
                        b.shape_ptr.load().load(0),
                        c.shape_ptr.load().load(2),
                        c.shape_ptr.load().load(3),
                    )
                    c.store_data(c_index, patch_sum)

    parallelize[batch_loop](
        a.shape_ptr.load().load(0),
        workers if workers > 0 else a.shape_ptr.load().load(0),
    )


fn bw_conv_2d(c: Node, a: Node, b: Node):
    let padding = c.other_params_ptr.load().load(0)
    let stride = c.other_params_ptr.load().load(1)

    for i in range(a.shape_ptr.load().load(1)):
        for j in range(b.shape_ptr.load().load(0)):
            for x in range(b.shape_ptr.load().load(2)):
                for y in range(b.shape_ptr.load().load(3)):
                    var patch_sum: Float32 = 0.0
                    for b in range(a.shape_ptr.load().load(0)):
                        for dx in range(c.shape_ptr.load().load(2)):
                            for dy in range(c.shape_ptr.load().load(3)):
                                let ix = x * stride - padding + dx
                                let iy = y * stride - padding + dy
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= a.shape_ptr.load().load(2)
                                    or iy >= a.shape_ptr.load().load(3)
                                ):
                                    let a_index = index(
                                        b,
                                        i,
                                        ix,
                                        iy,
                                        a.shape_ptr.load().load(1),
                                        a.shape_ptr.load().load(2),
                                        a.shape_ptr.load().load(3),
                                    )
                                    let c_grad_index = index(
                                        b,
                                        j,
                                        dx,
                                        dy,
                                        c.shape_ptr.load().load(1),
                                        c.shape_ptr.load().load(2),
                                        c.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        a.load_data(a_index) * c.load_grad(c_grad_index)
                                    ).reduce_add()
                    let b_grad_index = index(
                        i,
                        j,
                        x,
                        y,
                        b.shape_ptr.load().load(0),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    b.store_grad(b_grad_index, patch_sum)

    @parameter
    fn batch_loop(p: Int):
        for j in range(a.shape_ptr.load().load(1)):
            for i in range(b.shape_ptr.load().load(0)):
                for x in range(a.shape_ptr.load().load(2)):
                    for y in range(a.shape_ptr.load().load(3)):
                        var patch_sum: Float32 = 0.0
                        for dx in range(b.shape_ptr.load().load(2)):

                            @parameter
                            fn dy_loop[_nelts: Int](dy: Int):
                                let ix = x * stride - dx + padding
                                let iy = y * stride - dy + padding
                                if not (
                                    ix < 0
                                    or iy < 0
                                    or ix >= c.shape_ptr.load().load(2)
                                    or iy >= c.shape_ptr.load().load(3)
                                ):
                                    let c_grad_index = index(
                                        p,
                                        i,
                                        ix,
                                        iy,
                                        c.shape_ptr.load().load(1),
                                        c.shape_ptr.load().load(2),
                                        c.shape_ptr.load().load(3),
                                    )
                                    let b_index = index(
                                        i,
                                        j,
                                        b.shape_ptr.load().load(2) - dx - 1,
                                        b.shape_ptr.load().load(3) - dy - 1,
                                        b.shape_ptr.load().load(1),
                                        b.shape_ptr.load().load(2),
                                        b.shape_ptr.load().load(3),
                                    )
                                    patch_sum += (
                                        c.load_grad[_nelts](c_grad_index)
                                        * c.load_data[_nelts](b_index)
                                    ).reduce_add()

                            vectorize[nelts, dy_loop](b.shape_ptr.load().load(3))
                        let a_grad_index = index(
                            p,
                            j,
                            x,
                            y,
                            a.shape_ptr.load().load(1),
                            a.shape_ptr.load().load(2),
                            a.shape_ptr.load().load(3),
                        )
                        a.store_grad(
                            a_grad_index, a.load_grad(a_grad_index) + patch_sum
                        )

    parallelize[batch_loop](
        a.shape_ptr.load().load(0),
        workers if workers > 0 else a.shape_ptr.load().load(0),
    )


fn max_pool_2d(b: Node, a: Node):
    let padding = b.other_params_ptr.load().load(0)
    let stride = b.other_params_ptr.load().load(1)
    let kernel_width = b.other_params_ptr.load().load(2)
    let kernel_height = b.other_params_ptr.load().load(3)

    for p in range(a.shape_ptr.load().load(0)):
        for i in range(a.shape_ptr.load().load(1)):
            for x in range(
                0, a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding, stride
            ):
                for y in range(
                    0,
                    a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                    stride,
                ):
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= a.shape_ptr.load().load(2)
                                or iy >= a.shape_ptr.load().load(3)
                            ):
                                continue
                            let idx = index(
                                p,
                                i,
                                ix,
                                iy,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            let entry = a.load_data(idx)
                            if entry > max_val:
                                max_val = entry
                                arg_max = idx
                    let idx = index(
                        p,
                        i,
                        (x) // stride,
                        (y) // stride,
                        b.shape_ptr.load().load(1),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    b.store_data(idx, max_val)


fn bw_max_pool_2d(b: Node, a: Node):
    let padding = b.other_params_ptr.load().load(0)
    let stride = b.other_params_ptr.load().load(1)
    let kernel_width = b.other_params_ptr.load().load(2)
    let kernel_height = b.other_params_ptr.load().load(3)

    for p in range(a.shape_ptr.load().load(0)):
        for i in range(a.shape_ptr.load().load(1)):
            for x in range(
                0, a.shape_ptr.load().load(2) - kernel_width + 1 + 2 * padding, stride
            ):
                for y in range(
                    0,
                    a.shape_ptr.load().load(3) - kernel_height + 1 + 2 * padding,
                    stride,
                ):
                    var arg_max: Int = 0
                    var max_val: Float32 = -1000000.0
                    for dx in range(kernel_width):
                        for dy in range(kernel_height):
                            let ix = x - padding + dx
                            let iy = y - padding + dy
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= a.shape_ptr.load().load(2)
                                or iy >= a.shape_ptr.load().load(3)
                            ):
                                continue
                            let idx = index(
                                p,
                                i,
                                ix,
                                iy,
                                a.shape_ptr.load().load(1),
                                a.shape_ptr.load().load(2),
                                a.shape_ptr.load().load(3),
                            )
                            let entry = a.load_data(idx)
                            if entry > max_val:
                                max_val = entry
                                arg_max = idx
                    let b_grad_idx = index(
                        p,
                        i,
                        (x) // stride,
                        (y) // stride,
                        b.shape_ptr.load().load(1),
                        b.shape_ptr.load().load(2),
                        b.shape_ptr.load().load(3),
                    )
                    a.store_grad(
                        arg_max, a.load_grad(arg_max) + b.load_grad(b_grad_idx)
                    )


fn fw_dropout(node: Node, parent1: Node):
    let params = node.other_params_ptr.load()
    let keep_prob = params.load(0) / 1000000.0
    let scale = 1.0 / keep_prob
    # TODO: Implement mask shape
    
    @parameter
    fn v_dropout[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        for i in range(_nelts):
            let rand = random_float64()
            node.store_data[_nelts](i, (rand < keep_prob).cast[DType_F32]() * data * scale)
        

    vectorize[nelts, v_dropout](node.load_cap())


fn bw_dropout(node: Node, parent1: Node):
    let params = node.other_params_ptr.load()
    let keep_prob = params.load(0) / 1000000.0
    let scale = 1.0 / keep_prob

    @parameter
    fn v_dropout_bw[_nelts: Int](i: Int):
        let data = parent1.load_data[_nelts](i)
        for i in range(_nelts):
            let rand = random_float64()
            parent1.store_grad[_nelts](
                i, parent1.load_grad[_nelts](i) + (rand < keep_prob).cast[DType_F32]() * node.load_grad[_nelts](i) * scale
            )

    vectorize[nelts, v_dropout_bw](node.load_cap())