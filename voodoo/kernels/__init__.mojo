from .loader import load_kernels

alias unary_op = fn (b: Node, a: Node) -> None
alias binary_op = fn (c: Node, a: Node, b: Node) -> None
alias op_tuple = Tuple[unary_op, binary_op]


fn _u(b: Node, a: Node):
    ...


fn _b(c: Node, a: Node, b: Node):
    ...

@always_inline
fn get_kernels() -> Pointer[op_tuple]:
    return load_kernels()