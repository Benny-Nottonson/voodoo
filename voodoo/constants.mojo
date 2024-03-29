from math.limit import inf
from sys.intrinsics import PrefetchOptions

from voodoo.autograd import Node

alias PI = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342
alias MEMORY_POOL_SIZE = 20
alias EPSILON = 1e-8
alias F32_MAX = inf[DType.float32]()
alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()
alias NELTS = simdwidthof[DType.float32]() * 2

alias UNARY_OP = fn (b: Node, a: Node) -> None
alias BINARY_OP = fn (c: Node, a: Node, b: Node) -> None
alias OP_TUPLE = Tuple[UNARY_OP, BINARY_OP]


fn NU(b: Node, a: Node):
    ...


fn NB(c: Node, a: Node, b: Node):
    ...
