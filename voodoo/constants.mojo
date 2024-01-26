from sys.param_env import env_get_int
from math.limit import inf
from sys.intrinsics import PrefetchOptions

alias MEMORY_POOL_SIZE = 2500
alias EPSILON = 1e-8
alias F32_MAX = inf[DType.float32]()
alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()
alias NELTS = simdwidthof[DType.float32]() * 4

alias UNARY_OP = fn (b: Node, a: Node) -> None
alias BINARY_OP = fn (c: Node, a: Node, b: Node) -> None
alias OP_TUPLE = Tuple[UNARY_OP, BINARY_OP]


fn NU(b: Node, a: Node):
    ...


fn NB(c: Node, a: Node, b: Node):
    ...
