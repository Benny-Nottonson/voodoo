from sys.param_env import env_get_int
from math.limit import inf

alias nelts = simdwidthof[DType.float32]()
alias memory_pool_size = 30
alias DType_F32 = DType.float32
alias workers = env_get_int["WORKERS", 0]()
alias epsilon = 1e-8
alias f32_max = inf[DType_F32]()
