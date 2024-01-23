from sys.param_env import env_get_int
from math.limit import inf
from sys.intrinsics import PrefetchOptions
from autotune import autotune, search

alias memory_pool_size = 2500
alias workers = env_get_int["WORKERS", 0]()
alias epsilon = 1e-8
alias f32_max = inf[DType.float32]()
alias prefetch_read = PrefetchOptions().for_read().high_locality().to_data_cache()
alias prefetch_write = PrefetchOptions().for_write().high_locality().to_data_cache()
alias nelts = simdwidthof[DType.float32]() * 4
