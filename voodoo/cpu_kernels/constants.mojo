from sys.param_env import env_get_int

alias DType_F32 = DType.float32
alias nelts = simdwidthof[DType_F32]()
alias workers = env_get_int["WORKERS", 0]()
alias epsilon = 1e-8
