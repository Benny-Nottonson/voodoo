@always_inline
fn abs[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the absolute value of a vector of floats.
    """
    return llvm_intrinsic["llvm.abs.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn sqrt[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the square root of a vector of floats.
    """
    return llvm_intrinsic["llvm.sqrt.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn sin[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the sine of a vector of floats.
    """
    return llvm_intrinsic["llvm.sin.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn cos[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the cosine of a vector of floats.
    """
    return llvm_intrinsic["llvm.cos.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn exp[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the exponential base 3 of a vector of floats.
    """
    return llvm_intrinsic["llvm.exp.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn exp2[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the exponential base 2 of a vector of floats.
    """
    return llvm_intrinsic["llvm.exp2.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn exp10[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the exponential base 10 of a vector of floats.
    """
    return llvm_intrinsic["llvm.exp10.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn log[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the natural logarithm of a vector of floats.
    """
    return llvm_intrinsic["llvm.log.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn log10[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the base 10 logarithm of a vector of floats.
    """
    return llvm_intrinsic["llvm.log10.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn log2[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the base 2 logarithm of a vector of floats.
    """
    return llvm_intrinsic["llvm.log2.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn fabs[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the absolute value of a vector of floats.
    """
    return llvm_intrinsic["llvm.fabs.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn floor[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to round down a vector of floats.
    """
    return llvm_intrinsic["llvm.floor.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn ceil[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to round up a vector of floats.
    """
    return llvm_intrinsic["llvm.ceil.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn trunc[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to truncate a vector of floats.
    """
    return llvm_intrinsic["llvm.trunc.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn rint[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.rint.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn nearbyint[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.nearbyint.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn round[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.round.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn roundeven[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to round to the nearest even integer a vector of floats.
    """
    return llvm_intrinsic["llvm.roundeven.v4f32", SIMD[DType.float32, nelts]](x)


@always_inline
fn lround[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.int32, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.lround.v4f32", SIMD[DType.int32, nelts]](x)


@always_inline
fn llround[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.int64, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.llround.v4f32", SIMD[DType.int64, nelts]](x)


@always_inline
fn lrint[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.int32, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.lrint.v4f32", SIMD[DType.int32, nelts]](x)


@always_inline
fn llrint[nelts: Int](x: SIMD[DType.float32, nelts]) -> SIMD[DType.int64, nelts]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.llrint.v4f32", SIMD[DType.int64, nelts]](x)
