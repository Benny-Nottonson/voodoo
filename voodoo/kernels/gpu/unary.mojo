@always_inline
fn abs[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the absolute value of a vector of floats.
    """
    return llvm_intrinsic["llvm.abs.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn sqrt[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the square root of a vector of floats.
    """
    return llvm_intrinsic["llvm.sqrt.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn sin[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the sine of a vector of floats.
    """
    return llvm_intrinsic["llvm.sin.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn cos[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the cosine of a vector of floats.
    """
    return llvm_intrinsic["llvm.cos.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn exp[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the exponential base 3 of a vector of floats.
    """
    return llvm_intrinsic["llvm.exp.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn exp2[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the exponential base 2 of a vector of floats.
    """
    return llvm_intrinsic["llvm.exp2.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn exp10[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the exponential base 10 of a vector of floats.
    """
    return llvm_intrinsic["llvm.exp10.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn log[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the natural logarithm of a vector of floats.
    """
    return llvm_intrinsic["llvm.log.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn log10[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the base 10 logarithm of a vector of floats.
    """
    return llvm_intrinsic["llvm.log10.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn log2[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the base 2 logarithm of a vector of floats.
    """
    return llvm_intrinsic["llvm.log2.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn fabs[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to compute the absolute value of a vector of floats.
    """
    return llvm_intrinsic["llvm.fabs.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn floor[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to round down a vector of floats.
    """
    return llvm_intrinsic["llvm.floor.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn ceil[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to round up a vector of floats.
    """
    return llvm_intrinsic["llvm.ceil.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn trunc[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to truncate a vector of floats.
    """
    return llvm_intrinsic["llvm.trunc.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn rint[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.rint.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn nearbyint[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.nearbyint.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn round[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.round.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn roundeven[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    """
    GPU Kernel to round to the nearest even integer a vector of floats.
    """
    return llvm_intrinsic["llvm.roundeven.v4f32", SIMD[DType.float32, NELTS]](x)


@always_inline
fn lround[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.int32, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.lround.v4f32", SIMD[DType.int32, NELTS]](x)


@always_inline
fn llround[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.int64, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.llround.v4f32", SIMD[DType.int64, NELTS]](x)


@always_inline
fn lrint[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.int32, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.lrint.v4f32", SIMD[DType.int32, NELTS]](x)


@always_inline
fn llrint[NELTS: Int](x: SIMD[DType.float32, NELTS]) -> SIMD[DType.int64, NELTS]:
    """
    GPU Kernel to round to the nearest integer a vector of floats.
    """
    return llvm_intrinsic["llvm.llrint.v4f32", SIMD[DType.int64, NELTS]](x)
