@always_inline
fn smax[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the signed maximum of two vectors.
    """
    return llvm_intrinsic["llvm.smax.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn smin[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the signed minimum of two vectors.
    """
    return llvm_intrinsic["llvm.smin.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn umax[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the unsigned maximum of two vectors.
    """
    return llvm_intrinsic["llvm.umax.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn umin[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the unsigned minimum of two vectors.
    """
    return llvm_intrinsic["llvm.umin.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn powi[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the power of two vectors.
    """
    return llvm_intrinsic["llvm.powi.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn pow[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the power of two vectors.
    """
    return llvm_intrinsic["llvm.pow.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn ldexp[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the ldexp of two vectors.
    """
    return llvm_intrinsic["llvm.ldexp.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn frexp[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> (
    SIMD[DType.float32, nelts],
    SIMD[DType.float32, nelts],
):
    """
    GPU Kernel to compute the frexp of two vectors.
    """
    return llvm_intrinsic[
        "llvm.frexp.v4f32", (SIMD[DType.float32, nelts], SIMD[DType.float32, nelts])
    ](x, y)


@always_inline
fn minnum[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the minnum of two vectors.
    """
    return llvm_intrinsic["llvm.minnum.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn maxnum[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the maxnum of two vectors.
    """
    return llvm_intrinsic["llvm.maxnum.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn minimum[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the minimum of two vectors.
    """
    return llvm_intrinsic["llvm.minimum.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn maximum[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to compute the maximum of two vectors.
    """
    return llvm_intrinsic["llvm.maximum.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn copysign[
    nelts: Int
](x: SIMD[DType.float32, nelts], y: SIMD[DType.float32, nelts]) -> SIMD[
    DType.float32, nelts
]:
    """
    GPU Kernel to copy the sign of two vectors.
    """
    return llvm_intrinsic["llvm.copysign.v4f32", SIMD[DType.float32, nelts]](x, y)


@always_inline
fn fma[
    nelts: Int
](
    x: SIMD[DType.float32, nelts],
    y: SIMD[DType.float32, nelts],
    z: SIMD[DType.float32, nelts],
) -> SIMD[DType.float32, nelts]:
    """
    GPU Kernel to compute the fused multiply-add of three vectors.
    """
    return llvm_intrinsic["llvm.fma.v4f32", SIMD[DType.float32, nelts]](x, y, z)
