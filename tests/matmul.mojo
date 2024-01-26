from voodoo import Tensor, shape


fn main() raises:
    let a = Tensor(shape(4, 8)).initialize["ones"]()
    let b = Tensor(shape(8, 4)).initialize["ones"]()
    if not (a @ b == Tensor(shape(4, 4)).initialize["ones"]() * 8):
        raise "Matrix multiplication failed test one"
