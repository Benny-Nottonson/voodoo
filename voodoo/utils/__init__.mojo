from voodoo.utils.array import Vector, reduce_vector_mul
from voodoo.utils.broadcast import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    get_broadcasted_shape_for_ew_op,
    recursive_broadcast,
)
from voodoo.utils.console import warn, error, info, success, debug, clear
