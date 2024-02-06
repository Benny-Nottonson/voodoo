from .array import Vector, reduce_vector_mul
from .broadcast import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    get_broadcasted_shape_for_ew_op,
    recursive_broadcast,
)
from .console import warn, error, info, success, debug, clear
