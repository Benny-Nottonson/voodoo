from .array import Vector
from .broadcast import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    get_broadcasted_shape_for_ew_op,
    recursive_broadcast,
    recursive_broadcast_bw,
)
from .console import warn, error, info, success, debug, clear
