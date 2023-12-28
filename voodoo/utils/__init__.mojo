from .array import Vector
from .shape import shape, get_broadcasted_shape_for_ew_op
from .broadcast import (
    shape_a,
    shape_b,
    strides_a,
    strides_b,
    recursive_broadcast,
    recursive_broadcast_bw,
)
