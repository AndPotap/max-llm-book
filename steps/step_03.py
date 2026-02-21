import max.functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim, DimLike
from max.tensor import Tensor


@F.functional
def causal_mask(sequence_length: DimLike, num_tokens: DimLike, *, dtype: DType, device: Device) -> Tensor:
    n = Dim(sequence_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    mask = F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
    return mask
