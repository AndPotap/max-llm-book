import max.functional as F
from max.dtype import DType
from max.graph import DimLike
from max.nn import Module
from max.tensor import Tensor


class LayerNorm(Module):
    def __init__(self, dim: DimLike, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # @ap: why isn't device or dtype set?
        # I had to add dtype to pass the tests
        self.weight = Tensor.ones(shape=[dim], dtype=DType.bfloat16)
        self.bias = Tensor.zeros(shape=[dim], dtype=DType.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        # x: [...,D]
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)
