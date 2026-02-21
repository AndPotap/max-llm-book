import math

import max.functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim, DimLike
from max.nn import Linear
from max.nn.module import Module
from max.tensor import Tensor
from step_01 import GPT2Config


@F.functional
def causal_mask(seq_length: DimLike, num_tokens: DimLike, *, dtype: DType, device: Device) -> Tensor:
    n = Dim(seq_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(seq_length, n))
    mask = F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
    return mask


class GPT2MultiHeadAttention(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor: Tensor, num_heads: int, attn_head_dim: int) -> Tensor:
        new_shape = list(tensor.shape[:-1]) + [num_heads, attn_head_dim]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)

    def _merge_heads(self, tensor: Tensor, num_heads: int, attn_head_dim: int) -> Tensor:
        tensor = tensor.transpose(-3, -2)
        new_shape = list(tensor.shape[:-2]) + [num_heads * attn_head_dim]
        return tensor.reshape(new_shape)

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute attention for all heads in parallel.

        Args:
            query: Query tensor, shape [batch, num_heads, seq_length, head_dim]
            key: Key tensor, shape [batch, num_heads, seq_length, head_dim]
            value: Value tensor, shape [batch, num_heads, seq_length, head_dim]

        Returns:
            Attention output, shape [batch, num_heads, seq_length, head_dim]
        """
        _, num_heads, seq_length, head_dim = query.shape
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(int(head_dim))
        mask = causal_mask(seq_length, 0, dtype=query.dtype, device=query.device)
        attn_weights = F.softmax(attn_weights + mask)
        output = attn_weights @ value
        return output

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: [B,S,D]
        qkv = self.c_attn(hidden_states)
        query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)
        return attn_output
