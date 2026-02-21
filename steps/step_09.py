import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.tensor import Tensor
from transformers import GPT2Tokenizer


def encode_text(text: str, tokenizer: GPT2Tokenizer, device: Device, max_length: int) -> Tensor:
    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    return Tensor.constant([token_ids], dtype=DType.int64, device=device)


def decode_tokens(token_ids: Tensor, tokenizer: GPT2Tokenizer) -> str:
    token_ids_np = np.from_dlpack(token_ids.to(CPU()))

    if token_ids_np.ndim > 1:
        token_ids_np = token_ids_np.flatten()

    token_ids_list: list = token_ids_np.tolist()
    return tokenizer.decode(token_ids_list, skip_special_tokens=True)
