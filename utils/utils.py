import yaml
import zlib

from torch import Tensor
from typing import List
import json


def get_vocab_meta(path: str):
    if not isinstance(path, str):
        raise ValueError("path must be str")

    # Load the vocab data from the JSON file
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    # Extract required information
    vocab_size = len(vocab)
    start_of_text_index = vocab.get("<|startoftext|>")
    end_of_text_index = vocab.get("<|endoftext|>")

    return vocab_size, start_of_text_index, end_of_text_index


def get_configs(path: str):
    params = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return params


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def convert_tensor2list(x: Tensor) -> List:
    # print(f"Tensor on GPU: {x}")
    if not isinstance(x, Tensor):
        raise ValueError("x is not a tensor")

    out = x.cpu().numpy().tolist()
    # print(type(out))
    return out
