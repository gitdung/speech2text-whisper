import yaml
import zlib

from torch import Tensor
from typing import List
import json
import os


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
def lowercase_transcripts(folder_path, file_name):
    # Đường dẫn đầy đủ tới file json
    input_file_path = os.path.join(folder_path, file_name)

    # Tên file mới với hậu tố _lower
    base_name, ext = os.path.splitext(file_name)
    output_file_name = f"{base_name}_lower{ext}"
    output_file_path = os.path.join(folder_path, output_file_name)

    try:
        # Đọc nội dung file json
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Chuyển transcript thành chữ thường
        for entry in data:
            if 'transcript' in entry:
                entry['transcript'] = entry['transcript'].lower()

        # Ghi nội dung mới vào file _lower
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"File mới đã được lưu tại: {output_file_path}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
if __name__ == "__main__":
    folder_path = "D:\\speech2text-whisper\\data"
    file_name = "CommonVoice_test.json"
    lowercase_transcripts(folder_path, file_name)