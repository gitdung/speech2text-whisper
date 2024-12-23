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

def calculate_average_wer(file_paths):
    total_wer = 0.0
    count = 0

    for file_path in file_paths:
        # Kiểm tra xem file có tồn tại hay không
        if not os.path.isfile(file_path):
            print(f"File không tồn tại: {file_path}")
            continue

        # Đọc file JSON
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                wer = data.get('wer')

                if wer is not None:
                    total_wer += wer
                    count += 1
                else:
                    print(f"Không tìm thấy WER trong file: {file_path}")
            except json.JSONDecodeError:
                print(f"Đọc file JSON không thành công: {file_path}")

    # Tính trung bình WER
    if count > 0:
        average_wer = total_wer / count
        return average_wer
    else:
        return None  # Nếu không có tệp hợp lệ nào được xử lý


# Ví dụ sử dụng
file_paths = ['file1.json', 'file2.json', 'file3.json']  # Thay đổi danh sách tệp JSON của bạn tại đây
average_wer = calculate_average_wer(file_paths)

if average_wer is not None:
    print(f"Trung bình WER: {average_wer}")
else:
    print("Không có dữ liệu hợp lệ để tính toán WER.")

if __name__ == "__main__":
    # lowercase
    # folder_path = "D:\\speech2text-whisper\\data"
    # file_name = "CommonVoice_test.json"
    # lowercase_transcripts(folder_path, file_name)

    # WER
    calculate_average_wer()