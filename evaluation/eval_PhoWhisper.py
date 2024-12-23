import json
import os
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration
from jiwer import wer


# Load và tiền xử lý tệp âm thanh
def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0), target_sr


# Inference trên mô hình PhoWhisper
def transcribe_audio(file_path, processor, model):
    waveform, sr = load_audio(file_path)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


# Đọc JSON, duyệt file và tính WER
def process_json(json_path, audio_folder, model_name="vinai/PhoWhisper-tiny"):
    # Load processor và model PhoWhisper
    processor = AutoProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Đọc file JSON
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    total_wer = 0

    for entry in data:
        audio_file = entry["audio_path"]
        audio_path = os.path.join(audio_folder, audio_file)  # Kết hợp folder và file name
        ground_truth = entry["transcript"]

        # Kiểm tra file âm thanh có tồn tại không
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            entry.update({
                "predicted_transcript": None,
                "wer": None,
                "error": "File not found"
            })
            results.append(entry)
            continue

        # Dự đoán transcript
        print(f"Processing file: {audio_path}")
        try:
            predicted_transcript = transcribe_audio(audio_path, processor, model)
            print(f"Predicted: {predicted_transcript}")
            print(f"Ground Truth: {ground_truth}")

            # Tính WER
            file_wer = wer(ground_truth, predicted_transcript)
            total_wer += file_wer
            print(f"WER: {file_wer:.2f}\n")

            # Thêm kết quả vào danh sách
            entry.update({
                "predicted_transcript": predicted_transcript,
                "wer": file_wer
            })
            results.append(entry)
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            entry.update({
                "predicted_transcript": None,
                "wer": None,
                "error": str(e)
            })
            results.append(entry)

    # WER trung bình
    avg_wer = total_wer / len(data)
    print(f"Average WER: {avg_wer:.2f}")

    # Ghi kết quả vào file mới
    output_path = os.path.splitext(json_path)[0] + "_predict.json"
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")


# Chạy chương trình
if __name__ == "__main__":
    json_path = "D:\\speech2text-whisper\\data\\vivos_test_lower.json"
    audio_folder = "audio_folder"
    process_json(json_path, audio_folder)
