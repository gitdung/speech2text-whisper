import json
import os
import torchaudio
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from jiwer import wer

def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0), target_sr

def transcribe_audio(file_path, processor, model, target_length=3000):
    waveform, sr = load_audio(file_path)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)

    mel_features = inputs["input_features"]

    # check len
    if mel_features.size(-1) < target_length:
        # padding
        pad_length = target_length - mel_features.size(-1)
        mel_features = torch.nn.functional.pad(mel_features, (0, pad_length))
    elif mel_features.size(-1) > target_length:
        mel_features = mel_features[:, :, :target_length]

    with torch.no_grad():
        predicted_ids = model.generate(mel_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def process_json(json_path, audio_folder, model_name="vinai/Whisper-tiny"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    total_wer = 0

    for entry in data:
        audio_file = entry["audio_path"]
        audio_path = os.path.join(audio_folder, audio_file)
        ground_truth = entry["transcript"]

        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            entry.update({
                "predicted_transcript": None,
                "wer": None,
                "error": "File not found"
            })
            results.append(entry)
            continue

        # transcript
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
    output_path = os.path.splitext(json_path)[0] + "_predict_PhoWhisper.json"
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")


# Chạy chương trình
if __name__ == "__main__":
    json_path = "D:\\speech2text-whisper\\data\\vivos_test_lower.json"
    audio_folder = "D:\\speech2text-whisper\\data\\vivos_filtered"
    process_json(json_path, audio_folder)
