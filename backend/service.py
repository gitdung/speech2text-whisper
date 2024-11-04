from fastapi import HTTPException
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import io
processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-base")

def recognition_service(media):
    try:
        print(f"con cac: {media}")
        array, sample_rate = torchaudio.load(media)
        print(array.shape, 'haha', sample_rate)
        
        if sample_rate > 16000:
            array = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(array)
        if array.size(dim=0) > 1:
            array = array.mean(dim=0)

        array = array.squeeze(dim=0)

        input_features = processor(array, sampling_rate=16000, return_tensors="pt").input_features

        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

        transcription_cleaned = transcription[0]
        unwanted_tokens = ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]

        for token in unwanted_tokens:
            transcription_cleaned = transcription_cleaned.replace(token, "")

        transcription_cleaned = transcription_cleaned.strip()
        return transcription_cleaned
    except Exception as e:
        raise ValueError(e)