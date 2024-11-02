from fastapi import HTTPException
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import io
processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-base")

def recognition_service(media):
    try:
        array, sample_rate = torchaudio.load(media)
        print(array.shape)
        
        if sample_rate > 16000:
            array = torchaudio.transforms.Resample( sample_rate, 16000)(array)
        if array.size(dim=0) >1:
            array = array.mean(dim=0)

        array = array.squeeze(dim=0)

        input_features = processor(array, sampling_rate=sample_rate, return_tensors="pt").input_features

        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
        return transcription
    except Exception as e:
        raise ValueError(e)