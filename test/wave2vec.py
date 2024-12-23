import yaml
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio

# yaml
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0), target_sr

def test_wave2vec(audio_file, model_name):
    print("Testing Wave2Vec with model:", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    waveform, sr = load_audio(audio_file)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("Wave2Vec transcription:", transcription)

if __name__ == "__main__":
    config_path = "../backend/config/app_config.yaml"
    audio_file = "../audio/ElevenLab_audio_Mình sẽ se.mp3"

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
    else:
        print("true")
        # Load cấu hình
        config = load_config(config_path)
        model_name = config.get("model_name")
        test_wave2vec(audio_file, model_name)
