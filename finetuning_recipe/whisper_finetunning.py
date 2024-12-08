from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig, OmegaConf
import torchaudio
from dataclasses import dataclass
from utils.utils import get_configs

from typing import Any, List, Optional, Dict, Union
import torch
import evaluate


# datacollator
@dataclass
class DataCollatorSpeech:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class WhisperFinetuner:

    def __init__(self, conf: DictConfig, tokenizer, feature_extractor, processor):
        self.conf = OmegaConf.create(conf)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.processor = processor

        # model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.conf.model.name
        )

        for params in self.model.model.encoder.parameters():
            params.requires_grad = False

        # setup dataset
        self.dataset = DatasetDict()
        self.dataset["train"] = load_dataset(
            self.conf.dataset, split="train"
        )
        self.dataset["test"] = load_dataset(self.conf.dataset, split="test")
        self.dataset["train"] = self.dataset["train"].map(self.preprocess_ds)
        self.dataset["test"] = self.dataset["test"].map(self.preprocess_ds)

        # data collator
        self.data_collator = DataCollatorSpeech(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # metric
        self.metric = evaluate.load("wer")

    def preprocess_ds(self, batch):
        audio = batch["audio"]
        waveform = torch.from_numpy(audio["array"])

        batch["input_features"] = self.feature_extractor(
            waveform,
            sampling_rate=self.conf.processor.sampling_rate,
            return_tensors="pt",
        ).input_features[0]

        batch["labels"] = self.tokenizer(batch["transcription"]).input_ids

        return batch

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def setup_trainer(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper-base-vi",  # change to a repo name of your choice
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["wandb"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        return training_args

    def train(self):
        training_args = self.setup_trainer()

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

if __name__ == "__main__":
    conf = get_configs("./configs/default.yaml")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("vinai/PhoWhisper-base")
    processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
    fine_tuner = WhisperFinetuner(conf, tokenizer, feature_extractor, processor)
    fine_tuner.train()