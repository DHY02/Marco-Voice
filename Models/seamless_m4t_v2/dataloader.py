# 

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from fairseq2.data.text import TextTokenEncoder
from fairseq2.models.nllb import NllbTokenizer
from fairseq2.data.audio import WaveformToFbankConverter
from torch import Tensor
from torch.nn.functional import pad as pad_tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class AudioTextSample:
    audio_path: str      
    text: str         
    lang: str = "ch"     

@dataclass
class SeqsBatch:
    src_tokens: Optional[Tensor]
    src_lengths: Optional[Tensor]
    target_tokens: Optional[Tensor]
    prev_output_tokens: Optional[Tensor]
    target_lengths: Optional[Tensor]

@dataclass
class MultimodalSeqsBatch:
    speech_to_text: SeqsBatch
    text_to_units: Optional[SeqsBatch] = None  # 可选单元任务

@dataclass
class BatchingConfig:
    fbank_feats_pad_idx: int = 0
    batch_size: int = 5
    max_audio_length_sec: float = 15.0
    rank: int = 0
    world_size: int = 1
    num_workers: int = 2
    float_dtype: torch.dtype = torch.float16

class SeamlessM4TDataLoader:
    SAMPLE_RATE = 16_000

    def __init__(
        self,
        text_tokenizer: NllbTokenizer,
        dataset_manifest_path: str,
        batching_config: BatchingConfig,
        max_src_tokens_per_batch: int = 100000
    ):
        self.text_tokenizer = text_tokenizer
        self.text_encoders_per_lang: Dict[str, TextTokenEncoder] = {}
        self.batching_config = batching_config
        self._fbank_extract_params = {
            "num_mel_bins": 80,
            "waveform_scale": 32768,
            "channel_last": True,
            "standardize": True,
            "device": torch.device("cpu"),
            "dtype": self.batching_config.float_dtype,
        }
        self.dataset = self._load_manifest(dataset_manifest_path)
        self.max_src_tokens_per_batch = max_src_tokens_per_batch

    def get_dataloader(self) -> DataLoader[MultimodalSeqsBatch]:
        subset = split_dataset_by_node(
            self.dataset,
            rank=self.batching_config.rank,
            world_size=self.batching_config.world_size,
        )
        return DataLoader(
            dataset=subset,
            batch_size=self.batching_config.batch_size,
            shuffle=True,
            num_workers=self.batching_config.num_workers,
            collate_fn=self._prepare_batch,
            worker_init_fn=lambda _: np.random.seed(np.random.get_state()[1][0] + _),
        )

    def _load_manifest(self, dataset_manifest_path: str) -> Dataset:
        samples = []
        with open(dataset_manifest_path) as f:
            for line in f:
                if "|" not in line:
                    continue
                audio_path, text = line.strip().split("|", maxsplit=1)
                samples.append({
                    "audio_path": audio_path,
                    "text": text,
                    "lang": "ch"  #
                })
        return Dataset.from_list(samples)

    def _get_source_fbank(self, sample: AudioTextSample) -> Tensor:
        wav, sample_rate = torchaudio.load(sample.audio_path)
        assert sample_rate == self.SAMPLE_RATE, f"需重采样到 {self.SAMPLE_RATE}Hz"
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(-1)
        elif wav.shape[0] <= 2:  # 
            wav = wav.transpose(0, 1)
        return WaveformToFbankConverter(**self._fbank_extract_params)(
            {"waveform": wav, "sample_rate": self.SAMPLE_RATE}
        )["fbank"]

    def _get_tokenized_text(self, text: str, lang: str) -> Tensor:
        """返回格式: [eos, lang_tok, tokens, eos]"""
        if lang not in self.text_encoders_per_lang:
            self.text_encoders_per_lang[lang] = self.text_tokenizer.create_encoder(
                lang=lang, mode="target"
            )
        tokens = self.text_encoders_per_lang[lang](text)
        eos_idx = self.text_tokenizer.vocab_info.eos_idx
        return torch.cat([tokens, torch.LongTensor([eos_idx])])

    def _batch_tensors(self, tensors: List[Tensor], pad_value: Any) -> Tensor:
        max_len = max(t.shape[0] for t in tensors)
        padded = [
            pad_tensor(t, (0, max_len - t.shape[0]), value=pad_value)
            for t in tensors
        ]
        return torch.stack(padded)

    def _is_long_audio(self, audio_path: str) -> bool:
        try:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate > self.batching_config.max_audio_length_sec
        except:
            logger.warning(f"无法读取音频时长: {audio_path}")
            return True

    def _prepare_batch(self, raw_samples: List[Dict[str, Any]]) -> MultimodalSeqsBatch:
        # 1. 
        samples = [
            AudioTextSample(s["audio_path"], s["text"], s["lang"])
            for s in raw_samples if not self._is_long_audio(s["audio_path"])
        ]
        if not samples:
            samples = [AudioTextSample(raw_samples[0]["audio_path"], raw_samples[0]["text"], "ch")]

        with_fbanks = [(s, self._get_source_fbank(s)) for s in samples]
        valid_samples = [sb for sb in with_fbanks if not sb[1].isnan().any()]

        valid_samples.sort(key=lambda x: -x[1].shape[0]) 
        max_samples = max(1, self.max_src_tokens_per_batch // valid_samples[0][1].shape[0])
        valid_samples = valid_samples[:max_samples]

        samples = [s for s, _ in valid_samples]
        src_tokens = self._batch_tensors(
            [fbank for _, fbank in valid_samples],
            pad_value=self.batching_config.fbank_feats_pad_idx
        ).to(self.batching_config.float_dtype)

        # 5. text
        text_tokens = [self._get_tokenized_text(s.text, s.lang) for s in samples]
        text_pad_idx = self.text_tokenizer.vocab_info.pad_idx

        return MultimodalSeqsBatch(
            speech_to_text=SeqsBatch(
                src_tokens=src_tokens,
                src_lengths=torch.LongTensor([fbank.shape[0] for _, fbank in valid_samples]),
                target_tokens=self._batch_tensors([t[1:] for t in text_tokens], text_pad_idx),
                prev_output_tokens=self._batch_tensors([t[:-1] for t in text_tokens], text_pad_idx),
                target_lengths=torch.LongTensor([len(t)-1 for t in text_tokens])
            )
        )

if __name__ == "__main__":
    text_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    loader = SeamlessM4TDataLoader(
        text_tokenizer=text_tokenizer,
        dataset_manifest_path="data set",  # format: path|text
        batching_config=BatchingConfig()
    )

    for batch in loader.get_dataloader():
        print("speech feature shape:", batch.speech_to_text.src_tokens.shape)
        print("target Token:", batch.speech_to_text.target_tokens[0])