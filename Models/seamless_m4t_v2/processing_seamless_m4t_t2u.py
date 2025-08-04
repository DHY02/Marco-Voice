# 

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Audio/Text processor class for SeamlessM4T (Modified to use w2v-BERT 2.0)
"""

from typing import Optional, Union, List
import numpy as np
import torch
# t2u_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

from ...processing_utils import ProcessorMixin
from ...models.seamless_m4t import SeamlessM4TFeatureExtractor, SeamlessM4TTokenizer

class SeamlessM4TProcessor(ProcessorMixin):
    r"""
    Constructs a SeamlessM4T processor which wraps:
    - A SeamlessM4T feature extractor (for audio)
    - A SeamlessM4T tokenizer (for text)
    - Unit extraction via w2v-BERT 2.0
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = ("SeamlessM4TTokenizer", "SeamlessM4TTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.unit_tokenizer = getattr(feature_extractor, "extract_units", None)
        self.char_pad_token_id = 0  # Default padding ID for characters
        self.char_bos_token_id = 1

    def _process_text(
        self,
        text: Union[str, List[str]],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        return_char_info: bool = True,
        **kwargs
    ): 
        return_char_info =True
        """Enhanced text processing with character-level decomposition"""
        # Set language
        if tgt_lang is not None:
            self.tokenizer.tgt_lang = tgt_lang
        if src_lang is not None:
            self.tokenizer.src_lang = src_lang

        # Tokenize text
        text_encoding = self.tokenizer(text, **kwargs)

        if not return_char_info:
            return text_encoding

        # Character-level processing
        if isinstance(text, str):
            text = [text]

        # Get subword tokens
        subword_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in text_encoding["input_ids"]]

        # Initialize character arrays
        batch_size = len(text)
        max_subwords = max(len(seq) for seq in subword_tokens)
        max_chars_length = [len(token.replace("▁", "")) for seq in subword_tokens for token in seq]
        max_chars = max(max_chars_length) if batch_size > 0 else 0

        char_input_ids = torch.full(
            (batch_size, max_subwords, max_chars),
            fill_value=self.char_pad_token_id,
            dtype=torch.long
        )
        char_count_per_id = torch.zeros((batch_size, max_subwords), dtype=torch.long)

        # Fill character info
        for batch_idx, tokens in enumerate(subword_tokens):
            for subword_idx, token in enumerate(tokens):
                # Remove subword prefix (▁)
                clean_token = token.replace("▁", "")
                char_count = len(clean_token)
                char_count_per_id[batch_idx, subword_idx] = char_count

                if char_count > 0:
                    char_ids = []
                    for c in clean_token:
                        code = ord(c)
                        if code >= 10943:
                            # Handle out-of-vocab characters
                            code = 0  # or some other handling
                        char_ids.append(code)
                    # char_ids = [ord(c) for c in clean_token]
                    char_input_ids[batch_idx, subword_idx, :char_count] = torch.tensor(char_ids)

        text_encoding.update({
            "char_input_ids": char_input_ids,
            "char_count_per_id": char_count_per_id
        })

        return text_encoding

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audios: Optional[Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        return_units: bool = True,
        return_char_info: bool = False,
        **kwargs
    ):
        """
        Args:
            return_units: If True, uses w2v-BERT 2.0 to extract discrete units.
        """
        if text is not None:
            return self._process_text(text, src_lang, tgt_lang, return_char_info, **kwargs)

        # Process audio with w2v-BERT 2.0 unit extraction
        encoding = self.feature_extractor(audios, **kwargs)
        return_units = True
        if return_units:
            if self.unit_tokenizer is not None:
                units = self.unit_tokenizer(audios)
            else:
                units = self._extract_units_with_w2v_bert(audios)
            encoding["units"] = units
        print("encodingcd:", encoding)
        return encoding

    def _extract_units_with_w2v_bert(self, audios, sampling_rate=None):
        """Extract discrete units using w2v-BERT 2.0 with proper audio preprocessing."""
        # try:
        from transformers import Wav2Vec2BertModel, Wav2Vec2FeatureExtractor, AutoFeatureExtractor
        import torch

        # Initialize model and feature extractor
        processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

        # model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()
        # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        # Ensure audio is a list of tensors
        if isinstance(audios, (np.ndarray, torch.Tensor)):
            audios = [audios]
        print("audios:", len(audios), audios[0]) # .shape)
        # Preprocess audio (resample + normalize + chunk into frames)
        # inputs = _get_source_fbank(audios[0])

        inputs = processor(
            audios[0],
            sampling_rate=sampling_rate if sampling_rate else 16000,
            return_tensors="pt",
            padding="longest"
        )
        # input_values = inputs.input_values.to(model.device)

        # # Forward pass
        print("inputsshae:", inputs)
        # print("input_values:", input_values)
        # print("input_values shape:", input_values.shape)

        with torch.no_grad():
            print("model(**inputs):", model(**inputs)["extract_features"].shape, model(**inputs)["last_hidden_state"].shape)
            outputs = model(**inputs)
            # if hasattr(model, "quantizer"):
            #     units = model.quantizer(outputs.last_hidden_state).indices
            # else:
            #     units = outputs.last_hidden_state.argmax(dim=-1)
            # units = units.squeeze().cpu().numpy().tolist()
        print("outputs is :", outputs)
        return outputs
        # return units if isinstance(units[0], list) else [units]

        # except Exception as e:
        #     raise RuntimeError(f"Failed to extract units with w2v-BERT: {e}")

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

__all__ = ["SeamlessM4TProcessor"]