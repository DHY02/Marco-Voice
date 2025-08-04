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
Audio/Text processor class for SeamlessM4T
"""

from typing import Optional, Union, List
import numpy as np
import torch
from ...processing_utils import ProcessorMixin
from ...models.seamless_m4t import SeamlessM4TFeatureExtractor, SeamlessM4TTokenizer

class SeamlessM4TProcessor(ProcessorMixin):
    r"""
    Constructs a SeamlessM4T processor which wraps:
    - A SeamlessM4T feature extractor (for audio)
    - A SeamlessM4T tokenizer (for text)
    - A Unit Tokenizer (for extracting discrete units from audio)

    [`SeamlessM4TProcessor`] offers all the functionalities of:
    - [`SeamlessM4TFeatureExtractor`]
    - [`SeamlessM4TTokenizerFast`]
    - Unit extraction (via HuBERT or built-in method)
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = ("SeamlessM4TTokenizer", "SeamlessM4TTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # Initialize unit tokenizer if needed (SeamlessM4T may have its own)
        self.unit_tokenizer = getattr(feature_extractor, "extract_units", None)

        # Character processing parameters
        self.char_pad_token_id = 0  # Default padding ID for characters
        self.char_bos_token_id = 1
        # self.unit_tokenizer = None
        # if hasattr(feature_extractor, "extract_units"):
        #     self.unit_tokenizer = feature_extractor.extract_units

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
        return_units: bool = False,
        return_char_info: bool = False,
        **kwargs
    ):
        """
        Args:
            text (`str` or `List[str]`): Input text to tokenize.
            audios (`np.ndarray` or `torch.Tensor`): Raw audio to process.
            src_lang (`str`): Source language code (e.g., "eng").
            tgt_lang (`str`): Target language code (e.g., "fra").
            return_units (`bool`): If True, returns extracted units for audio inputs.
            **kwargs: Additional arguments for tokenizer/feature extractor.

        Returns:
            - If `text`: Tokenized text (input_ids, attention_mask).
            - If `audios`: Audio features (input_features) + units (if `return_units=True`).
        """
        sampling_rate = kwargs.pop("sampling_rate", None)

        if text is None and audios is None:
            raise ValueError("You have to specify either text or audios. Both cannot be none.")
        elif text is not None and audios is not None:
            raise ValueError("Text and audios are mutually exclusive. Specify one or the other.")

        # Process text
        if text is not None:
            return self._process_text(
                text, 
                src_lang=src_lang, 
                tgt_lang=tgt_lang,
                return_char_info=return_char_info,
                **kwargs
            )
            # if tgt_lang is not None:
            #     self.tokenizer.tgt_lang = tgt_lang
            # if src_lang is not None:
            #     self.tokenizer.src_lang = src_lang
            # return self.tokenizer(text, **kwargs)

        # Process audio
        else:
            # encoding["attention_mask"].shape torch.Size([1, 24000])
            encoding = self.feature_extractor(audios, sampling_rate=sampling_rate, **kwargs) # torch.Size([1, 24000, 160]) encoding["input_features"].shape

            # Extract units if requested
            return_units = True
            if return_units:
                if self.unit_tokenizer is not None:
                    units = self.unit_tokenizer(audios, sampling_rate=sampling_rate)
                else:
                    units = self._extract_units_with_hubert(audios, sampling_rate)
                encoding["units"] = units # [[96]]
            return encoding

    def _extract_units_with_hubert(
        self,
        audios: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        sampling_rate: int,
    ) -> List[List[int]]:
        """
        Fallback method to extract units using HuBERT if SeamlessM4T's built-in unit tokenizer is missing.
        """
        try:
            from transformers import HubertModel
            hubert = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
            hubert.eval()

            if isinstance(audios, (list, np.ndarray)):
                audios = torch.tensor(audios)

            with torch.no_grad():
                features = hubert(audios).last_hidden_state
                units = features.argmax(dim=-1).cpu().numpy().tolist()

            return units

        except ImportError:
            raise ImportError(
                "`transformers.HubertModel` is required for unit extraction. "
                "Install with `pip install transformers` or use SeamlessM4T's built-in unit tokenizer."
            )

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