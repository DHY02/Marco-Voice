# CSEMOTIONS: High-Quality Mandarin Emotional Speech Dataset

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**CSEMOTIONS** is a high-quality Mandarin emotional speech dataset designed for expressive speech synthesis, emotion recognition, and voice cloning research. The dataset contains studio-quality recordings from six professional voice actors across seven carefully curated emotional categories, supporting research in controllable and natural language speech generation.


## Dataset Summary

- **Name:** CSEMOTIONS
- **Total Duration:** ~10 hours
- **Speakers:** 10 (5 male, 5 female) native Mandarin speakers, all professional voice actors
- **Emotions:** Neutral, Happy, Angry, Sad, Surprise, Disgust, Fear
- **Language:** Mandarin Chinese
- **Sampling Rate:** 48kHz, 24-bit PCM
- **Recording Setting:** Professional studio environment
- **Evaluation Prompts:** 100 per emotion, in both English and Chinese

## Dataset Structure

Each data sample includes:

- **audio**: The speech waveform (48kHz, 24-bit, WAV)
- **transcript**: The transcribed sentence in Mandarin
- **emotion**: One of {neutral, happy, angry, sad, surprise, disgust, fear}
- **speaker_id**: An anonymized speaker identifier (e.g., `S01`)
- **gender**: Male/Female
- **prompt_id**: Unique identifier for each utterance


## Intended Uses

CSEMOTIONS is intended for:

- Expressive text-to-speech (TTS) and voice cloning systems
- Speech emotion recognition (SER) research
- Cross-lingual and cross-emotional synthesis experiments
- Benchmarking emotion transfer or disentanglement models

## Dataset Details

| Property                | Value                                 |
|-------------------------|---------------------------------------|
| Total audio hours       | ~10                                   |
| Number of speakers      | 6 (3♂, 3♀, anonymized IDs)           |
| Emotions                | Neutral, Happy, Angry, Sad, Surprise, Disgust, Fear |
| Language                | Mandarin Chinese                      |
| Format                  | WAV, mono, 48kHz/24bit                |
| Studio quality          | Yes                                   |

## Download and Usage

To use CSEMOTIONS with [🤗 Datasets](https://huggingface.co/docs/datasets):

```python
from datasets import load_dataset

dataset = load_dataset("AIDC-AI/csemotions")
```

## Acknowledgements

We would like to thank our professional voice actors and the recording studio staff for their contributions.


## License

The project is licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0, SPDX-License-identifier: Apache-2.0).

## Disclaimer
                                     
We used compliance checking algorithms during the training process, to ensure the compliance of the trained model and dataset to the best of our ability. Due to the complexity of the data and the diversity of language model usage scenarios, we cannot guarantee that the dataset is completely free of copyright issues or improper content. If you believe anything infringes on your rights or contains improper content, please contact us, and we will promptly address the matter.
                                     
---
