"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""


import numpy as np
import torch
import soundfile as sf
from hifi_gan_bwe import BandwidthExtender

class AudioBandwidthExtender:
    def __init__(self, pretrained_model_name="hifi-gan-bwe-10-42890e3-vctk-48kHz"):
        # Load the HiFi-GAN Bandwidth Extender model
        self.model = BandwidthExtender.from_pretrained(pretrained_model_name)

    def process_file(self, input_path, output_path=None):
        """
        Process an audio file, extending its bandwidth.

        Args:
            input_path (str): Path to the input WAV file.
            output_path (str, optional): Path to save the processed file.
                                        If None, returns processed audio as numpy array.

        Returns:
            np.ndarray or None: Processed audio (if output_path is None).
        """
        # Load audio
        x, fs = sf.read(input_path, dtype=np.float32)

        # Process with the model
        with torch.no_grad():
            y = self.model(torch.from_numpy(x), fs)
            y_np = y.detach().cpu().numpy()

        # Save or return
        if output_path:
            sf.write(output_path, y_np, int(self.model.sample_rate))
            return None
        else:
            return y_np

    def process_array(self, audio_array, sample_rate):
        """
        Process a numpy array representing audio.

        Args:
            audio_array (np.ndarray): Audio data.
            sample_rate (int): Sample rate of the audio.

        Returns:
            np.ndarray: Processed audio.
        """
        with torch.no_grad():
            y = self.model(torch.from_numpy(audio_array), sample_rate)
            return y.detach().cpu().numpy()
