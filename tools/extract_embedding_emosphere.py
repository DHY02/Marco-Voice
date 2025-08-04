"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
from funasr import AutoModel
from torch.nn.functional import pairwise_distance

import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "iic/emotion2vec_plus_seed"
model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)
class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

# device = 'cpu'
emo_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
emo_processor = Wav2Vec2Processor.from_pretrained(emo_model_name)
emo_model = EmotionModel.from_pretrained(emo_model_name).to(device)

# dummy signal
# sampling_rate = 16000
# signal = np.zeros((1, sampling_rate), dtype=np.float32)

def extractor_low_level_embedding(
    x: np.ndarray,
    emo_model,
    sampling_rate: int,
    embeddings: bool = False,
): 
    """Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = emo_processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    emo_model = emo_model.to(device)  
    with torch.no_grad():
        y = emo_model(y)[0 if embeddings else 1]
    # convert to numpy
    y = y.detach().cpu().numpy()
    return y

# (process_func(signal, sampling_rate))
# #  Arousal    dominance valence
# # [[0.5460754  0.6062266  0.40431657]]
# print(process_func(signal, sampling_rate, embeddings=True))

def single_job(utt, emo_model):
    audio, sample_rate = torchaudio.load(utt2wav[utt])
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    low_level_embedding = extractor_low_level_embedding(audio, emo_model, 16000)
    return utt, embedding, low_level_embedding

VAD_Neu_center = torch.tensor([0.4135, 0.5169, 0.3620]).to(device)
VAD_I2I_Ang_center = torch.tensor([0.37068613, 0.4814421, 0.37709341]).to(device)
VAD_I2I_Hap_center = torch.tensor([0.36792182, 0.48303542, 0.34562907]).to(device)
VAD_I2I_Sad_center = torch.tensor([0.48519272, 0.57727589, 0.39524114]).to(device)
VAD_I2I_Sur_center = torch.tensor([0.36877737, 0.48431942, 0.36644742]).to(device)

def convert_tensor_to_polar_coordinates(tensor, q1, q3, emo_id):
        tensor = torch.tensor(tensor)   
        q1 = torch.tensor(q1)
        q3 = torch.tensor(q3)
        r = torch.sqrt(torch.sum(tensor**2, dim=1))
        IQR = (q3 - q1) * 1.5
        max_emo = q3 + IQR
        min_emo = q1 - IQR
        r_clamp = torch.clamp(r, min=min_emo, max=max_emo)
        r_norm = (r_clamp - min_emo) / (max_emo - min_emo)
        theta = torch.acos(tensor[:, 2] / r)  
        phi = torch.atan2(tensor[:, 1], tensor[:, 0]) 

        if r_norm == 0 :
            theta = torch.zeros_like(theta)
            phi = torch.zeros_like(phi)

        if emo_id == "Neutral":
            r_norm = torch.zeros_like(r_norm)
            theta = torch.zeros_like(theta)
            phi = torch.zeros_like(phi)

        polar_coordinates_tensor = torch.stack((r_norm, theta, phi), dim=1)

        return polar_coordinates_tensor.cpu().numpy()

def main(args):
    all_task = [executor.submit(single_job, utt, emo_model) for utt in utt2wav.keys()]
    utt2embedding, spk2embedding, emotion_embedding, low_level_embedding, utt_emo  = {}, {}, {}, {}, {}
    VAD_values, VAD_values2, VAD_sphere, q1, q3 = {}, {}, {}, {}, {}

    for k, v in utt2wav.items():
        utt_emo[k] = utt2wav[k].split("/")[-2]

        emo_id = utt2wav[k].split("/")[-2]
        if emo_id not in VAD_values:
            VAD_values[emo_id] = []
        if emo_id not in VAD_values2:
            VAD_values2[emo_id] = []

    for future in tqdm(as_completed(all_task)):
        utt, embedding, low_embedding = future.result()
        utt2embedding[utt] = embedding
        low_level_embedding[utt] = low_embedding
        spk = utt2spk[utt]
        emo_id = utt2wav[utt].split("/")[-2]
        VAD_values[emo_id].append(low_level_embedding[utt])
        VAD_values2[emo_id].append((low_level_embedding[utt], utt))

        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)

    for emo_id in tqdm(VAD_values):
        print(f"Calculating statistics for emo_id: {emo_id}")
        all_distances = []
        for emo_vad, item_name in VAD_values2[emo_id]:
            if emo_id == "Angry":
                VAD_center = VAD_I2I_Ang_center
            elif emo_id == "Happy":
                VAD_center = VAD_I2I_Hap_center
            elif emo_id == "Neutral":
                VAD_center = VAD_Neu_center
            elif emo_id == "Sad":
                VAD_center = VAD_I2I_Sad_center
            elif emo_id == "Surprise":
                VAD_center = VAD_I2I_Sur_center

            distance_to_neutral = pairwise_distance(torch.tensor(emo_vad).to(device), torch.tensor(VAD_center).unsqueeze(0).to(device), p=2)
            all_distances.append(distance_to_neutral.item())
        if emo_id not in q1:
            q1[emo_id] = []
        if emo_id not in q3:
            q3[emo_id] = []

        q1[emo_id] = np.percentile(all_distances, 25)
        q3[emo_id] = np.percentile(all_distances, 75)

    for k, v in low_level_embedding.items():

        emo_id = utt_emo[k] # .split("/")[-2]
        if emo_id == "Angry":
            VAD_center = VAD_I2I_Ang_center
        elif emo_id == "Happy":
            VAD_center = VAD_I2I_Hap_center
        elif emo_id == "Neutral":
            VAD_center = VAD_Neu_center
        elif emo_id == "Sad":
            VAD_center = VAD_I2I_Sad_center
        elif emo_id == "Surprise":
            VAD_center = VAD_I2I_Sur_center

        emo_VAD = v
        q1_bar = q1[emo_id]
        q3_bar = q3[emo_id]
        re_emo_VAD = emo_VAD - VAD_center.cpu().numpy()
        VAD_sphere[k] = convert_tensor_to_polar_coordinates(re_emo_VAD, q1_bar, q3_bar, emo_id).squeeze().tolist()

    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    # utt2embedding_finale = {}
    for k, v in utt2embedding.items():
        emotion_vector = model.generate(utt2wav[k], output_dir="./outputs", granularity="utterance", extract_embedding=True)
        emotion_embedding[k] = emotion_vector[0]["feats"]
        # print("v:", len(v), "emotion_vector:", emotion_vector)
        # print("v:", v)
        # print("emotion_vector[0]:", emotion_vector[0]["feats"], emotion_vector[0]["feats"].shape)

        # utt2embedding_finale[k] = v + torch.tensor(emotion_vector[0]["feats"])

    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))
    torch.save(emotion_embedding, "{}/emotion_embedding.pt".format(args.dir))
    torch.save(VAD_sphere, "{}/low_level_embedding.pt".format(args.dir))
    torch.save(utt_emo, "{}/utt_emo.pt".format(args.dir))
    torch.save(q1, "{}/q1.pt".format(args.dir))
    torch.save(q3, "{}/q3.pt".format(args.dir))
    torch.save(VAD_values, "{}/VAD_values.pt".format(args.dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
