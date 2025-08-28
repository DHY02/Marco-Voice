# 

#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Rotational Emotion Embedding Integration by SV model
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
import random
from collections import defaultdict
import numpy as np

def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    return utt, embedding

def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2embedding, spk2embedding = {}, {}
    for future in tqdm(as_completed(all_task)):
        utt, embedding = future.result()
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    # mean spk_embedding
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    emotion_list = ["Angry", "Surprise", "Sad", "Happy", "Disgusted", "Fearful"]
    utt2emo_dic = {}
    utt2neutral_dic = {}
    utt2embedding_dic = {}
    speaker_emotion_dic = defaultdict(lambda: defaultdict(list))

    for k, v in utt2wav.items():
        v = v.strip()
        utt2emo_dic[k] = v.split("/")[-2]
        speak_id = k.split("_")[0]
        emotion_key = v.split("/")[-2]
        speaker_emotion_dic[speak_id][emotion_key].append(utt2embedding[k])
        if v.split("/")[-2] not in emotion_list:
            utt2neutral_dic[k] = v.split("/")[-2]

    utt2embedding_finale = {}
    emotion_vector_dic = {}
    emo2embedding = {}
    eps = 1e-8

    for k, v in utt2embedding.items():
        # if k in utt2neutral_dic:
        #     utt2embedding_finale[k] = v
        # else:
        k_prefix = k.split("_")[0]
        # neutral utt of the same spk as k
        candidates = [utt for utt in utt2neutral_dic.keys() if utt.split("_")[0] == k_prefix]

        if candidates:
            emotion_vector = torch.zeros(len(v), dtype=torch.float64)

            for i in range(10):
                candidate = random.choice(candidates)

                if candidate not in utt2embedding:
                    continue

                #  neutral_embedding and emotion_embedding
                neutral_embedding = torch.tensor(utt2embedding[candidate], dtype=torch.float64)

                # check if  emotion_embedding exists
                speaker = k.split("_")[0]
                emotion_type = utt2emo_dic[k]
                if speaker not in speaker_emotion_dic or emotion_type not in speaker_emotion_dic[speaker]:
                    continue
                
                # all the embs of this emotion
                emotion_list = speaker_emotion_dic[speaker][emotion_type]
                if not emotion_list:
                    continue

                # randomly choose one emotion_embedding
                emotion_embedding = torch.tensor(emotion_list[random.randint(0, len(emotion_list) - 1)], dtype=torch.float64)

                diff = emotion_embedding - neutral_embedding
                norm = torch.norm(diff)
                if norm > eps:  
                    emotion_vector += diff / (norm + eps)

        emotion_vector /= 10

        if torch.isnan(emotion_vector).any() or torch.isinf(emotion_vector).any():
            print(f"Warning: emotion_vector contains NaN or Inf values for key {k}")
            emotion_vector = torch.zeros_like(emotion_vector)  

        speaker = k.split("_")[0]
        emotion_type = utt2emo_dic[k]
        if speaker not in emotion_vector_dic:
            emotion_vector_dic[speaker] = {}
        emotion_vector_dic[speaker][emotion_type] = emotion_vector  
        emo2embedding[k] = emotion_vector.tolist()
        v_tensor = torch.tensor(v, dtype=torch.float64)
        utt2embedding_finale[k] = (v_tensor + emotion_vector).numpy().tolist()

    # utt2embedding_finale = {k: np.array(v) for k, v in utt2embedding_finale.items()}
    # spk2embedding = {k: np.array(v) for k, v in spk2embedding.items()}
    # spk emb of the utt
    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))
    # list of spk emb
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))
    torch.save(emo2embedding, "{}/utt2emotion_embedding.pt".format(args.dir))
    torch.save(emotion_vector_dic, "{}/embedding_info.pt".format(args.dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.strip().split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:
        for l in f:
            l = l.strip().split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)