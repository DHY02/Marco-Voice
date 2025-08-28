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
from funasr import AutoModel

model_id = "iic/emotion2vec_plus_seed"
model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])

    emotion_coficient = model.generate(utt2wav[utt], output_dir="./outputs", granularity="utterance", extract_embedding=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    corpus_name = utt.split('-')[0]
    for item in emotion_coficient:
        item['key'] = corpus_name + "-" + item['key']
    # emotion_embedding[k] = emotion_vector[0]["feats"]
    return utt, embedding, emotion_coficient

def main(args):
    all_task = [executor.submit(single_job, utt) 
           for utt in utt2wav.keys() 
           if utt2wav[utt].split("/")[-2].lower() == "neutral"]
    utt2embedding, spk2embedding, confidence_coefficient = {}, {}, {}
    for future in tqdm(as_completed(all_task)):
        # utt, spk_emb, emotion_info
        utt, embedding, emotion_coefficient = future.result()
        for item in emotion_coefficient:
            for label, score in zip(item['labels'], item['scores']):
                # print("label,", label.split('/')[-1].capitalize(), score)
                
                # emotion 置信度分布
                confidence_coefficient[item['key']] = {label.split('/')[-1].capitalize(): score for label, score in zip(item['labels'], item['scores'])}
       
        # confidence_coefficient = {item['key']: dict(zip(item['labels'], item['scores'])) for item in emotion_coficient[0]}

        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)

    # print("confidence_coefficient:", confidence_coefficient)
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    emotion_list = ["Angry", "Surprise", "Sad", "Happy", "Disgusted", "Fearful", "Playfulness", "Neutral"]
    utt2emo_dic = {}
    utt2neutral_dic = {}
    utt2embedding_dic = {}
    speaker_emotion_dic = defaultdict(lambda: defaultdict(list))
    spk2neutral = {}
    for k, v in utt2wav.items():
        if utt2wav[k].split("/")[-2].lower() != "neutral":
            continue
        v = v.strip()
        utt2emo_dic[k] = v.split("/")[-2]
        speak_id = k.split("_")[0]
        emotion_key = v.split("/")[-2]
        emotion_key = emotion_key.lower().capitalize()
        if emotion_key == "Surprise":
            confidence_coefficient_emotion_key = "Surprised"
        else:
            confidence_coefficient_emotion_key = emotion_key
        
        emotion_coeff = confidence_coefficient[k][confidence_coefficient_emotion_key]
        spk2neutral[speak_id]=(np.array(utt2embedding[k]) * emotion_coeff).tolist()
        print(f"speak_id: {speak_id}, 存在Neutral")
        # 收集spk的各种情感的emb（乘以置信度系数）
        speaker_emotion_dic[speak_id][emotion_key].append((np.array(utt2embedding[k]) * emotion_coeff).tolist())

        if v.split("/")[-2] not in emotion_list and v.split("/")[-2].lower() == "neutral":
            utt2neutral_dic[k] = v.split("/")[-2]

    utt2embedding_finale = {}
    emotion_vector_dic = {}
    emo2embedding = {}
    eps = 1e-8

    for k, v in utt2embedding.items():
        break
        if k in utt2neutral_dic:
            utt2embedding_finale[k] = v
            speaker = k.split("_")[0]
            spk2neutral[speaker] = speaker_emotion_dic[speaker]['Neutral']
        else:
            continue
            k_prefix = k.split("_")[0]
            candidates = [utt for utt in utt2neutral_dic.keys() if utt.split("_")[0] == k_prefix]

            if candidates:
                emotion_vector = torch.zeros(len(v), dtype=torch.float64)
                # 对于该句子的情感，提取情感vector
                for i in range(10):
                    candidate = random.choice(candidates)

                    if candidate not in utt2embedding:
                        continue
                    
                    # 没有减去中性emo embd而是spk embd?
                    neutral_embedding = torch.tensor(utt2embedding[candidate], dtype=torch.float64)

                    speaker = k.split("_")[0]
                    emotion_type = utt2emo_dic[k]
                    if speaker not in speaker_emotion_dic or emotion_type not in speaker_emotion_dic[speaker]:
                        continue

                    emotion_list = speaker_emotion_dic[speaker][emotion_type]
                    if not emotion_list:
                        continue
                    # 随机采样一个该说话人的emotion_embedding
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
            # final emb = spk_emb + emo_vec
            utt2embedding_finale[k] = (v_tensor + emotion_vector).numpy().tolist()

    # utt2embedding_finale = {k: np.array(v) for k, v in utt2embedding_finale.items()}
    # spk2embedding = {k: np.array(v) for k, v in spk2embedding.items()}
    # print("utt2embedding_finale:", utt2embedding_finale)

    # spk_emb + emo_vec
    # torch.save(utt2embedding_finale, "{}/utt2embedding.pt".format(args.dir))
    # torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))
    # torch.save(emo2embedding, "{}/utt2emotion_embedding.pt".format(args.dir))
    # # emo_vec
    # torch.save(emotion_vector_dic, "{}/embedding_info.pt".format(args.dir))
    torch.save(spk2neutral, "{}/spk2neutral.pt".format(args.dir))

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