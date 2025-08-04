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
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import time
import torch
import numpy as np

def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    emotionembedding_list = []
    try:
        for utt in tqdm(utt_list):
            data = open(utt2wav[utt], 'rb').read()
            data_list.append(data)
        wav_list = [utt2wav[utt] for utt in utt_list]
        text_list = [utt2text[utt] for utt in utt_list]
        spk_list = [utt2spk[utt] for utt in utt_list]
        uttembedding_list = [utt2embedding[utt] for utt in utt_list]
        spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list]
        speech_token_list = [utt2speech_token[utt] for utt in utt_list]
        for utt in utt_list:
            emotionembedding_list.append(utt2emotion_embedding[utt])

        # 检查 emotion_embedding 数据类型并转换为列表
        emotionembedding_list = [x.tolist() if isinstance(x, (torch.Tensor, np.ndarray)) else x for x in emotionembedding_list]

        # 保存到 parquet, utt2parquet_file, spk2parquet_file
        df = pd.DataFrame()
        df['utt'] = utt_list
        df['wav'] = wav_list
        df['audio_data'] = data_list
        df['text'] = text_list
        df['spk'] = spk_list
        df['utt_embedding'] = uttembedding_list
        df['spk_embedding'] = spkembedding_list
        df['emotion_embedding'] = emotionembedding_list  # 新增 emotion_embedding
        df['speech_token'] = speech_token_list

        # 检查 DataFrame 内容
        print("DataFrame columns:", df.columns)
        print("DataFrame emotion_embedding sample:", df['emotion_embedding'].iloc[0])

        df.to_parquet(parquet_file)

        with open(utt2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
        with open(spk2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
        logging.info('spend time {}'.format(time.time() - start_time))
    except Exception as e:
        logging.error(f"Error in job function: {e}", exc_info=True)  # 打印完整的异常信息

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()

    # 加载数据
    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))
    utt2emotion_embedding = torch.load('{}/utt2emotion_embedding.pt'.format(args.src_dir))  # 新增 emotion_embedding
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))
    utts = list(utt2wav.keys())

    # 单进程模式
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        job(utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file)  # 直接调用 job 函数

    # 保存文件列表
    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')