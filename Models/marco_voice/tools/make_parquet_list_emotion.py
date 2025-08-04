# 

import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch

# Global dictionaries will be populated in main.
utt2wav = {}
utt2text = {}
utt2spk = {}
utt2emotion = {}  # New mapping for emotion.
utt2embedding = {}
spk2embedding = {}
utt2speech_token = {}

def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list, desc="Processing utterances"):
        with open(utt2wav[utt], 'rb') as f:
            data = f.read()
        data_list.append(data)
    wav_list = [utt2wav[utt] for utt in utt_list]
    text_list = [utt2text[utt] for utt in utt_list]
    spk_list = [utt2spk[utt] for utt in utt_list]
    uttembedding_list = [utt2embedding[utt] for utt in utt_list]
    spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list]
    speech_token_list = [utt2speech_token[utt] for utt in utt_list]
    # New: extract emotion tag per utterance.
    emotion_list = [utt2emotion[utt] for utt in utt_list]

    df = pd.DataFrame()
    df['utt'] = utt_list
    df['wav'] = wav_list
    df['audio_data'] = data_list
    df['text'] = text_list
    df['spk'] = spk_list
    df['utt_embedding'] = uttembedding_list
    df['spk_embedding'] = spkembedding_list
    df['speech_token'] = speech_token_list
    df['emotion'] = emotion_list  # New column in the parquet file.
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w', encoding="utf-8") as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w', encoding="utf-8") as f:
        json.dump({k: parquet_file for k in set(spk_list)}, f, ensure_ascii=False, indent=2)
    logging.info('Time spent: {:.2f}s'.format(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make parquet files for emotional TTS data")
    parser.add_argument('--num_utts_per_parquet', type=int, default=1000, help='Number of utterances per parquet file')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes for making parquets')
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing mapping files and feature files')
    parser.add_argument('--des_dir', type=str, required=True, help='Destination directory for parquet files')
    args = parser.parse_args()

    # Load mapping files.
    utt2wav = {}
    with open(os.path.join(args.src_dir, 'wav.scp')) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                utt2wav[parts[0]] = parts[1]
    utt2text = {}
    with open(os.path.join(args.src_dir, 'text')) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                utt2text[parts[0]] = ' '.join(parts[1:])
    utt2spk = {}
    with open(os.path.join(args.src_dir, 'utt2spk')) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                utt2spk[parts[0]] = parts[1]
    # Load utterance-to-emotion mapping.
    utt2emotion = {}
    utt2emotion_path = os.path.join(args.src_dir, "utt2emotion")
    if os.path.exists(utt2emotion_path):
        with open(utt2emotion_path, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2emotion[parts[0]] = parts[1]
    else:
        logging.warning(f"utt2emotion file not found in {args.src_dir}")

    # Load additional features.
    utt2embedding = torch.load(os.path.join(args.src_dir, 'utt2embedding.pt'))
    spk2embedding = torch.load(os.path.join(args.src_dir, 'spk2embedding.pt'))
    utt2speech_token = torch.load(os.path.join(args.src_dir, 'utt2speech_token.pt'))
    utts = list(utt2wav.keys())

    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, f'parquet_{i:09d}.tar')
        utt2parquet_file = os.path.join(args.des_dir, f'utt2parquet_{i:09d}.json')
        spk2parquet_file = os.path.join(args.des_dir, f'spk2parquet_{i:09d}.json')
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    pool.close()
    pool.join()

    with open(os.path.join(args.des_dir, 'data.list'), 'w', encoding='utf8') as f1, \
         open(os.path.join(args.des_dir, 'utt2data.list'), 'w', encoding='utf8') as f2, \
         open(os.path.join(args.des_dir, 'spk2data.list'), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')