# 

#!/usr/bin/env python3
import argparse
import logging
import os
from tqdm import tqdm

logger = logging.getLogger()

def main():
    # Dictionaries for mapping information.
    utt2wav, utt2text, utt2spk, spk2utt, utt2emotion = {}, {}, {}, {}, {}

    # Each subdirectory in src_dir represents a speaker.
    speakers = [d for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))]
    for spk in speakers:
        spk_dir = os.path.join(args.src_dir, spk)
        metadata_file = os.path.join(spk_dir, "metadata.txt")
        if not os.path.exists(metadata_file):
            logger.warning(f"{metadata_file} does not exist.")
            continue

        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Processing speaker {spk}"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                logger.warning(f"Malformed line in {metadata_file}: {line}")
                continue

            utt_id_raw = parts[0].strip()
            transcription = parts[1].strip()
            emotion = parts[2].strip()
            # Form a unique utterance ID (combining speaker, emotion, and raw id)
            utt = f"{spk}_{emotion}_{utt_id_raw}"
            # Audio file assumed to be in the emotion subfolder: <src_dir>/<speaker>/<emotion>/<utterance_id_raw>.wav
            wav_path = os.path.join(spk_dir, emotion, utt_id_raw + ".wav")
            if not os.path.exists(wav_path):
                logger.warning(f"{wav_path} does not exist for utterance {utt}.")
                continue

            utt2wav[utt] = os.path.abspath(wav_path)
            utt2text[utt] = transcription
            utt2spk[utt] = spk
            utt2emotion[utt] = emotion

            spk2utt.setdefault(spk, []).append(utt)

    # Ensure destination directory exists.
    os.makedirs(args.des_dir, exist_ok=True)

    # Write mapping files.
    with open(os.path.join(args.des_dir, "wav.scp"), "w", encoding="utf-8") as f:
        for utt, wav in utt2wav.items():
            f.write(f"{utt} {wav}\n")
    with open(os.path.join(args.des_dir, "text"), "w", encoding="utf-8") as f:
        for utt, txt in utt2text.items():
            f.write(f"{utt} {txt}\n")
    with open(os.path.join(args.des_dir, "utt2spk"), "w", encoding="utf-8") as f:
        for utt, spk in utt2spk.items():
            f.write(f"{utt} {spk}\n")
    with open(os.path.join(args.des_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")
    with open(os.path.join(args.des_dir, "utt2emotion"), "w", encoding="utf-8") as f:
        for utt, emo in utt2emotion.items():
            f.write(f"{utt} {emo}\n")
    logger.info("Emotional speech data preparation completed.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare emotional speech data for training")
    parser.add_argument('--src_dir', type=str, required=True,
                        help="Source directory containing emotional data organized by speaker")
    parser.add_argument('--des_dir', type=str, required=True,
                        help="Destination directory for prepared mapping files")
    args = parser.parse_args()
    main()