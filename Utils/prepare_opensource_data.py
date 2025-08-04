"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import json
import glob

def main(input_dir):
    # initial date struct
    spk2utt = {}
    utt2spk = {}
    text_dict = {}
    wav_dict = {}
    # reading JSON file
    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        utterance_id = data["_id"]
        speaker_id = data["speaker"]
        text = data["text"].strip()

        json_dir = os.path.dirname(json_file)

        wav_path = os.path.join(json_dir, f"{utterance_id}.wav")
        if not os.path.exists(wav_path):
            print(f"Warning: audio file {wav_path} does not exist, skipping it")
            continue
        wav_dict[utterance_id] = wav_path
        utt2spk[utterance_id] = speaker_id

        if speaker_id not in spk2utt:
            spk2utt[speaker_id] = []
        spk2utt[speaker_id].append(utterance_id)

        text_dict[utterance_id] = text

    with open("wav.scp", "w", encoding="utf-8") as f:
        for utt_id in sorted(wav_dict.keys()):
            f.write(f"{utt_id}\t{wav_dict[utt_id]}\n")

    with open("spk2utt", "w", encoding="utf-8") as f:
        for spk_id in sorted(spk2utt.keys()):
            utts = " ".join(sorted(spk2utt[spk_id]))
            f.write(f"{spk_id}\t{utts}\n")

    with open("utt2spk", "w", encoding="utf-8") as f:
        for utt_id in sorted(utt2spk.keys()):
            f.write(f"{utt_id}\t{utt2spk[utt_id]}\n")

    with open("text", "w", encoding="utf-8") as f:
        for utt_id in sorted(text_dict.keys()):
            f.write(f"{utt_id}\t{text_dict[utt_id]}\n")

    print("File generated：wav.scp, spk2utt, utt2spk, text")
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <input directory>")
        sys.exit(1)
    input_dir = sys.argv[1]
    main(input_dir)

