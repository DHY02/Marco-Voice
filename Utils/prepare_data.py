"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
from collections import defaultdict

def generate_kaldi_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    wav_scp = []
    utt2spk = []
    text = []
    spk2utt_dict = defaultdict(list)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            wav_file = os.path.join(input_dir, file_name)
            base_name = os.path.splitext(file_name)[0]  
            txt_file = os.path.join(input_dir, base_name + ".txt")

            if not os.path.exists(txt_file):
                print(f"Warning: No matching .txt file for {wav_file}. Skipping...")
                continue

            with open(txt_file, "r", encoding="utf-8") as f:
                transcript = f.read().strip() 

            utt_id = base_name
            spk_id = base_name.split("_")[1]  

            wav_scp.append(f"{utt_id} {wav_file}")
            utt2spk.append(f"{utt_id} {spk_id}")
            text.append(f"{utt_id} {transcript}")

            spk2utt_dict[spk_id].append(utt_id)

    spk2utt = [f"{spk_id} {' '.join(utt_list)}" for spk_id, utt_list in spk2utt_dict.items()]

    with open(os.path.join(output_dir, "wav.scp"), "w", encoding="utf-8") as f:
        f.write("\n".join(wav_scp) + "\n")

    with open(os.path.join(output_dir, "utt2spk"), "w", encoding="utf-8") as f:
        f.write("\n".join(utt2spk) + "\n")

    with open(os.path.join(output_dir, "text"), "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")

    with open(os.path.join(output_dir, "spk2utt"), "w", encoding="utf-8") as f:
        f.write("\n".join(spk2utt) + "\n")

    print("Files generated successfully!")
    print(f"-> wav.scp, utt2spk, text, spk2utt are saved in {output_dir}")

import sys

if __name__ == "__main__":

    input_dir = sys.argv[1] # data path, include *.wav and *.txt
    output_dir = sys.argv[2]  # path to save

    generate_kaldi_files(input_dir, output_dir)
