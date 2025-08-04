"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import json
import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": str(device)} 
).eval()

def extract_embedding(audio_path):
    """Extract embedding from audio"""
    try:
        signal, fs = torchaudio.load(audio_path)
        
        signal = signal.to(device)
        
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)
        
        return embeddings.squeeze(0).cpu().numpy()
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def compute_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    
    embedding1 = np.asarray(embedding1).reshape(1, -1)
    embedding2 = np.asarray(embedding2).reshape(1, -1)
    
    return cosine_similarity(embedding1, embedding2)[0][0]

def process_json(input_json_path, output_txt_path):
    """
    Process JSON file and calculate speaker similarity
    
    Args:
        input_json_path: Input JSON file path
        output_txt_path: Output TXT file path
    """
    # Read JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    
    total_similarity = 0.0
    valid_items = 0
    results = []
    
    for data in data_all:
        for utt_id, item in tqdm(data.items(), desc="Processing audio files"):
            ref_audio = item.get("ref_audio", "")
            gen_audio = item.get("generate_audio", "")
            
            if not ref_audio or not gen_audio:
                print(f"Skipping entry {utt_id}: Missing ref_audio or generate_audio field")
                continue
            
            if not os.path.exists(ref_audio):
                print(f"Reference audio does not exist: {ref_audio}")
                continue
            if not os.path.exists(gen_audio):
                print(f"Generated audio does not exist: {gen_audio}")
                continue
            
            ref_embedding = extract_embedding(ref_audio)
            gen_embedding = extract_embedding(gen_audio)
            
            if ref_embedding is None or gen_embedding is None:
                continue
            
            similarity = compute_similarity(ref_embedding, gen_embedding)
            if similarity is None:
                continue
            
            results.append(f"{utt_id}, {similarity:.4f}")
            total_similarity += similarity
            valid_items += 1
    
    avg_similarity = total_similarity / valid_items if valid_items > 0 else 0.0
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("utt_id, similarity\n")
        f.write("\n".join(results))
        f.write(f"\n\nAverage Similarity: {avg_similarity:.4f}")
    
    print(f"Processing done, results saved to {output_txt_path}")
    print(f"Average speaker similarity: {avg_similarity:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker similarity computation script")
    parser.add_argument("--input_json", required=True, help="Input JSON file path")
    parser.add_argument("--output_txt", required=True, help="Output TXT file path")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    process_json(args.input_json, args.output_txt)