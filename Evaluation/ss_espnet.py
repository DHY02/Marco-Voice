"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import json
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

def check_gpu_available():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and PyTorch installation.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def process_json_speaker_verification(input_json_path, output_txt_path, threshold=0.5):
    """
    Process JSON file and calculate speaker verification similarity
    
    Args:
        input_json_path: Input JSON file path
        output_txt_path: Output TXT file path
        threshold: Similarity threshold
    """
    check_gpu_available()
    
    sv_pipeline = pipeline(
        task=Tasks.speaker_verification,
        model='iic/speech_eres2net_sv_zh-cn_16k-common',
        model_revision='v1.0.5',
        device='cuda' 
    )
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    
    total_similarity = 0.0
    valid_items = 0
    results = []
    
    sv_pipeline.model.to('cuda')
    
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
            
            try:
                with torch.cuda.device(0): 
                    result = sv_pipeline([ref_audio, gen_audio], thr=threshold)
                
                similarity = result['score']
                is_same_speaker = "yes" if result['text'] == "yes" else "no"
                results.append(f"{utt_id}, {similarity:.4f}, {is_same_speaker}")
                total_similarity += similarity
                valid_items += 1
            except Exception as e:
                print(f"Error processing {utt_id}: {str(e)}")
                continue
    
    avg_similarity = total_similarity / valid_items if valid_items > 0 else 0.0
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("utt_id, similarity, is_same_speaker\n")
        f.write("\n".join(results))
        f.write(f"\n\nAverage similarity: {avg_similarity:.4f}")
        f.write(f"\nSimilarity threshold: {threshold}")
    
    print(f"Processing done, results saved to {output_txt_path}")
    print(f"Average speaker similarity: {avg_similarity:.4f}")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker verification similarity calculation script")
    parser.add_argument("--input_json", required=True, help="Input JSON file path")
    parser.add_argument("--output_txt", required=True, help="Output TXT file path")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Speaker verification threshold (default 0.5)")
    
    args = parser.parse_args()
    
    process_json_speaker_verification(
        args.input_json, 
        args.output_txt,
        args.threshold
    )