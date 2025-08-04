"""
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import sys
import os
import json
from collections import defaultdict
from funasr import AutoModel

def process_directory(json_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue
            
        input_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        with open(input_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {json_file}, skipping")
                continue
                
        for utt_id in data:
            entry = data[utt_id]
            audio_path = entry.get("audio_path", "")
            true_label = entry.get("style_label", "").lower()  
            
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue
                
            try:
                predicted_result = model.generate(
                    audio_path,
                    output_dir="./outputs",
                    granularity="utterance",
                    extract_embedding=True
                )[0]
                
                scores = predicted_result["scores"]
                best_idx = scores.index(max(scores))
                predicted_label = predicted_result["labels"][best_idx].split("/")[0].lower()
                
            except Exception as e:
                print(f"Prediction failed for {audio_path}: {str(e)}")
                predicted_label = "error"
            
            entry["predicted_style"] = predicted_label
            
            if true_label and predicted_label != "error":
                total_counts[true_label] += 1
                if predicted_label == true_label:
                    correct_counts[true_label] += 1
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    return correct_counts, total_counts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_json_dir> <output_json_dir>")
        sys.exit(1)
        
    json_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    model = AutoModel(
        model="iic/emotion2vec_plus_large",
        hub="ms"  
    )
    
    correct_counts, total_counts = process_directory(json_dir, output_dir, model)
    
    print("\nEmotion Label\tAccuracy\tSamples")
    all_labels = sorted(set(total_counts.keys()).union(correct_counts.keys()))
    
    total_correct = 0
    total_samples = 0
    
    for label in all_labels:
        correct = correct_counts.get(label, 0)
        total = total_counts.get(label, 0)
        acc = correct / total if total > 0 else 0.0
        
        print(f"{label:<15}\t{acc:.2%}\t\t{total}")
        total_correct += correct
        total_samples += total
        
    if total_samples > 0:
        print(f"\nOverall Accuracy: {total_correct/total_samples:.2%} ({total_correct}/{total_samples})")
    else:
        print("\nNo valid samples processed")
