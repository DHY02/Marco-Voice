#!/usr/bin/env python3
"""
Extract Chinese/English subset from ASVP-ESD emotional speech dataset
and organize it in the required hierarchical structure for Marco-Voice training.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import librosa
import soundfile as sf

def get_emotion_mapping():
    """
    ASVP-ESD emotion code mapping
    Based on RAVDESS-style encoding where third field indicates emotion
    """
    return {
        "01": "neutral",      # 中性
        "03": "happy",        # 快乐  
        "04": "sad",          # 伤心
        "05": "angry",        # 生气
        "06": "fearful",      # 恐惧
        "07": "disgust",      # 厌恶
        "08": "surprised"     # 惊喜
    }

def get_target_emotions():
    """Target emotions we want to extract"""
    return {
        "neutral": "Neutral",
        "sad": "Sad", 
        "fearful": "Fearful",
        "happy": "Happy",
        "surprised": "Surprise", 
        "angry": "Angry",
        "disgust": "Disgust"  # Only if a speaker has this emotion
    }

def extract_info_from_filename(filename):
    """
    Extract information from ASVP-ESD filename
    Format: 03-01-XX-01-XX-XX-03-03-01-XX.wav
    Where the 3rd field (index 2) indicates emotion
    And 6th field (index 5) indicates actor
    """
    parts = filename.replace('.wav', '').split('-')
    if len(parts) >= 6:
        emotion_code = parts[2]
        actor_id = parts[5]
        return emotion_code, actor_id
    return None, None

def create_speaker_text_content(speaker_id, audio_files, target_emotions, emotion_mapping):
    """
    Create text content for speaker's .txt file
    Format: speaker_uttid\ttext\temotion
    """
    lines = []
    
    # Generate simple emotional texts based on emotion
    emotion_texts = {
        "Neutral": "这是一段中性的语音内容",
        "Happy": "今天天气真好，我感到很开心",
        "Sad": "这让我感到很失望和难过", 
        "Angry": "这种行为让我非常愤怒",
        "Fearful": "我对这种情况感到很害怕",
        "Surprise": "天呐，这真是太令人惊讶了",
        "Disgust": "这种情况让我感到厌恶"
    }
    
    for audio_file in audio_files:
        # Extract emotion and create utterance ID
        emotion_code, _ = extract_info_from_filename(os.path.basename(audio_file))
        if emotion_code and emotion_code in emotion_mapping:
            emotion_name = emotion_mapping[emotion_code]
            if emotion_name in target_emotions:
                target_emotion = target_emotions[emotion_name]
                
                # Create utterance ID
                base_name = os.path.basename(audio_file).replace('.wav', '')
                utt_id = f"{speaker_id}_{base_name}"
                
                # Get text for this emotion
                text = emotion_texts.get(target_emotion, "这是一段语音内容")
                
                lines.append(f"{utt_id}\t{text}\t{emotion_name.lower()}")
    
    return "\n".join(lines)

def process_asvpesd_dataset(json_file, output_dir):
    """
    Process ASVP-ESD dataset and create hierarchical structure
    Only include speakers who have disgust emotion
    
    Args:
        json_file: Path to asvpesd.json
        output_dir: Output directory for processed data
    """
    
    print(f"Loading dataset information from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    emotion_mapping = get_emotion_mapping()
    target_emotions = get_target_emotions()
    
    # Group files by speaker and emotion
    speaker_emotions = defaultdict(lambda: defaultdict(list))
    
    for item_id, item_info in dataset.items():
        wav_path = item_info['wav']
        emotion_code = item_info['emo']
        
        # Extract actor info from path or filename
        if 'actor_' in wav_path:
            actor_id = wav_path.split('actor_')[1].split('/')[0]
            speaker_id = f"speaker_{actor_id.zfill(3)}"
        else:
            # Fallback: extract from filename
            emotion_code_from_name, actor_id = extract_info_from_filename(os.path.basename(wav_path))
            if actor_id:
                speaker_id = f"speaker_{actor_id.zfill(3)}"
            else:
                continue
        
        # Map emotion code to emotion name
        if emotion_code in emotion_mapping:
            emotion_name = emotion_mapping[emotion_code]
            if emotion_name in target_emotions:
                speaker_emotions[speaker_id][emotion_name].append(wav_path)
    
    print(f"Found {len(speaker_emotions)} speakers")
    
    # Filter speakers - ONLY keep those with disgust emotion
    valid_speakers = {}
    speakers_with_disgust = []
    
    for speaker_id, emotions in speaker_emotions.items():
        # Check if speaker has disgust emotion
        if 'disgust' in emotions and len(emotions['disgust']) > 0:
            speakers_with_disgust.append(speaker_id)
            valid_speakers[speaker_id] = emotions
            print(f"✓ Including {speaker_id} (has disgust emotion: {len(emotions['disgust'])} files)")
        else:
            print(f"✗ Excluding {speaker_id} (no disgust emotion)")
    
    print(f"\nSelected {len(valid_speakers)} speakers for extraction")
    print(f"All selected speakers have disgust emotion: {speakers_with_disgust}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each valid speaker
    for speaker_id, emotions in valid_speakers.items():
        speaker_dir = output_path / speaker_id
        speaker_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing {speaker_id}:")
        
        all_audio_files = []
        
        # Create emotion directories and copy files
        for emotion_name, files in emotions.items():
            if emotion_name in target_emotions and len(files) > 0:
                target_emotion = target_emotions[emotion_name]
                emotion_dir = speaker_dir / target_emotion
                emotion_dir.mkdir(exist_ok=True)
                
                print(f"  {target_emotion}: {len(files)} files")
                
                # Copy audio files
                for i, src_path in enumerate(files):
                    if os.path.exists(src_path):
                        # Create new filename
                        base_name = os.path.basename(src_path).replace('.wav', '')
                        new_filename = f"{speaker_id}_{base_name}.wav"
                        dst_path = emotion_dir / new_filename
                        
                        try:
                            # Copy and potentially resample audio
                            audio, sr = librosa.load(src_path, sr=None)
                            if sr != 22050:  # Resample to 22050 Hz if needed
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                                sr = 22050
                            
                            sf.write(dst_path, audio, sr)
                            all_audio_files.append(str(dst_path))
                            
                        except Exception as e:
                            print(f"    Warning: Failed to process {src_path}: {e}")
        
        # Create speaker text file
        if all_audio_files:
            speaker_txt = speaker_dir / f"{speaker_id}.txt"
            txt_content = create_speaker_text_content(
                speaker_id, all_audio_files, target_emotions, emotion_mapping
            )
            
            with open(speaker_txt, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            print(f"  Created {speaker_txt} with {len(all_audio_files)} entries")
    
    # Print summary
    print(f"\n=== Extraction Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Total speakers extracted: {len(valid_speakers)}")
    
    # Print statistics
    emotion_stats = defaultdict(int)
    total_files = 0
    
    for speaker_dir in output_path.iterdir():
        if speaker_dir.is_dir():
            for emotion_dir in speaker_dir.iterdir():
                if emotion_dir.is_dir() and emotion_dir.name in target_emotions.values():
                    count = len(list(emotion_dir.glob('*.wav')))
                    emotion_stats[emotion_dir.name] += count
                    total_files += count
    
    print(f"Total audio files: {total_files}")
    print("\n📊 Emotion distribution:")
    for emotion, count in sorted(emotion_stats.items()):
        print(f"   🎭 {emotion}: {count} files")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ASVP-ESD emotional speech subset")
    parser.add_argument("--json_file", required=True, 
                       help="Path to asvpesd.json file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for extracted data")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file {args.json_file} not found")
        return
    
    process_asvpesd_dataset(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()