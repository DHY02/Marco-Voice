import pandas as pd
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm
import glob
import io

def extract_audio_from_local():
    print("开始处理本地CSEMOTIONS数据...")
    
    # 查找所有parquet文件
    parquet_files = glob.glob("/tsdata2/dhy/data/CSEMOTIONS/data/*.parquet")
    if not parquet_files:
        parquet_files = glob.glob("train-*.parquet")
    
    if not parquet_files:
        print("❌ 未找到parquet文件")
        print("请确保在包含parquet文件的目录中运行此脚本")
        return
    
    print(f"✅ 找到 {len(parquet_files)} 个parquet文件:")
    for f in parquet_files:
        print(f"   📁 {f}")
    
    # 创建输出目录
    output_dir = "extracted_audio_hierarchical"
    os.makedirs(output_dir, exist_ok=True)
    
    total_count = 0
    success_count = 0
    
    # 用于收集统计信息
    emotion_stats = {}
    speaker_stats = {}
    speaker_texts = {}  # 存储每个说话人的文本信息

    # 处理每个parquet文件
    for file in parquet_files:
        print(f"\n🔄 正在处理: {file}")
        
        df = pd.read_parquet(file)
        print(f"   📊 数据行数: {len(df)}")
        print(f"   📋 列名: {list(df.columns)}")
        
        # 处理每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取音频"):
            # 获取数据
            audio_data = row['audio']
            text = row['text']
            emotion = row['emotion']
            speaker = row['speaker']

            # 处理音频字节数据
            if isinstance(audio_data, dict) and 'bytes' in audio_data:
                audio_bytes = audio_data['bytes']
                
                if audio_bytes is None:
                    print(f"⚠️  第 {idx} 行音频字节为空，跳过")
                    continue
                
                # 将字节数据转换为BytesIO对象
                audio_io = io.BytesIO(audio_bytes)
                
                try:
                    # 使用soundfile读取音频数据
                    audio_array, sample_rate = sf.read(audio_io)
                    
                    # 确保是1维数组
                    if audio_array.ndim > 1:
                        # 如果是立体声，转换为单声道
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # 创建层次化目录结构：speaker/emotion/
                    speaker_dir = os.path.join(output_dir, speaker)
                    emotion_dir = os.path.join(speaker_dir, emotion.capitalize())  # 首字母大写，与ESD保持一致
                    
                    # 创建目录
                    os.makedirs(emotion_dir, exist_ok=True)
                    
                    # 生成文件名：speaker_序号.wav
                    audio_id = f"{speaker}_{total_count:06d}"
                    filename = f"{audio_id}.wav"
                    filepath = os.path.join(emotion_dir, filename)
                    
                    # 保存音频文件
                    sf.write(filepath, audio_array, sample_rate)
                    
                    # 收集该说话人的文本信息
                    if speaker not in speaker_texts:
                        speaker_texts[speaker] = []
                    
                    # 存储文本信息，格式与ESD一致
                    speaker_texts[speaker].append(f"{audio_id}\t{text}\t{emotion}")
                    
                    # 统计信息
                    emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
                    speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
                    
                    success_count += 1

                except Exception as audio_error:
                    print(f"⚠️  第 {idx} 行音频解码失败: {audio_error}")
                    continue

                total_count += 1
            
            else:
                print(f"⚠️  第 {idx} 行音频格式不支持: {type(audio_data)}")
                continue

    # 为每个说话人创建文本文件
    print("\n📝 创建说话人文本文件...")
    for speaker, texts in speaker_texts.items():
        speaker_dir = os.path.join(output_dir, speaker)
        txt_filepath = os.path.join(speaker_dir, f"{speaker}.txt")
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            for text_line in texts:
                f.write(text_line + '\n')
        
        print(f"   ✅ 创建 {txt_filepath} ({len(texts)} 条记录)")

    print(f"\n🎉 提取完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎵 成功处理音频文件数: {success_count}")
    print(f"🎵 总尝试处理数: {total_count}")
    
    # 统计各情感类别
    print("\n📊 各情感类别统计:")
    for emotion, count in sorted(emotion_stats.items()):
        print(f"   🎭 {emotion}: {count} 个文件")
    
    print(f"\n🎤 各说话人统计:")
    for speaker, count in sorted(speaker_stats.items()):
        print(f"   👤 {speaker}: {count} 个文件")
    
    # 显示最终目录结构示例
    print(f"\n📂 目录结构示例:")
    print(f"{output_dir}/")
    sample_speakers = list(speaker_stats.keys())[:2]  # 显示前两个说话人
    for speaker in sample_speakers:
        print(f"├── {speaker}/")
        print(f"│   ├── {speaker}.txt")
        sample_emotions = set()
        for emotion in emotion_stats.keys():
            emotion_dir = os.path.join(output_dir, speaker, emotion.capitalize())
            if os.path.exists(emotion_dir):
                sample_emotions.add(emotion.capitalize())
        for emotion in sorted(sample_emotions):
            print(f"│   └── {emotion}/")
            print(f"│       └── {speaker}_*.wav")

def check_directory_structure():
    """检查生成的目录结构是否符合预期"""
    output_dir = "extracted_audio_hierarchical"
    
    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在，请先运行提取脚本")
        return
    
    print("🔍 检查目录结构...")
    
    speakers = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for speaker in speakers[:3]:  # 检查前3个说话人
        speaker_dir = os.path.join(output_dir, speaker)
        print(f"\n👤 说话人: {speaker}")
        
        # 检查文本文件
        txt_file = os.path.join(speaker_dir, f"{speaker}.txt")
        if os.path.exists(txt_file):
            print(f"   ✅ 文本文件存在: {speaker}.txt")
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   📄 文本条目数: {len(lines)}")
                if lines:
                    print(f"   📝 示例: {lines[0].strip()}")
        else:
            print(f"   ❌ 文本文件缺失: {speaker}.txt")
        
        # 检查情感目录
        emotions = [d for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir, d))]
        print(f"   🎭 情感目录: {emotions}")
        
        for emotion in emotions:
            emotion_dir = os.path.join(speaker_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            print(f"      📁 {emotion}: {len(wav_files)} 个wav文件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_directory_structure()
    else:
        extract_audio_from_local()
        print("\n" + "="*50)
        print("💡 提示: 运行 'python extract_from_parq.py check' 来检查生成的目录结构")