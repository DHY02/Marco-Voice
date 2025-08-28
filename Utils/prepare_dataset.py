#!/usr/bin/env python3
"""
合并多个情感语音数据集的脚本，生成Marco-Voice训练所需的格式
支持ESD和CSEMOTIONS数据集的合并
"""
import os
import glob
from collections import defaultdict

def get_prompt_token(x):
    return "<|endofprompt|>" if x == "2" else "<endofprompt>"

def prepare_esd_data(esd_root_path, ver):
    """
    处理ESD数据集，返回数据列表
    
    Args:
        esd_root_path: ESD数据集根目录路径
    
    Returns:
        tuple: (wav_scp_lines, text_lines, utt2spk_lines, spk2utt_dict)
    """
    print(f"正在处理ESD数据集: {esd_root_path}")
    
    wav_scp_lines = []
    text_lines = []
    utt2spk_lines = []
    spk2utt_dict = defaultdict(list)
    emo = {"伤心": "Sad", "恐惧":"Fearful", "快乐": "Happy", "惊喜": "Surprise", "生气": "Angry", "中立":"Neutral"} 

    # 遍历所有说话人目录
    speaker_dirs = sorted(glob.glob(os.path.join(esd_root_path, "*")))
    
    for speaker_dir in speaker_dirs:
        if not os.path.isdir(speaker_dir):
            continue
            
        speaker_id = os.path.basename(speaker_dir)
        # 给ESD数据集的说话人ID添加前缀，避免与其他数据集冲突
        speaker_id_prefixed = f"esd-{speaker_id}"
        print(f"  处理说话人: {speaker_id} -> {speaker_id_prefixed}")
        
        # 读取该说话人的文本文件
        text_file = os.path.join(speaker_dir, f"{speaker_id}.txt")
        utt_id_to_text = {}
        
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            utt_id = parts[0]
                            text_content = parts[1]
                            emotion = parts[2]
                            if emo.get(emotion):
                                emotion = emo.get(emotion)
                            utt_id_to_text[utt_id] = emotion + get_prompt_token(ver) + text_content
        
        # 遍历所有情感目录
        emotion_dirs = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        
        for emotion in emotion_dirs:
            emotion_dir = os.path.join(speaker_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
                
            # 获取该情感目录下的所有wav文件
            wav_files = sorted(glob.glob(os.path.join(emotion_dir, "*.wav")))
            
            for wav_file in wav_files:
                # 提取音频ID（文件名不含扩展名）
                wav_filename = os.path.basename(wav_file)
                original_utt_id = wav_filename.replace('.wav', '')
                # 给音频ID添加前缀，确保唯一性
                utt_id = f"esd-{original_utt_id}"
                
                # 获取对应的文本
                text_content = utt_id_to_text.get(original_utt_id, "")
                if not text_content:
                    print(f"    警告: 找不到音频 {original_utt_id} 对应的文本")
                    continue
                
                # 添加到各个列表中
                wav_scp_lines.append(f"{utt_id} {wav_file}")
                text_lines.append(f"{utt_id} {text_content}")
                utt2spk_lines.append(f"{utt_id} {speaker_id_prefixed}")
                spk2utt_dict[speaker_id_prefixed].append(utt_id)
    
    print(f"ESD数据集处理完成: {len(wav_scp_lines)} 条音频, {len(spk2utt_dict)} 个说话人")
    return wav_scp_lines, text_lines, utt2spk_lines, spk2utt_dict

def prepare_csemotions_data(csemotions_root_path, ver):
    """
    处理CSEMOTIONS数据集（层次化结构），返回数据列表
    
    Args:
        csemotions_root_path: CSEMOTIONS数据集根目录路径
    
    Returns:
        tuple: (wav_scp_lines, text_lines, utt2spk_lines, spk2utt_dict)
    """
    print(f"正在处理CSEMOTIONS数据集: {csemotions_root_path}")
    
    wav_scp_lines = []
    text_lines = []
    utt2spk_lines = []
    spk2utt_dict = defaultdict(list)
    
    # 遍历所有说话人目录
    speaker_dirs = sorted(glob.glob(os.path.join(csemotions_root_path, "*")))
    
    for speaker_dir in speaker_dirs:
        if not os.path.isdir(speaker_dir):
            continue
            
        speaker_id = os.path.basename(speaker_dir)
        # 给CSEMOTIONS数据集的说话人ID添加前缀，避免与其他数据集冲突
        speaker_id_prefixed = f"cse-{speaker_id}"
        print(f"  处理说话人: {speaker_id} -> {speaker_id_prefixed}")
        
        # 读取该说话人的文本文件
        text_file = os.path.join(speaker_dir, f"{speaker_id}.txt")
        utt_id_to_text = {}
        
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            utt_id = parts[0]
                            text_content = parts[1]
                            emotion = parts[2]
                            utt_id_to_text[utt_id] = emotion + get_prompt_token(ver) + text_content
        
        # 遍历所有情感目录
        emotion_dirs = [d for d in os.listdir(speaker_dir) 
                       if os.path.isdir(os.path.join(speaker_dir, d)) and d != speaker_id]
        
        for emotion in emotion_dirs:
            if emotion == "Playfulness":
                print(f"忽略Playfulness")
                continue
            emotion_dir = os.path.join(speaker_dir, emotion)
            
            # 获取该情感目录下的所有wav文件
            wav_files = sorted(glob.glob(os.path.join(emotion_dir, "*.wav")))
            
            for wav_file in wav_files:
                # 提取音频ID（文件名不含扩展名）
                wav_filename = os.path.basename(wav_file)
                original_utt_id = wav_filename.replace('.wav', '')
                # 给音频ID添加前缀，确保唯一性
                utt_id = f"cse-{original_utt_id}"
                
                # 获取对应的文本
                text_content = utt_id_to_text.get(original_utt_id, "")
                if not text_content:
                    print(f"    警告: 找不到音频 {original_utt_id} 对应的文本")
                    continue
                
                # 添加到各个列表中
                wav_scp_lines.append(f"{utt_id} {wav_file}")
                text_lines.append(f"{utt_id} {text_content}")
                utt2spk_lines.append(f"{utt_id} {speaker_id_prefixed}")
                spk2utt_dict[speaker_id_prefixed].append(utt_id)
    
    print(f"CSEMOTIONS数据集处理完成: {len(wav_scp_lines)} 条音频, {len(spk2utt_dict)} 个说话人")
    return wav_scp_lines, text_lines, utt2spk_lines, spk2utt_dict

def merge_datasets(output_dir, *dataset_results):
    """
    合并多个数据集的结果
    
    Args:
        output_dir: 输出目录
        *dataset_results: 多个数据集的处理结果
    """
    print("\n开始合并数据集...")
    
    # 初始化合并后的数据结构
    all_wav_scp_lines = []
    all_text_lines = []
    all_utt2spk_lines = []
    all_spk2utt_dict = defaultdict(list)
    
    # 合并所有数据集的结果
    for wav_scp_lines, text_lines, utt2spk_lines, spk2utt_dict in dataset_results:
        all_wav_scp_lines.extend(wav_scp_lines)
        all_text_lines.extend(text_lines)
        all_utt2spk_lines.extend(utt2spk_lines)
        
        for spk, utts in spk2utt_dict.items():
            all_spk2utt_dict[spk].extend(utts)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入wav.scp文件
    with open(os.path.join(output_dir, "wav.scp"), 'w', encoding='utf-8') as f:
        for line in sorted(all_wav_scp_lines):
            f.write(line + '\n')
    
    # 写入text文件
    with open(os.path.join(output_dir, "text"), 'w', encoding='utf-8') as f:
        for line in sorted(all_text_lines):
            f.write(line + '\n')
    
    # 写入utt2spk文件
    with open(os.path.join(output_dir, "utt2spk"), 'w', encoding='utf-8') as f:
        for line in sorted(all_utt2spk_lines):
            f.write(line + '\n')
    
    # 写入spk2utt文件
    with open(os.path.join(output_dir, "spk2utt"), 'w', encoding='utf-8') as f:
        for speaker_id in sorted(all_spk2utt_dict.keys()):
            utts = ' '.join(sorted(all_spk2utt_dict[speaker_id]))
            f.write(f"{speaker_id} {utts}\n")
    
    # 统计信息
    total_utts = len(all_wav_scp_lines)
    total_speakers = len(all_spk2utt_dict)
    
    print(f"\n✅ 数据集合并完成!")
    print(f"📊 合并后统计信息:")
    print(f"   🎵 总音频数量: {total_utts}")
    print(f"   👤 总说话人数量: {total_speakers}")
    print(f"   📁 输出目录: {output_dir}")
    print(f"   📄 生成文件: wav.scp, text, utt2spk, spk2utt")
    
    # 按数据集前缀统计
    esd_count = sum(1 for line in all_utt2spk_lines if line.split()[1].startswith('esd_'))
    cse_count = sum(1 for line in all_utt2spk_lines if line.split()[1].startswith('cse_'))
    
    print(f"\n📈 各数据集音频数量:")
    if esd_count > 0:
        print(f"   🎭 ESD数据集: {esd_count} 条音频")
    if cse_count > 0:
        print(f"   🎭 CSEMOTIONS数据集: {cse_count} 条音频")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='合并ESD和CSEMOTIONS数据集用于Marco-Voice训练')
    parser.add_argument('--ver', 
                        help='cosyvoice1 or 2')
    parser.add_argument('--esd_dir', 
                        help='ESD数据集根目录路径')
    parser.add_argument('--csemotions_dir', 
                        help='CSEMOTIONS数据集根目录路径')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 检查至少提供一个数据集
    if not args.esd_dir and not args.csemotions_dir:
        print("❌ 错误: 至少需要提供一个数据集目录 (--esd_dir 或 --csemotions_dir)")
        return
    
    # 处理各个数据集
    dataset_results = []
    
    if args.esd_dir:
        if os.path.exists(args.esd_dir):
            esd_result = prepare_esd_data(args.esd_dir, args.ver)
            dataset_results.append(esd_result)
        else:
            print(f"⚠️  警告: ESD数据集目录不存在: {args.esd_dir}")
    
    if args.csemotions_dir:
        if os.path.exists(args.csemotions_dir):
            cse_result = prepare_csemotions_data(args.csemotions_dir, args.ver)
            dataset_results.append(cse_result)
        else:
            print(f"⚠️  警告: CSEMOTIONS数据集目录不存在: {args.csemotions_dir}")
    
    # 合并数据集
    if dataset_results:
        merge_datasets(args.output_dir, *dataset_results)
    else:
        print("❌ 错误: 没有找到可用的数据集")
    print(f"当前prompt 格式适合cosyvocie{args.ver}")
if __name__ == "__main__":
    main()