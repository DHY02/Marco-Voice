#!/usr/bin/env python3
import torch
import os
import sys
import json
import numpy as np

def print_emotion_info(file_path):
    """
    详细打印指定的emotion_info.pt文件内容
    
    Args:
        file_path: emotion_info.pt文件的路径
    """
    
    print(f"=== Analyzing emotion_info.pt file ===")
    print(f"File path: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ Error: File does not exist!")
        return None
    
    # 获取文件信息
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    if file_size == 0:
        print("❌ File is empty!")
        return None
    
    try:
        # 加载文件
        print(f"\n📁 Loading file...")
        emotion_info = torch.load(file_path, map_location='cpu')
        print(f"✅ File loaded successfully!")
        
        # 打印基本信息
        print(f"\n📊 Basic Information:")
        print(f"Type: {type(emotion_info)}")
        
        if isinstance(emotion_info, dict):
            print(f"Number of keys: {len(emotion_info)}")
            
            if len(emotion_info) == 0:
                print("❌ Dictionary is empty!")
                return emotion_info
            
            print(f"\n🔑 Top-level keys: {list(emotion_info.keys())}")
            
            # 详细分析每个键
            print(f"\n📝 Detailed analysis:")
            for i, (key, value) in enumerate(emotion_info.items()):
                print(f"\n--- Key {i+1}: '{key}' ---")
                print(f"Value type: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"Sub-dictionary with {len(value)} keys: {list(value.keys())}")
                    
                    # 打印前几个子键的详细信息
                    for j, (sub_key, sub_value) in enumerate(list(value.items())[:5]):
                        print(f"  Sub-key '{sub_key}':")
                        print(f"    Type: {type(sub_value)}")
                        
                        if hasattr(sub_value, 'shape'):
                            print(f"    Shape: {sub_value.shape}")
                            print(f"    Dtype: {sub_value.dtype}")
                            if sub_value.numel() <= 10:
                                print(f"    Values: {sub_value.tolist()}")
                            else:
                                print(f"    Min: {sub_value.min().item():.6f}")
                                print(f"    Max: {sub_value.max().item():.6f}")
                                print(f"    Mean: {sub_value.mean().item():.6f}")
                        
                        elif isinstance(sub_value, (list, tuple)):
                            print(f"    Length: {len(sub_value)}")
                            if len(sub_value) > 0:
                                print(f"    First item: {sub_value[0]} (type: {type(sub_value[0])})")
                        
                        else:
                            print(f"    Value: {sub_value}")
                    
                    if len(value) > 5:
                        print(f"  ... and {len(value) - 5} more sub-keys")
                
                elif isinstance(value, (list, tuple)):
                    print(f"List/Tuple with {len(value)} items")
                    if len(value) > 0:
                        print(f"First item type: {type(value[0])}")
                        if hasattr(value[0], 'shape'):
                            print(f"First item shape: {value[0].shape}")
                
                elif hasattr(value, 'shape'):
                    print(f"Tensor shape: {value.shape}")
                    print(f"Tensor dtype: {value.dtype}")
                    if value.numel() <= 20:
                        print(f"Values: {value.tolist()}")
                    else:
                        print(f"Min: {value.min().item():.6f}")
                        print(f"Max: {value.max().item():.6f}")
                        print(f"Mean: {value.mean().item():.6f}")
                
                else:
                    print(f"Value: {value}")
                
                # 只显示前3个键的详细信息，避免输出太长
                if i >= 2:
                    remaining = len(emotion_info) - 3
                    if remaining > 0:
                        print(f"\n... and {remaining} more top-level keys")
                    break
        
        elif isinstance(emotion_info, (list, tuple)):
            print(f"List/Tuple with {len(emotion_info)} items")
            for i, item in enumerate(emotion_info[:3]):
                print(f"Item {i}: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"  Shape: {item.shape}")
            if len(emotion_info) > 3:
                print(f"... and {len(emotion_info) - 3} more items")
        
        elif hasattr(emotion_info, 'shape'):
            print(f"Tensor shape: {emotion_info.shape}")
            print(f"Tensor dtype: {emotion_info.dtype}")
        
        else:
            print(f"Content: {emotion_info}")
        
        # 尝试检查常见的键名
        if isinstance(emotion_info, dict):
            common_keys = ['song', 'speaker', 'emotion', 'angry', 'happy', 'sad', 'neutral', 'surprise']
            print(f"\n🔍 Checking for common keys:")
            for key in common_keys:
                if key in emotion_info:
                    print(f"  ✅ Found: '{key}'")
                else:
                    print(f"  ❌ Missing: '{key}'")
        
        return emotion_info
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数，处理命令行参数"""
    
    if len(sys.argv) > 1:
        # 使用命令行参数指定的文件路径
        file_path = sys.argv[1]
    else:
        # 默认路径
        file_path = "/root/gpufree-data/Marco-Voice/data/spk2neutral.pt"
        print(f"No file path provided, using default: {file_path}")
    
    # 分析文件
    emotion_info = print_emotion_info(file_path)
    
    # 如果文件为空或有问题，搜索其他可能的文件
    if emotion_info is None or (isinstance(emotion_info, dict) and len(emotion_info) == 0):
        print(f"\n🔍 Searching for alternative emotion files...")
        
        search_patterns = [
            "*emotion*.pt",
            "*embedding*.pt", 
            "utt2emotion_embedding.pt",
            "embedding_info.pt"
        ]
        
        found_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if any(pattern.replace("*", "") in file.lower() for pattern in search_patterns) and file.endswith(".pt"):
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path)
                    if file_size > 0:  # 只显示非空文件
                        found_files.append((full_path, file_size))
        
        if found_files:
            print(f"Found {len(found_files)} alternative files:")
            for file_path, file_size in found_files:
                print(f"  📁 {file_path} ({file_size:,} bytes)")
            
            print(f"\nTry running: python {sys.argv[0]} <file_path>")
        else:
            print("No alternative emotion files found.")

if __name__ == "__main__":
    main()