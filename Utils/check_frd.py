#!/usr/bin/env python3
import os
import sys

def check_frd_resources():
    """检查FRD相关的资源文件"""
    
    print("=== Checking FRD Resources ===")
    
    # 可能的FRD资源位置
    possible_paths = [
        "Models/marco_voice/../../pretrained_models/CosyVoice-ttsfrd/resource",
        "Models/marco_voice/pretrained_models/CosyVoice-ttsfrd/resource", 
        "pretrained_models/CosyVoice-ttsfrd/resource",
        "Models/marco_voice/cosyvoice_rodis/../../pretrained_models/CosyVoice-ttsfrd/resource"
    ]
    
    print("Checking possible FRD resource paths:")
    frd_found = False
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {path}")
        print(f"    Absolute: {abs_path}")
        print(f"    Exists: {exists}")
        
        if exists:
            frd_found = True
            print(f"    Contents:")
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    is_dir = os.path.isdir(item_path)
                    size = "DIR" if is_dir else f"{os.path.getsize(item_path)} bytes"
                    print(f"      {item} ({size})")
            except Exception as e:
                print(f"      Error reading directory: {e}")
        print()
    
    if not frd_found:
        print("❌ No FRD resource directory found!")
        return False
    
    # 检查必要的FRD文件
    print("Checking for essential FRD files:")
    essential_files = [
        "jieba_dict.txt",
        "lexicon.txt", 
        "phone.txt",
        "polyphone.txt",
        "word.txt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\nIn {path}:")
            for file in essential_files:
                file_path = os.path.join(path, file)
                exists = os.path.exists(file_path)
                size = os.path.getsize(file_path) if exists else 0
                print(f"  {file}: {'✓' if exists else '✗'} ({size} bytes)")
    
    return True

def check_ttsfrd_installation():
    """检查ttsfrd包的安装"""
    
    print("\n=== Checking ttsfrd Installation ===")
    
    try:
        import ttsfrd
        print(f"✓ ttsfrd imported successfully")
        print(f"  Version: {getattr(ttsfrd, '__version__', 'unknown')}")
        
        # 检查ttsfrd的属性和方法
        print(f"  Available attributes:")
        for attr in dir(ttsfrd):
            if not attr.startswith('_'):
                print(f"    {attr}")
                
    except ImportError as e:
        print(f"❌ ttsfrd import failed: {e}")
        return False
    
    try:
        import ttsfrd_dependency
        print(f"✓ ttsfrd_dependency imported successfully")
    except ImportError as e:
        print(f"⚠️  ttsfrd_dependency import failed: {e}")
    
    return True

def test_frd_initialization():
    """测试FRD初始化"""
    
    print("\n=== Testing FRD Initialization ===")
    
    # 添加路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, 'Models', 'marco_voice'))
    
    try:
        from cosyvoice_rodis.cli.frontend import CosyVoiceFrontEnd
        
        # 尝试创建frontend
        print("Attempting to create CosyVoiceFrontEnd...")
        
        # 这里需要提供必要的参数
        # 你可能需要根据实际情况调整这些参数
        frontend = CosyVoiceFrontEnd(
            get_tokenizer=None,  # 这些需要根据实际配置调整
            fe_config=None,
            config=None
        )
        
        print("✓ FRD initialization successful!")
        
        # 测试文本处理
        test_text = "今天天气很好"
        result = frontend.text_normalize(test_text)
        print(f"Test text: {test_text}")
        print(f"Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ FRD initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_frd_resources()
    check_ttsfrd_installation() 
    # test_frd_initialization()  # 需要调整参数后再启用