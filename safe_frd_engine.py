
import os
import ttsfrd

class SafeFRDEngine:
    def __init__(self, resource_path=None):
        self.engine = None
        self.initialized = False
        self.resource_path = resource_path or "/root/gpufree-data/Marco-Voice/pretrained_models/CosyVoice-ttsfrd/resource"
        
    def initialize(self):
        """安全初始化FRD"""
        
        try:
            self.engine = ttsfrd.TtsFrontendEngine()
            
            # 尝试不同的资源路径
            paths_to_try = [
                self.resource_path,
                os.path.join(self.resource_path, "ws_zhsc"),
                os.path.join(self.resource_path, "ws"),
                os.path.join(self.resource_path, "ws_chhk"),
            ]
            
            for path in paths_to_try:
                if os.path.exists(path):
                    print(f"Trying FRD path: {path}")
                    try:
                        result = self.engine.initialize(path)
                        if result:
                            # 测试是否真的能处理文本
                            test_result = self.engine.get_frd_extra_info("测试", 'input')
                            if test_result and test_result.strip():
                                print(f"✓ FRD working with path: {path}")
                                self.initialized = True
                                return True
                            else:
                                print(f"FRD initialized but text processing failed: {path}")
                    except Exception as e:
                        print(f"Error with path {path}: {e}")
            
            print("❌ All FRD initialization attempts failed")
            return False
            
        except Exception as e:
            print(f"FRD initialization error: {e}")
            return False
    
    def get_frd_extra_info(self, text, mode='input'):
        """安全的文本处理"""
        
        if not self.initialized or not self.engine:
            print("FRD not initialized, returning original text")
            return text
        
        try:
            result = self.engine.get_frd_extra_info(text, mode)
            if result and result.strip():
                return result
            else:
                print(f"FRD returned empty result for: {text}")
                return text
        except Exception as e:
            print(f"FRD processing error: {e}")
            return text

# 测试包装器
if __name__ == "__main__":
    frd = SafeFRDEngine()
    if frd.initialize():
        test_texts = ["今天天气很好", "你好世界", "123456"]
        for text in test_texts:
            result = frd.get_frd_extra_info(text)
            print(f"'{text}' -> '{result}'")
    else:
        print("FRD initialization failed")
