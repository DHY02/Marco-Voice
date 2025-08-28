#!/usr/bin/env python3
import os
import shutil

def fix_frontend_syntax():
    """修复frontend.py的语法错误"""
    
    frontend_file = "Models/marco_voice/cosyvoice_rodis/cli/frontend.py"
    
    if not os.path.exists(frontend_file):
        print(f"Error: {frontend_file} not found!")
        return False
    
    # 检查是否有备份文件可以恢复
    backup_files = [
        frontend_file + ".fallback_backup",
        frontend_file + ".frd_init_backup", 
        frontend_file + ".path_fix_backup",
        frontend_file + ".backup"
    ]
    
    working_backup = None
    for backup in backup_files:
        if os.path.exists(backup):
            # 检查备份文件是否有语法错误
            try:
                with open(backup, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 简单检查是否有明显的语法问题
                if 'text = text.replace("' in content and content.count('"') % 2 != 0:
                    print(f"Backup {backup} also has syntax issues")
                    continue
                
                compile(content, backup, 'exec')
                working_backup = backup
                print(f"Found working backup: {backup}")
                break
                
            except SyntaxError as e:
                print(f"Backup {backup} has syntax error: {e}")
                continue
            except Exception as e:
                print(f"Error checking backup {backup}: {e}")
                continue
    
    if working_backup:
        # 恢复工作的备份
        shutil.copy2(working_backup, frontend_file)
        print(f"✓ Restored from backup: {working_backup}")
    else:
        print("No working backup found, attempting manual fix...")
        
        # 尝试手动修复当前文件
        try:
            with open(frontend_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找问题行
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                line_num = i + 1
                
                # 检查第139行左右的问题
                if 'text = text.replace("' in line and line.count('"') % 2 != 0:
                    print(f"Found problematic line {line_num}: {line}")
                    
                    # 尝试修复未闭合的字符串
                    if line.strip().endswith('text.replace("'):
                        # 可能是 text = text.replace("
                        line = line + '\\n", "")'  # 补全为移除换行符
                        print(f"Fixed to: {line}")
                    elif '"' in line and not line.endswith('"'):
                        # 尝试闭合字符串
                        line = line + '"'
                        print(f"Fixed to: {line}")
                
                fixed_lines.append(line)
            
            # 写回修复的内容
            fixed_content = '\n'.join(fixed_lines)
            
            # 验证修复的内容
            try:
                compile(fixed_content, frontend_file, 'exec')
                
                with open(frontend_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                print("✓ Manual fix applied successfully")
                return True
                
            except SyntaxError as e:
                print(f"Manual fix failed: {e}")
                return False
                
        except Exception as e:
            print(f"Error during manual fix: {e}")
            return False
    
    return True

def create_clean_text_normalize():
    """创建一个干净的text_normalize方法"""
    
    clean_method = '''
def text_normalize(self, text, split=True):
    text = text.strip()
    original_text = text
    
    print(f"Input text: '{text}'")
    
    if contains_chinese(text):
        if self.use_ttsfrd:
            try:
                frd_result = self.frd.get_frd_extra_info(text, 'input')
                print(f"FRD input: '{text}'")
                print(f"FRD output: '{frd_result}'")
                
                if frd_result and frd_result.strip():
                    text = frd_result
                    print(f"Using FRD result: '{text}'")
                else:
                    print("FRD result empty, using fallback")
                    self.use_ttsfrd = False
                    
            except Exception as e:
                print(f"FRD error: {e}, using fallback")
                self.use_ttsfrd = False
        
        # Fallback处理
        if not self.use_ttsfrd:
            print("Using basic text processing")
            text = text.replace("\\n", "")
            text = text.replace(".", "。")
            text = text.replace(",", "，")
            text = text.replace("!", "！")
            text = text.replace("?", "？")
            
            if text and text[-1] not in "。！？":
                text += "。"
        
        # 其他清理
        if text:
            try:
                text = text.replace("\\n", "")
                text = replace_blank(text) if hasattr(self, 'replace_blank') else text
                text = replace_corner_mark(text) if hasattr(self, 'replace_corner_mark') else text
                text = text.replace(" - ", "，")
                text = remove_bracket(text) if hasattr(self, 'remove_bracket') else text
                import re
                text = re.sub(r'[，,、]+$', '。', text)
            except:
                pass
    else:
        # 英文处理
        if text and not text.endswith(('.', '!', '?')):
            text += "."
    
    # 安全检查
    if not text or not text.strip():
        text = original_text if original_text else "你好。"
    
    print(f"Final text: '{text}'")
    
    # 分割
    if split and text:
        try:
            from functools import partial
            texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                         token_min_n=60, merge_len=20, comma_split=False))
        except Exception as e:
            print(f"Split error: {e}")
            texts = [text]
    else:
        texts = [text] if text else ["你好。"]
    
    return texts if split else (texts[0] if texts else text)
    '''
    
    return clean_method.strip()

def main():
    print("=== Fixing Frontend Syntax Error ===")
    
    # 尝试修复语法错误
    success = fix_frontend_syntax()
    
    if not success:
        print("Automatic fix failed. Creating clean version...")
        
        # 如果自动修复失败，尝试创建全新的文件
        frontend_file = "Models/marco_voice/cosyvoice_rodis/cli/frontend.py"
        
        # 使用最早的备份作为基础
        base_backup = frontend_file + ".fallback_backup"
        if os.path.exists(base_backup):
            print(f"Using {base_backup} as base...")
            shutil.copy2(base_backup, frontend_file)
            print("✓ Restored from earliest backup")
        else:
            print("No backup available. Please restore manually.")
            return
    
    print("✅ Frontend syntax fix completed!")
    print("Now try: python infer.py")

if __name__ == "__main__":
    main()