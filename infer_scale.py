import sys
import os
import json
import torch
import torchaudio
from pathlib import Path
import hashlib

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
marco_voice_path = os.path.join(current_dir, 'Models', 'marco_voice')
sys.path.insert(0, marco_voice_path)
matcha_tts_path = os.path.join(current_dir, 'Models', 'marco_voice', 'third_party', 'Matcha-TTS')
if matcha_tts_path not in sys.path:
    sys.path.insert(0, matcha_tts_path)

from Models.marco_voice.cosyvoice_rodis.cli.cosyvoice import CosyVoice
from Models.marco_voice.cosyvoice_rodis.utils.file_utils import load_wav

def load_prompt_info(prompt_id, prompt_dir):
    """加载prompt音频的信息"""
    json_path = os.path.join(prompt_dir, f"{prompt_id}.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Prompt info file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 解码text字段
    prompt_text = data['text']
    if isinstance(prompt_text, str) and '\\u' in prompt_text:
        prompt_text = prompt_text.encode().decode('unicode_escape')
    
    return prompt_text

def batch_inference():
    # 配置
    model_path = 'trained_models/CosyVoice-300M-emo-1'
    prompt_dir = '/root/gpufree-data/Marco-Voice/hw-prompt'
    output_dir = 'batch_inference_results_emo_1'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("Loading Marco-Voice model...")
    model = CosyVoice(model_path, load_jit=False, load_onnx=False, fp16=False)
    
    # 情感映射
    emo_mapping = {
        "伤心": "Sad", 
        "恐惧": "Fearful", 
        "快乐": "Happy", 
        "惊喜": "Surprise", 
        "生气": "Angry",
        "中性": "Neutral",
        "厌烦": "Disgust"
    }
    
    # 加载emotion info
    emotion_info_dict = torch.load("data/embedding_info.pt")
    neutral_info_dict = torch.load("data/spk2neutral.pt")
    # 待合成文本
    texts = {
        "喜": [
            "哇塞！美食节！简直就是天堂啊！各种各样的好吃的都汇聚在那儿，想想都要流口水啦！我得赶紧去尝尝，说不定还能发现好多宝藏美食呢！哈哈，简直太棒啦！",
            "天呐，路边那家小店居然还记得我喜欢的拿铁比例，第一口就回想起留学时的春天，幸福感爆棚。"
        ],
        "怒": [
            "又倒闭一家！现在的品牌怎么都这么不靠谱啊！质量不行、设计又烂，就知道圈钱，简直太气人了！以后买东西真得好好掂量掂量，再也不能随便花钱了！"
        ],
        "哀": [
            "为什么？为什么它要离开我？它是我最爱的家人啊！我还没来得及多陪陪它，还想和它一起度过好多好多的日子呢，可它怎么就这么走了呀？"
        ],
        "惊": [
            "我的天呐，真不知道你怎么想的,居然做出这种事!",
            "打开快递才发现朋友竟把整套限量漫画寄给我，我以为断货了两年的第 0 卷此生无缘，简直像做梦。"
        ],
        "惧": [
            "这地儿阴森得慌，感觉空气里都透着邪乎劲儿。咱赶紧走吧，再待下去，我非得被吓死不可。",
            "电梯突然停在十三楼，灯闪了两下就暗了，我能听见自己心跳，按下紧急按钮的那一秒，手心全是汗。"
        ],
        "厌": [
            "这食物都馊了，还散发着奇怪的气味，看着就倒胃口，真不知道怎么能下咽。",
            "他朋友圈一水儿的豪车和名表，“随手晒一晒”四个字看得我指尖发麻，炫耀成瘾真让人反感。"
        ],
        "英文": [
            "The sun finally broke through the clouds this morning. I opened the window and let the fresh air fill the room. ",
            "I just brewed a cup of coffee and sat down with my favorite book. The quiet moments before work always feel like a small vacation. ",
            "My phone buzzed with a message from an old friend I haven't seen in years. We're meeting for lunch tomorrow, and I can't stop smiling at the thought."
        ]
    }
    
    # Prompt音频列表
    prompt_ids = [
        "ZH_B00000_S00020_W000009",
        "ZH_B00000_S09330_W000035", 
        "ZH_B00000_S04110_W000002",
        "ZH_B00000_S03150_W000001"
    ]
    
    # 情感类别与中文对应
    emotion_categories = {
        "喜": "快乐",
        "怒": "生气", 
        "哀": "伤心",
        "惊": "惊喜",
        "惧": "恐惧",
        "厌": "厌烦",
        "英文": ["快乐", "生气", "伤心", "惊喜", "恐惧", "中性", "厌烦"]  # 英文需要合成所有情感
        # "英文": ["中性"]
    }
    
    # 存储结果的字典
    results_json = {}
    utt_counter = 1
    
    print("Starting batch inference...")
    
    for prompt_id in prompt_ids:
        print(f"\nProcessing prompt: {prompt_id}")
        
        # 加载prompt音频和文本
        try:
            prompt_audio_path = os.path.join(prompt_dir, f"{prompt_id}.mp3")
            if not os.path.exists(prompt_audio_path):
                print(f"Warning: Prompt audio not found: {prompt_audio_path}")
                continue
                
            prompt_speech_16k = load_wav(prompt_audio_path, 16000)
            prompt_text = load_prompt_info(prompt_id, prompt_dir)
            print(f"Prompt text: {prompt_text}")
            
        except Exception as e:
            print(f"Error loading prompt {prompt_id}: {e}")
            continue
        
        # 遍历每种文本类别
        for text_category, text_list in texts.items():
            print(f"  Processing category: {text_category}")
            
            # 确定要合成的情感类型
            if text_category == "英文":
                emotions_to_synthesize = emotion_categories[text_category]
            else:
                emotions_to_synthesize = [emotion_categories[text_category]]
            
            # 遍历该类别下的每个文本
            for text_idx, synthesis_text in enumerate(text_list):
                
                # 遍历要合成的情感
                for emo_type in emotions_to_synthesize:
                    print(f"    Synthesizing emotion: {emo_type}")
                    
                    # try:
                    # 获取情感embedding
                    emo_key = emo_mapping[emo_type]
                    if emo_key in emotion_info_dict["esd-0002"]:
                        emotion_info = torch.tensor(emotion_info_dict["esd-0002"][emo_key], dtype=torch.float32)
                    elif emo_key.lower() == "fearful":
                        emotion_info = torch.tensor(emotion_info_dict["cse-male001"][emo_key], dtype=torch.float32)
                    elif emo_key.lower() == "neutral":
                        emotion_info = torch.tensor(neutral_info_dict["esd-0020"], dtype=torch.float32)
                    elif emo_key.lower() == "disgust":
                        text_hash = int(hashlib.md5(synthesis_text.encode()).hexdigest()[:8], 16)
                        torch.manual_seed(text_hash % 10000)
                        
                        angry_emotion = torch.tensor(emotion_info_dict["esd-0002"]["Angry"], dtype=torch.float32)
                        # 添加小幅随机扰动来模拟disgust
                        noise = torch.randn_like(angry_emotion) * 0.05  # 可调整噪声强度
                        emotion_info = angry_emotion + noise
                    else:
                        print(f"Warning: Emotion {emo_key} not found in embedding dict")
                        raise Exception
                    
                    # 进行推理
                    for i, result in enumerate(model.synthesize(
                        synthesis_text,
                        prompt_text,
                        prompt_speech_16k,
                        emo_type,
                        emotion_info,
                        False if text_category != "英文" else True
                    )):
                        # 生成输出文件名
                        filename = f"uttr_{utt_counter:03d}_{prompt_id}_{text_category}_{text_idx+1:02d}_{emo_type}.wav"
                        output_path = os.path.join(output_dir, filename)
                        
                        # 保存音频
                        torchaudio.save(output_path, result['tts_speech'], 22050)
                        
                        # 确定语言ID
                        language_id = 'en' if text_category == "英文" else 'ch'
                        
                        # 添加到结果JSON
                        utt_id = f"uttr_{utt_counter:03d}"
                        results_json[utt_id] = {
                            "text": synthesis_text,
                            "audio_path": output_path,
                            "style_label": emo_key.lower(),
                            "language_id": language_id,
                            "prompt_id": prompt_id,
                            "text_category": text_category,
                            "text_index": text_idx + 1
                        }
                        
                        print(f"      Saved: {filename}")
                        utt_counter += 1
                        break  # 只取第一个结果


                            
                    # except Exception as e:
                    #     print(f"Error synthesizing {emo_type} for text {text_idx+1}: {e}")
                    #     continue
    
    # 保存结果JSON
    json_output_path = os.path.join(output_dir, "metadata.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    print(f"\nBatch inference completed!")
    print(f"Generated {utt_counter-1} audio files")
    print(f"Results saved in: {output_dir}")
    print(f"Metadata saved as: {json_output_path}")

if __name__ == "__main__":
    batch_inference()