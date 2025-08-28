import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
marco_voice_path = os.path.join(current_dir, 'Models', 'marco_voice')
sys.path.insert(0, marco_voice_path)
matcha_tts_path = os.path.join(current_dir, 'Models', 'marco_voice', 'third_party', 'Matcha-TTS')
if matcha_tts_path not in sys.path:
    sys.path.insert(0, matcha_tts_path)
from Models.marco_voice.cosyvoice_rodis.cli.cosyvoice import CosyVoice
from Models.marco_voice.cosyvoice_emosphere.cli.cosyvoice import CosyVoice as cosy_emosphere

from Models.marco_voice.cosyvoice_rodis.utils.file_utils import load_wav
import torch, torchaudio

# Load pre-trained model
model = CosyVoice('trained_models/CosyVoice-300M-emo-2', load_jit=False, load_onnx=False, fp16=False)
emo = {"伤心": "Sad", "恐惧":"Fearful", "快乐": "Happy", "惊喜": "Surprise", "生气": "Angry", "戏谑":"jolliest"} 
prompt_speech_16k = load_wav("/root/gpufree-data/ESD/Emotion_Speech_Dataset/0001/Angry/0001_000351.wav", 16000)
emo_type="恐惧"
if emo_type in ["恐惧"]:
    emotion_info = torch.load("processed/train/embedding_info.pt")["cse-female002"][emo.get(emo_type)]
else:
    emotion_info = torch.load("processed/train/embedding_info.pt")["cse-female002"][emo.get(emo_type)]
emotion_info = torch.tensor(emotion_info, dtype=torch.float32) 
# Voice cloning with discrete emotion
for i, j in enumerate(model.synthesize(
    "天呐，路边那家小店居然还记得我喜欢的拿铁比例，第一口就回想起留学时的春天，幸福感爆棚。",
    "打远一看，它们的确很是美丽",
    prompt_speech_16k,
    emo_type,
    emotion_info
)):
  torchaudio.save('emotional_{}.wav'.format(emo_type), j['tts_speech'], 22050)

# # Continuous emotion control
# model_emosphere = cosy_emosphere('trained_model/v5', load_jit=False, load_onnx=False, fp16=False)

# for i, j in enumerate(model_emosphere.synthesize(
#     text="今天的天气真不错，我们出去散步吧！",
#     prompt_text="",
#     reference_speech=prompt_speech_16k,
#     emotion_embedding=emotion_info,
#     low_level_emo_embedding=[0.1, 0.4, 0.5]
# )):
#   torchaudio.save('emosphere_{}.wav'.format(emo_type), j['tts_speech'], 22050)



# ### More Features
# # Cross-lingual emotion transfer
# for i, j in enumerate(model.synthesize(
#     text="hello, i'm a speech synthesis model, how are you today? ",
#     prompt_text="",
#     reference_speech=prompt_speech_16k,
#     emo_type=emo_type,
#     emotion_embedding=emotion_info
# )):
#   torchaudio.save('emosphere_ross_lingual_{}.wav'.format(emo_type), j['tts_speech'], 22050)