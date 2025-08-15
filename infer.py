from Models.marco_voice.cosyvoice_rodis.cli.cosyvoice import CosyVoice
from Models.marco_voice.cosyvoice_emosphere.cli.cosyvoice import CosyVoice as cosy_emosphere

from Models.marco_voice.cosyvoice_rodis.utils.file_utils import load_wav

# Load pre-trained model
model = CosyVoice('trained_model/v4', load_jit=False, load_onnx=False, fp16=False)
emo = {"伤心": "Sad", "恐惧":"fearful", "快乐": "Happy", "惊喜": "Surprise", "生气": "Angry", "戏谑":"jolliest"} 
prompt_speech_16k = load_wav("your_audio_path/exam.wav", 16000)
emo_type="开心"
if emo_type in ["伤心", "恐惧"]:
    emotion_info = torch.load("./assets/emotion_info.pt")["zhu"][emo.get(emo_type)]
else:
    emotion_info = torch.load("./assets/emotion_info.pt")["song"][emo.get(emo_type)]
# Voice cloning with discrete emotion
for i, j in enumerate(model.synthesize(
    text="今天的天气真不错，我们出去散步吧！",
    prompt_text="",
    reference_speech=prompt_speech_16k,
    emo_type=emo_type,
    emotion_embedding=emotion_info
)):
  torchaudio.save('emotional_{}.wav'.format(emo_type), j['tts_speech'], 22050)

# Continuous emotion control
model_emosphere = cosy_emosphere('trained_model/v5', load_jit=False, load_onnx=False, fp16=False)

for i, j in enumerate(model_emosphere.synthesize(
    text="今天的天气真不错，我们出去散步吧！",
    prompt_text="",
    reference_speech=prompt_speech_16k,
    emotion_embedding=emotion_info,
    low_level_emo_embedding=[0.1, 0.4, 0.5]
)):
  torchaudio.save('emosphere_{}.wav'.format(emo_type), j['tts_speech'], 22050)



### More Features
# Cross-lingual emotion transfer
for i, j in enumerate(model.synthesize(
    text="hello, i'm a speech synthesis model, how are you today? ",
    prompt_text="",
    reference_speech=prompt_speech_16k,
    emo_type=emo_type,
    emotion_embedding=emotion_info
)):
  torchaudio.save('emosphere_ross_lingual_{}.wav'.format(emo_type), j['tts_speech'], 22050)