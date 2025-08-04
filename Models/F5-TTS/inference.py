# 

import argparse
import os
import time
from importlib.resources import files

import torch
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

def main(config_path, ref_audio_directory, output_directory):
    # cnofig
    config = OmegaConf.load(config_path)下载并加载模型和分词器
    # model_path = "code/F5-TTS/ckpts/basemodel/model_1200000.pt" 
    model_path = "model_pat"  
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    if "F5TTS" in config.model.name:
        model_cls = DiT
    elif "E2TTS" in config.model.name:
        model_cls = UNetT
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")
    model_cls = DiT
    print("Loading model and tokenizer from local path...")
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    # F5TTS_model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model(DiT, F5TTS_model_cfg, model_path, device="cuda")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded from local path.")

    # vocoder
    vocoder = load_vocoder(config)

    happy_text = "阳光透过窗帘的缝隙洒进房间，带来了新一天的活力与希望。清晨的空气中弥漫着芬芳的花香，让人忍不住深吸一口气，感受生命的美好。今天是一个令人期待的日子。朋友们早早就约好一起去郊外野餐，享受春日的美好。在宽阔的草地上，大家围坐在一起，欢声笑语此起彼伏。食物的香气在空气中飘扬，令人垂涎欲滴。孩子们在一旁追逐嬉戏，笑声如同悦耳的音乐般在空中回荡。有人放起了风筝，看着它在蓝天白云间自由翱翔，心中不禁生出无尽的欢愉。大人们则享受着这难得的闲暇时光，聊着生活中的趣事，分享彼此的喜悦。随着时间的流逝，夕阳渐渐染红了天空，给大地披上了一层金色的外衣。我们在笑声中拍下了一张张合影，记录下这份难以言喻的快乐与满足。这个幸福的日子，让人感受到友情的温暖和生活的美好。无论未来怎样，这些快乐的瞬间都将成为心中宝贵的财富，永远闪烁着温馨的光芒。"
    surprise_text = "天呐！我简直不敢相信自己的眼睛！刚刚在街上，我看到了我最喜欢的电影明星就在我面前走过。他竟然比电视上还要帅气迷人！这种近距离接触偶像的感觉太不可思议了，我的心脏到现在还在狂跳不已！哇塞！我刚刚收到了一封邮件，竟然是我梦寐以求的公司发来的录用通知！我反复确认了好几遍，生怕是在做梦。这可是全球顶尖的科技公司啊，我从未想过自己真的能得到这个机会。这简直是对我多年努力最好的回报！我的天！我刚才在后院发现了一个不明飞行物！它悬浮在空中，发出奇怪的光芒。我揉了揉眼睛，以为自己出现了幻觉，但它确实就在那里！这太不可思议了，我现在满脑子都是疑问，难道真的有外星生命来访问地球了吗？我的老天爷！我刚刚在阁楼里发现了一本古老的日记，翻开一看，竟然是我曾曾祖父的！更让人惊讶的是，里面详细记载了一个惊人的家族秘密。我完全被震惊了，这个秘密可能会改变我对家族历史的整个认知。我的上帝啊！我刚才在散步时，亲眼目睹了一只猫从五层楼高的地方跳下来，但它竟然毫发无伤！我惊得下巴都要掉下来了。我简直不敢相信自己的眼睛，这完全颠覆了我对物理定律的认知，感觉像是亲眼见证了一个奇迹！"
    angry_text = "我简直气炸了！今天在公司开会时，那个自以为是的同事又一次抢走了我的创意。我花了整整一周时间精心准备的方案，他却若无其事地当成自己的想法提出来。更可气的是，老板还称赞了他的创新思维。我忍不住想大声质问他的良心是不是被狗吃了！今天真是让我火冒三丈！一大早，我就被闹钟的尖锐铃声吵醒，比平时晚了整整半个小时。匆匆忙忙赶着出门，结果发现公交车居然提前开走了，我只好无奈地等着下一班。真是倒霉透顶！这些车主简直太过分了！我刚刚目睹一辆豪车强行加塞，差点造成连环追尾。他们以为有钱就可以为所欲为吗？我恨不得立刻冲上去对他们大吼大叫，让他们明白规则面前人人平等。这种目中无人的态度真是让人火冒三丈！我受够了这个邻居家的噪音！每天深夜他们都在放震耳欲聋的音乐，完全不考虑其他人的感受。我现在只要一听到那刺耳的音乐声就怒火中烧。这种自私自利的行为真是让人忍无可忍！我气得浑身发抖！刚才在超市排队结账时，一个人竟然毫不客气地插队到我前面。这种毫无公德心的人怎么好意思活在这个社会上？我真想给他一个响亮的耳光，让他明白什么叫基本礼仪。这种无视规则的行为真是令人发指！"
    sad_text = "我记得那是一个灰蒙蒙的下午，天空笼罩着厚重的云层，仿佛预示着即将到来的悲伤。我坐在窗边，看着雨水缓缓地流下，像是泪水无声地诉说着心中的烦恼。房间里静得出奇，只有墙角的钟表在无情地滴答作响。它提醒我，时间在不停地流逝，而我却无法改变什么。曾经的欢声笑语如今已化作遥不可及的回忆，留在心底的只有那份沉重的孤独。桌上那张泛黄的照片，记录了我们曾经的美好时光。你灿烂的笑容仿佛能驱散所有阴霾，而现在，我只能独自面对这无尽的寂寥。我们的故事似乎就此划上了句号，留下的只有未完的心愿和未说出口的再见。随着夜幕的降临，黑暗慢慢吞噬着整个房间，犹如孤独悄然逼近。我闭上双眼，试图找到一丝安慰，但心中的悲伤依旧如潮水般涌来，无法停止。这个夜晚，我知道，我必须学会接受，学会面对那份失去，尽管它如此刺痛人心。生活总是在不经意间教会我们成长，而成长，往往伴随着无法逃避的痛楚。"

    emotions = {
        "Happy": happy_text,
        "Surprise": surprise_text,
        "Angry": angry_text,
        "Sad": sad_text
    }
    emotions_map = {
        "Happy": "快乐",
        "Surprise": "惊喜",
        "Angry": "生气",
        "Sad": "伤心"
    }

    text_prompt = {
        "zhaijianing": "这个节目就是把四个男嘉宾，四个女嘉宾放一个大别墅里让他们朝夕相处一整个月，月末选择心动的彼此。",
        "fanzhiyi": "没这个能力知道吗，我已经说了，你像这样的比赛本身就没有打好基础。",
        "hulan": "发完之后那个工作人员说，老师，呼兰老师你还要再加个标签儿，我说加什么标签儿，他说你就加一个呼兰太好笑了。",
        "jiangzhihao": "就是很多我们这帮演员一整年也就上这么一个脱口秀类型的节目。",
        "lixueqin": "我就劝他，我说你呀，你没事儿也放松放松，你那身体都亮红灯儿了你还不注意。",
        "liuchang": "比如这三年我在街上开车，会在开车的时候进行一些哲思，我有天开车的时候路过一个地方。",
        "tangxiangyu": "大家好我叫唐香玉， 我年前把我的工作辞了，成了一个全职脱口秀演员。",
        "xiaolu": "然后我就老家的亲戚太多了，我也记不清谁该叫谁，所以我妈带着我和我。",
        "yuxiangyu": "我大学专业学的是哲学，然后节目组就说那这期主题你可以刚好聊一下哲学专业毕业之后的就业方向。",
        "zhaoxiaohui": "终于没有人问我为什么不辞职了，结果谈到现在，谈恋爱第一天人家问我，能打个电话吗？我说你有啥事儿。",
        "xuzhisheng": "最舒服的一个方式，这个舞台也不一定就是说是来第一年就好嘛，只要你坚持，肯定会有发光发热的那天嘛。"
    }

    emotion_dic = torch.load("emotion_data")
    for root, dirs, files in os.walk(ref_audio_directory):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):  
                ref_audio_path = os.path.join(root, file)

                person = ref_audio_path.split("/")[-1].split(".wav")[0]
                # for person, ref_text in text_prompt.items():
                # if person in file:
                for emotion, gen_text in emotions.items():
                    ref_text = emotions_map.get(emotion) + "<endofprompt>" + text_prompt.get(ref_audio_path.split("/")[-1].split(".wav")[0])
                    ref_audio, _ = preprocess_ref_audio_text(ref_audio_path, ref_text, config)
                    emotion_embedding = emotion_dic["0003"][emotion]

                    start_time = time.time()

                    audio_out, final_sample_rate, spectrogram = infer_process(
                        ref_audio,
                        ref_text,
                        emotion_embedding,
                        gen_text,
                        model,
                        vocoder,
                        mel_spec_type=config.model.mel_spec.mel_spec_type,
                        target_rms=0.1,
                        cross_fade_duration=0.15,  # config.cross_fade_duration,
                        nfe_step=128,  # config.nfe_step,
                        cfg_strength=2.0,  # config.cfg_strength,
                        sway_sampling_coef=-1.0,  # config.sway_sampling_coef,
                        speed=0.8,  # config.speed,
                        fix_duration=None,  # config.fix_duration,
                    )

                    end_time = time.time()
                    infer_time = end_time - start_time

                    audio_duration = len(audio_out) / final_sample_rate

                    # 
                    rtf = infer_time / audio_duration
                    print(f"参考音频: {ref_audio_path}, 人物: {person}, 情绪: {emotion}")
                    print(f"Inference time: {infer_time:.2f} seconds")
                    print(f"Audio duration: {audio_duration:.2f} seconds")
                    print(f"Real-time factor (RTF): {rtf:.2f}")

                    output_audio_name = os.path.splitext(file)[0] + f"_{person}_{emotion}.wav"
                    output_audio_path = os.path.join(output_directory, output_audio_name)
                    sf.write(output_audio_path, audio_out, final_sample_rate)
                    print(f"Generated audio saved to {output_audio_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local inference script for TTS model."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    parser.add_argument(
        "-r",
        "--ref_audio_directory",
        type=str,
        required=True,
        help="Path to the directory containing reference audio files."
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="Path to the directory to save the generated audio files."
    )

    args = parser.parse_args()

    main(args.config, args.ref_audio_directory, args.output_directory)