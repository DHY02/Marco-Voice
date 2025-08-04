# 

import argparse
import os
import time
from importlib.resources import files

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

def main(config_path, ref_audio_path, ref_text, gen_text, output_audio_path):
    # 加载配置文件
    config = OmegaConf.load(config_path)

    # 下载并加载模型和分词器
    # model_path = "code/F5-TTS/ckpts/basemodel/model_1200000.pt"  # 替换为本地模型路径
    model_path = "code/F5-TTS/ckpts/F5TTS_Small_vocos_pinyin_common_voice/model_last.pt"  # 替换为本地模型路径
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    if "F5TTS" in config.model.name:
        model_cls = DiT
    elif "E2TTS" in config.model.name:
        model_cls = UNetT
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")
    model_cls = DiT
    print("Loading model and tokenizer from local path...")
    # F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    F5TTS_model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model(DiT, F5TTS_model_cfg, model_path, device="cuda")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded from local path.")

    # 加载声码器
    vocoder = load_vocoder(config)

    # 预处理参考音频和文本
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text, config)
    # print(f"Input data dtype: {ref_audio.dtype}")

    # warmup_iterations = 10
    # for _ in range(warmup_iterations):
    #     _ = infer_process(
    #         ref_audio,
    #         ref_text,
    #         gen_text,
    #         model,
    #         vocoder,
    #         mel_spec_type=config.model.mel_spec.mel_spec_type,
    #         target_rms=0.1,
    #         cross_fade_duration=0.15,
    #         nfe_step=32,
    #         cfg_strength=2.0,
    #         sway_sampling_coef=-1.0,
    #         speed=1,
    #         fix_duration=None,
    #     )
    # print(f"Warmup completed with {warmup_iterations} iterations.")

    # 记录推理开始时间
    start_time = time.time()

    # 调用推理函数
    audio_out, final_sample_rate, spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        model,
        vocoder,
        mel_spec_type=config.model.mel_spec.mel_spec_type,
        target_rms=0.1,
        cross_fade_duration=0.15, # config.cross_fade_duration,
        nfe_step=128, # config.nfe_step,
        cfg_strength=2.0, # config.cfg_strength,
        sway_sampling_coef=-1.0, # config.sway_sampling_coef,
        speed=1, # config.speed,
        fix_duration=None, # config.fix_duration,
    )

    # 记录推理结束时间
    end_time = time.time()
    infer_time = end_time - start_time

    # 计算音频时长
    audio_duration = len(audio_out) / final_sample_rate

    # 计算并打印实时因子（RTF）
    rtf = infer_time / audio_duration
    print(f"Inference time: {infer_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Real-time factor (RTF): {rtf:.2f}")

    # 保存生成的音频
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
        "--ref_audio",
        type=str,
        required=True,
        help="Path to the reference audio file."
    )
    parser.add_argument(
        "-t",
        "--ref_text",
        type=str,
        required=True,
        help="Reference text."
    )
    parser.add_argument(
        "-g",
        "--gen_text",
        type=str,
        required=True,
        help="Text to generate audio for."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save the generated audio."
    )

    args = parser.parse_args()

    main(args.config, args.ref_audio, args.ref_text, args.gen_text, args.output)
