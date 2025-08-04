# 

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torchaudio

# from transformers.modeling_utils import _load_state_dict_into_meta_model as load_state_dict_into_model

try:
    from transformers import AutoModelForSeamlessM4T 
except ImportError:
    from modeling_seamless_m4t_v2 import SeamlessM4Tv2ForTextToSpeechWithEmotion as AutoModelForSeamlessM4T

from transformers import SeamlessM4Tv2Config
from modeling_seamless_m4t_v2 import SeamlessM4Tv2TextToUnitForConditionalGenerationWithEmotionEmbedding
from processing_seamless_m4t_t2u_w2v import SeamlessM4TProcessor

import json
import deepspeed

# env
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from deepspeed.ops.adam import DeepSpeedCPUAdam

# Monkey-patch the __del__ method to avoid AttributeError
def safe_del(self):
    if hasattr(self, 'ds_opt_adam'):
        self.ds_opt_adam.destroy_adam(self.opt_id)

DeepSpeedCPUAdam.__del__ = safe_del

class Config:
    # data
    manifest_path = "data dir"
    emotion_embedding_path = "emotion data dir"

    # model
    pretrained_model = "facebook/hf-seamless-m4t-large"

    # parameters
    batch_size = 1 
    gradient_accumulation_steps = 8  
    num_epochs = 10
    learning_rate = 3e-5
    max_audio_length = 3
    max_text_length = 50
    pad_token_id = 1
    bos_token_id = 0

    deepspeed_config = "ds_config.json"
    num_workers = 1
    mixed_precision = "fp16" 

    save_steps = 1  
    save_epochs = 1    
    keep_checkpoints = 3 

    output_dir = "output_dir"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

ds_config = {
    "train_batch_size": 512,  # # 192,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "weight_decay": 0.01
        }
    },
    "bfloat16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 20,
    "wall_clock_breakdown": False
}

class SeamlessM4Tv2WithEmotionConfig(SeamlessM4Tv2Config):
    def __init__(self, emotion_embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.emotion_embed_dim = emotion_embed_dim

def init_model(config):
    """init"""
    model_config = SeamlessM4Tv2WithEmotionConfig.from_pretrained("facebook/seamless-m4t-v2-large")
    model_config.SeamlessM4Tv2TextToUnitForConditionalGeneration = SeamlessM4Tv2TextToUnitForConditionalGenerationWithEmotionEmbedding
    # model = AutoModelForSeamlessM4T._from_config(model_config)
    model = AutoModelForSeamlessM4T.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        config=model_config)

    pretrained_model = AutoModelForSeamlessM4T.from_pretrained("facebook/seamless-m4t-v2-large")
    pretrained_state_dict = pretrained_model.state_dict()
    model.load_state_dict(pretrained_state_dict, strict=False)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainable_params = []
    for name, param in model.named_parameters():
        if "t2u_model" in name:  
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False

    print(f"Trainable parameters: {trainable_params}")  
    return model

def shift_tokens_right(input_ids, pad_token_id, bos_token_id):
    """
    Args:
        input_ids: (batch_size, seq_len) 
        pad_token_id: 
        bos_token_id:
    Returns:
        shifted_input_ids: (batch_size, seq_len)
    """
    shifted_input_ids = torch.full_like(input_ids, fill_value=pad_token_id)

    shifted_input_ids[1:] = input_ids[:-1].clone()

    shifted_input_ids[0] = bos_token_id

    shifted_input_ids.masked_fill_(
        (shifted_input_ids == bos_token_id) & (input_ids == pad_token_id),
        pad_token_id
    )

    return shifted_input_ids

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.transforms = torch.nn.Sequential(
            # torchaudio.transforms.Vol(gain=0.2),  
            # torchaudio.transforms.TimeStretch(rate_range=(0.9, 1.1)),
            # torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            # torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

    def __call__(self, waveform):
        return self.transforms(waveform)

class SeamlessM4TDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.processor = SeamlessM4TProcessor.from_pretrained(config.pretrained_model, return_units=True, use_fast=False)
        self.audio_augment = AudioAugmentation()
        self.samples = self._load_manifest()

    def _load_manifest(self):
        samples = []
        # import pdb;pdb.set_trace()
        utt_emotion_embedding = torch.load(self.config.emotion_embedding_path)
        with open(self.config.manifest_path) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4 and os.path.exists(parts[1]):
                    duration = torchaudio.info(parts[1]).num_frames / 16000
                    if 1.0 <= duration <= self.config.max_audio_length:
                        samples.append({
                            "emotion_embedding": utt_emotion_embedding[parts[0]], 
                            "audio": parts[1],
                            "text": parts[2],
                            "src_lang": parts[3],
                            "tgt_lang": parts[4]
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        waveform, sr = torchaudio.load(sample["audio"])
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            audios = resampler(audios)

        waveform = self.audio_augment(waveform)  

        audio_inputs = self.processor(
            audios=waveform,
            sampling_rate=sr,
            return_tensors="pt",
            max_length=int(sr * self.config.max_audio_length),
            truncation=True,
            padding="max_length"
        )
        text = self._text_augmentation(sample["text"])
        text_inputs = self.processor(
            text=text,
            src_lang=sample["src_lang"],
            tgt_lang=sample["tgt_lang"],
            return_tensors="pt",
            max_length=self.config.max_text_length,
            truncation=True,
            padding="max_length"
        )

        return {
            "input_ids": text_inputs["input_ids"][0].to(torch.int32), #ori_text torch.Size([50])  
            "char_input_ids": text_inputs["char_input_ids"][0].to(torch.int32), # torch.Size([50, 7])
            "char_count_per_id": text_inputs["char_count_per_id"][0].to(torch.int32),  # torch.Size([50])
            "decoder_input_ids": shift_tokens_right(text_inputs["input_ids"][0].to(torch.int32), self.config.pad_token_id, self.config.bos_token_id),
            "attention_mask": audio_inputs["attention_mask"][0],  # # torch.Size([24000])
            "labels": torch.tensor(text_inputs["input_ids"][0]).to(torch.int32), #  target_text  torch.Size([50]) 
            "emotion_embedding": sample["emotion_embedding"].to(torch.bfloat16), 
            "decoder_attention_mask": audio_inputs["attention_mask"][0], # torch.Size([24000])
            "t2u_labels": shift_tokens_right(torch.tensor(audio_inputs["units"]).to(torch.int32), self.config.pad_token_id, self.config.bos_token_id) # torch.Size([130])
        }

    def _text_augmentation(self, text, p=0.3):
        return text

def collate_fn(batch):
    audio_inputs = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    emotion_embedding = torch.stack([item["emotion_embedding"] for item in batch])
    decoder_input_ids = torch.stack([item["decoder_input_ids"] for item in batch])
    decoder_attention_mask = torch.stack([item["decoder_attention_mask"] for item in batch])
    t2u_labels = torch.stack([item["t2u_labels"] for item in batch])
    char_input_ids = torch.stack([item["char_input_ids"] for item in batch])
    char_count_per_id = torch.stack([item["char_count_per_id"] for item in batch])
    return {
        "input_ids": labels,
        "attention_mask": attention_masks,
        "labels": labels,
        "emotion_embedding": emotion_embedding,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "t2u_labels": t2u_labels,
        "char_input_ids": char_input_ids, 
        "char_count_per_id": char_count_per_id
    }

def setup_dataloaders(config, rank, world_size):
    full_dataset = SeamlessM4TDataset(config)
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, val_loader

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def save_model(engine, config, epoch=None):
    if dist.get_rank() == 0:  
        save_dir = os.path.join(config.output_dir, f"epoch_{epoch}") if epoch else os.path.join(config.output_dir, "final")

        engine.save_checkpoint(save_dir)

        model_to_save = engine.module if hasattr(engine, 'module') else engine
        model_to_save.save_pretrained(save_dir)

        print(f"Model saved at {save_dir}")

def train(config):
    # ddp 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    deepspeed.init_distributed(dist_backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # model init
    model = init_model(config)
    model.config.use_cache = False  # 强制梯度检查点
    model.gradient_checkpointing_enable()

    # trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found! Check your model freezing logic.")

    # init DeepSpeed engine
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        # model_parameters=trainable_params,  # 
        config=config.deepspeed_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        lr_scheduler=get_linear_schedule_with_warmup(
            AdamW(trainable_params, lr=config.learning_rate),
            num_warmup_steps=100,
            num_training_steps=1000
        )
    )

    # data load 
    train_loader, _ = setup_dataloaders(config, rank, world_size)

    # training loop
    for epoch in range(config.num_epochs):
        print("epoch......", epoch)
        engine.train()
        train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=rank!=0):
            batch = {k: v.to(engine.device) for k, v in batch.items()}
            loss = engine(**batch) #.loss
            engine.backward(loss)
            engine.step()
            torch.cuda.empty_cache()
            print("batch size:")

        #     if rank == 0 and engine.global_steps % 10 == 0:
        #         mem_stats = get_accelerator().memory_stats(engine.device)
        #         print(f"Step {engine.global_steps} | Loss: {loss.item():.4f} | "
        #               f"GPU Mem: {mem_stats['allocated']/1024**3:.2f}GB")
        # print("saving model")
        save_model(engine, config, epoch+1)

    save_model(engine, config)
if __name__ == "__main__":
    config = Config()

    # save deepspeed congif
    with open(config.deepspeed_config, "w") as f:
        json.dump(ds_config, f)

    # train
    train(config)