# 

import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

try:
    from transformers import AutoModelForSeamlessM4T
except ImportError:
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForTextToSpeechWithEmotion as AutoModelForSeamlessM4T

from transformers import SeamlessM4Tv2Config
from modeling_seamless_m4t_v2 import SeamlessM4Tv2TextToUnitForConditionalGenerationWithEmotionEmbedding
from transformers.models.seamless_m4t.processing_seamless_m4t_t2u import SeamlessM4TProcessor

import json

# env
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Config:
    # data
    manifest_path = "data_dir"
    emotion_embedding_path = "emotion data dir"

    # model
    pretrained_model = "facebook/hf-seamless-m4t-large"

    # parameters
    batch_size = 8
    gradient_accumulation_steps = 8
    num_epochs = 10
    learning_rate = 3e-5
    max_audio_length = 3
    max_text_length = 512
    pad_token_id = 1
    bos_token_id = 0

    num_workers = 1
    mixed_precision = "fp16"

    save_steps = 1
    save_epochs = 1
    keep_checkpoints = 3

    output_dir = "output_dir"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

class SeamlessM4Tv2WithEmotionConfig(SeamlessM4Tv2Config):
    def __init__(self, emotion_embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.emotion_embed_dim = emotion_embed_dim

def init_model(config):
    """init"""
    model_config = SeamlessM4Tv2WithEmotionConfig.from_pretrained("facebook/seamless-m4t-v2-large")
    model_config.SeamlessM4Tv2TextToUnitForConditionalGeneration = SeamlessM4Tv2TextToUnitForConditionalGenerationWithEmotionEmbedding
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

        # 
        input_ids = text_inputs["input_ids"][0].to(torch.bfloat16)
        char_input_ids = text_inputs["char_input_ids"][0].to(torch.bfloat16)
        char_count_per_id = text_inputs["char_count_per_id"][0].to(torch.bfloat16)
        decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.bos_token_id)
        labels = torch.tensor(input_ids).to(torch.bfloat16)
        emotion_embedding = sample["emotion_embedding"].to(torch.bfloat16)
        decoder_attention_mask = audio_inputs["attention_mask"][0]
        t2u_labels = shift_tokens_right(torch.tensor(audio_inputs["units"][0]).to(torch.bfloat16), self.config.pad_token_id, self.config.bos_token_id)

        max_lengths = {
            "input_ids": self.config.max_text_length,
            "char_input_ids": self.config.max_text_length,
            "char_count_per_id": self.config.max_text_length,
            "decoder_input_ids": self.config.max_text_length,
            "labels": self.config.max_text_length,
            "t2u_labels": self.config.max_text_length,  # 假设最大长度不超过文本最大长度
            "attention_mask": int(self.config.max_audio_length * sr),
            "decoder_attention_mask": int(self.config.max_audio_length * sr)
        }

        def pad_tensor(tensor, max_length, pad_value):
            return F.pad(tensor, (0, max_length - tensor.size(0)), "constant", pad_value)

        input_ids = pad_tensor(input_ids, max_lengths["input_ids"], self.config.pad_token_id)
        char_input_ids = pad_tensor(char_input_ids, max_lengths["char_input_ids"], self.config.pad_token_id)
        char_count_per_id = pad_tensor(char_count_per_id, max_lengths["char_count_per_id"], self.config.pad_token_id)
        decoder_input_ids = pad_tensor(decoder_input_ids, max_lengths["decoder_input_ids"], self.config.pad_token_id)
        labels = pad_tensor(labels, max_lengths["labels"], self.config.pad_token_id)
        t2u_labels = pad_tensor(t2u_labels, max_lengths["t2u_labels"], self.config.pad_token_id)
        attention_mask = pad_tensor(audio_inputs["attention_mask"][0], max_lengths["attention_mask"], 0)
        decoder_attention_mask = pad_tensor(decoder_attention_mask, max_lengths["decoder_attention_mask"], 0)

        return {
            "input_ids": input_ids,
            "char_input_ids": char_input_ids,
            "char_count_per_id": char_count_per_id,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "emotion_embedding": emotion_embedding,
            "decoder_attention_mask": decoder_attention_mask,
            "t2u_labels": t2u_labels
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

def save_model(model, config, epoch=None):
    if dist.get_rank() == 0:
        save_dir = os.path.join(config.output_dir, f"epoch_{epoch}") if epoch else os.path.join(config.output_dir, "final")
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(save_dir)
        print(f"Model saved at {save_dir}")

def train(config):
    # ddp
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # model init
    model = init_model(config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found! Check your model freezing logic.")

    # optimizer and scheduler
    optimizer = AdamW(trainable_params, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000
    )

    # data load
    train_loader, _ = setup_dataloaders(config, rank, world_size)

    # training loop
    for epoch in range(config.num_epochs):
        print("epoch......", epoch)
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=rank != 0):
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            print("batch size:")

            if rank == 0 and (epoch * len(train_loader) + train_loader.batch_sampler.sampler.offset) % 10 == 0:
                mem_stats = torch.cuda.memory_stats(rank)
                print(f"Step {epoch * len(train_loader) + train_loader.batch_sampler.sampler.offset} | Loss: {loss.item():.4f} | "
                      f"GPU Mem: {mem_stats['allocated_bytes.all.current'] / 1024 ** 3:.2f}GB")
        print("saving model")
        save_model(model, config, epoch + 1)

    save_model(model, config)

if __name__ == "__main__":
    config = Config()
    train(config)
