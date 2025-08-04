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
from torch.utils.data import Dataset
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

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
from torch.utils.data import Dataset
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import json
import deepspeed
import numpy as np

try:
    from transformers import AutoModelForSeamlessM4T 
except ImportError:
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForTextToSpeechWithEmotion as AutoModelForSeamlessM4T

from transformers import SeamlessM4Tv2Config
from modeling_seamless_m4t_v2 import SeamlessM4Tv2TextToUnitForConditionalGenerationWithEmotionEmbedding
from transformers.models.seamless_m4t.processing_seamless_m4t_t2u_w2v import SeamlessM4TProcessor

# Environment setup
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
    # Data configuration
    manifest_path = "data_path"
    emotion_embedding_path = "emotion data path"

    # Model configuration
    pretrained_model = "facebook/hf-seamless-m4t-large"

    # Training parameters
    batch_size = 64
    gradient_accumulation_steps = 8
    num_epochs = 10
    learning_rate = 3e-5
    max_audio_length = 3
    max_text_length = 50
    pad_token_id = 1
    bos_token_id = 0

    # DeepSpeed configuration
    deepspeed_config = "ds_config.json"
    num_workers = 0
    mixed_precision = "fp16" 

    # Checkpoint saving
    save_steps = 1  
    save_epochs = 1    
    keep_checkpoints = 3 

    # Output directories
    output_dir = "output_dir"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Logging configuration
    log_interval = 10  # Log every 10 batches
    eval_interval = 1  # Evaluate every epoch

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 64,
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
    """Initialize the model with emotion embedding support"""
    model_config = SeamlessM4Tv2WithEmotionConfig.from_pretrained("facebook/seamless-m4t-v2-large")
    model_config.SeamlessM4Tv2TextToUnitForConditionalGeneration = SeamlessM4Tv2TextToUnitForConditionalGenerationWithEmotionEmbedding

    model = AutoModelForSeamlessM4T.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        config=model_config)

    # Load pretrained weights
    pretrained_model = AutoModelForSeamlessM4T.from_pretrained("facebook/seamless-m4t-v2-large")
    pretrained_state_dict = pretrained_model.state_dict()
    model.load_state_dict(pretrained_state_dict, strict=False)

    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Only train the T2U model parameters
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
    """Shift input tokens to the right for decoder input"""
    shifted_input_ids = torch.full_like(input_ids, fill_value=pad_token_id)
    shifted_input_ids[1:] = input_ids[:-1].clone()
    shifted_input_ids[0] = bos_token_id
    shifted_input_ids.masked_fill_(
        (shifted_input_ids == bos_token_id) & (input_ids == pad_token_id),
        pad_token_id
    )
    return shifted_input_ids

class AudioAugmentation:
    """Audio data augmentation pipeline"""
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
    """Dataset class for SeamlessM4T training"""
    def __init__(self, config):
        self.config = config
        self.processor = SeamlessM4TProcessor.from_pretrained(config.pretrained_model, return_units=True, use_fast=False)
        self.audio_augment = AudioAugmentation()
        self.samples = self._load_manifest()

    def _load_manifest(self):
        """Load dataset manifest and filter by duration"""
        samples = []
        utt_emotion_embedding = torch.load(self.config.emotion_embedding_path)

        with open(self.config.manifest_path) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4 and os.path.exists(parts[1]):
                    duration = torchaudio.info(parts[1]).num_frames / 16000
                    if 0.0 <= duration <= self.config.max_audio_length:
                        samples.append({
                            "emotion_embedding": utt_emotion_embedding[parts[0]], 
                            "audio": parts[1],
                            "text": parts[2],
                            "src_lang": parts[3],
                            "tgt_lang": parts[4]
                        })
        return samples

    def __len__(self):
        print("len(self.samples):", len(self.samples))
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single training sample with consistent tensor sizes"""
        sample = self.samples[idx]
        # Load and preprocess audio
        waveform, sr = torchaudio.load(sample["audio"])
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Audio augmentation
        waveform = self.audio_augment(waveform)

        # Calculate target lengths based on config
        target_audio_length = int(sr * self.config.max_audio_length)
        target_text_length = self.config.max_text_length

        # Process audio inputs with fixed length
        audio_inputs = self.processor(
            audios=waveform,
            sampling_rate=target_sr,
            return_tensors="pt",
            max_length=target_audio_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True
        )
        # Process text inputs with fixed length
        text = self._text_augmentation(sample["text"])
        text_inputs = self.processor(
            text=text,
            src_lang=sample["src_lang"],
            tgt_lang=sample["tgt_lang"],
            return_tensors="pt",
            max_length=target_text_length,
            truncation=True,
            padding="max_length",
            return_char_info=True
        )

        # Get units and ensure consistent length
        units = audio_inputs.get("units", torch.zeros(1))  # Fallback if units not available
        if isinstance(units, list):
            units = torch.tensor(units[0]) if units else torch.zeros(1)

        # Pad/Cut units to fixed length
        unit_target_length = 130  # Set your target unit length here
        if units.dim() == 1:
            units = units.unsqueeze(0)  # Add batch dim if missing

        if units.size(1) > unit_target_length:
            units = units[:, :unit_target_length]
        elif units.size(1) < unit_target_length:
            pad_size = unit_target_length - units.size(1)
            units = torch.nn.functional.pad(units, (0, pad_size), value=self.config.pad_token_id)

        # Ensure attention masks have consistent length
        audio_attention_mask = audio_inputs["attention_mask"][0]
        if audio_attention_mask.size(0) > target_audio_length:
            audio_attention_mask = audio_attention_mask[:target_audio_length]
        elif audio_attention_mask.size(0) < target_audio_length:
            pad_size = target_audio_length - audio_attention_mask.size(0)
            audio_attention_mask = torch.nn.functional.pad(audio_attention_mask, (0, pad_size))

        return {
            # Text inputs (fixed length)
            "input_ids": text_inputs["input_ids"][0].to(torch.int32),
            "char_input_ids": text_inputs["char_input_ids"][0].to(torch.int32),
            "char_count_per_id": text_inputs["char_count_per_id"][0].to(torch.int32),
            "decoder_input_ids": shift_tokens_right(
                text_inputs["input_ids"][0].to(torch.int32), 
                self.config.pad_token_id, 
                self.config.bos_token_id
            ),
            "labels": text_inputs["input_ids"][0].to(torch.int32),

            # Audio inputs (fixed length)
            "attention_mask": audio_attention_mask,
            "decoder_attention_mask": audio_attention_mask.clone(),

            # Units (fixed length)
            "t2u_labels": shift_tokens_right(
                units.to(torch.int32),
                self.config.pad_token_id,
                self.config.bos_token_id
            ).squeeze(0),  # Remove batch dim for consistency

            # Emotion embedding
            "emotion_embedding": sample["emotion_embedding"].to(torch.bfloat16)
        }

    def _text_augmentation(self, text, p=0.3):
        """Simple text augmentation (placeholder)"""
        return text

def collate_fn(batch):
    """Batch collation with safety checks"""
    # Convert list of dicts to dict of lists
    batch = {key: [d[key] for d in batch] for key in batch[0].keys()}

    # Stack tensors with consistent shapes
    collated = {}
    for key, values in batch.items():
        try:
            # Special handling for emotion embedding which might have different dims
            if key == "emotion_embedding":
                collated[key] = torch.stack(values)
            else:
                # For all other tensors, ensure they have same shape before stacking
                max_len = max(v.size(0) if v.dim() > 0 else 1 for v in values)
                padded_values = []
                for v in values:
                    if v.dim() == 1:
                        if v.size(0) < max_len:
                            v = torch.nn.functional.pad(v, (0, max_len - v.size(0)), 
                                value=0 if "mask" in key else Config.pad_token_id)
                    padded_values.append(v)
                collated[key] = torch.stack(padded_values)
        except Exception as e:
            print(f"Error collating key {key}: {str(e)}")
            raise

    return collated

def setup_dataloaders(config, rank, world_size):
    """Initialize train and validation dataloaders"""
    full_dataset = SeamlessM4TDataset(config)
    # import pdb;pdb.set_trace()

    # Split dataset
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    # Distributed samplers
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

    # DataLoaders
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
    """Evaluation function"""
    model.eval()
    total_loss = 0
    total_batches = 1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0

def save_model(engine, config, epoch=None):
    """Save model checkpoint"""
    if dist.get_rank() == 0:  
        save_dir = os.path.join(config.output_dir, f"epoch_{epoch}") if epoch else os.path.join(config.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)

        # Save DeepSpeed checkpoint
        engine.save_checkpoint(save_dir)

        # Save HuggingFace model format
        model_to_save = engine.module if hasattr(engine, 'module') else engine
        model_to_save.save_pretrained(save_dir)

        print(f"Model saved at {save_dir}")

def train(config):
    """Main training function"""
    # Initialize distributed training
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    deepspeed.init_distributed(dist_backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Initialize TensorBoard (only on rank 0)
    if rank == 0:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, current_time))
        print(f"TensorBoard logs will be saved to: {os.path.join(config.log_dir, current_time)}")

    # Initialize model
    model = init_model(config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Verify trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found! Check your model freezing logic.")

    # Initialize DeepSpeed engine
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=config.deepspeed_config,
        model_parameters=trainable_params,
        lr_scheduler=get_linear_schedule_with_warmup(
            AdamW(trainable_params, lr=config.learning_rate),
            num_warmup_steps=100,
            num_training_steps=1000
        )
    )

    # Setup data loaders
    train_loader, val_loader = setup_dataloaders(config, rank, world_size)

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        engine.train()
        train_loader.sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        total_loss = 0
        total_batches = 1

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=rank!=0)

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(engine.device) for k, v in batch.items()}

            # Forward pass
            outputs = engine(**batch)
            loss = outputs # .loss

            # Backward pass and optimizer step
            engine.backward(loss)
            engine.step()

            # Update tracking metrics
            total_loss += loss.item()
            total_batches += 1

            # Log training metrics
            if global_step % config.log_interval == 0 and rank == 0:
                tb_writer.add_scalar('train/loss', loss.item(), global_step)
                tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

                # Log memory usage
                mem_stats = torch.cuda.memory_stats(engine.device)
                tb_writer.add_scalar('mem/allocated', mem_stats['allocated_bytes.all.current']/1024**3, global_step)
                tb_writer.add_scalar('mem/reserved', mem_stats['reserved_bytes.all.current']/1024**3, global_step)

            global_step += 1
            torch.cuda.empty_cache()

            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # Calculate epoch metrics
        epoch_loss = total_loss / total_batches
        epoch_time = time.time() - epoch_start_time

        # Validation
        if epoch % config.eval_interval == 0:
            val_loss = evaluate(engine, val_loader, engine.device)

            # Log validation metrics
            if rank == 0:
                tb_writer.add_scalar('val/loss', val_loss, epoch)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(engine, config, "best")

        # Log epoch metrics
        if rank == 0:
            tb_writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            tb_writer.add_scalar('meta/epoch_time', epoch_time, epoch)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss: {epoch_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}" if epoch % config.eval_interval == 0 else "  (Skipped validation)")
            print(f"  Time: {epoch_time:.2f} seconds")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if (epoch + 1) % config.save_epochs == 0:
            save_model(engine, config, epoch+1)

    # Final save and cleanup
    save_model(engine, config)
    if rank == 0:
        tb_writer.close()
        print("Training completed!")

if __name__ == "__main__":
    # Initialize configuration
    config = Config()

    # Save DeepSpeed configuration
    with open(config.deepspeed_config, "w") as f:
        json.dump(ds_config, f)
        print(f"DeepSpeed config saved to {config.deepspeed_config}")

    # Start training
    print("Starting training...")
    train(config)