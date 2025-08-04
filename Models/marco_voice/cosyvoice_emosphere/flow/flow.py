# 

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice_emosphere.utils.mask import make_pad_mask
import numpy as np

def OrthogonalityLoss(speaker_embedding, emotion_embedding):
        speaker_embedding_t = speaker_embedding.t()
        dot_product_matrix = torch.matmul(emotion_embedding, speaker_embedding_t)
        emotion_norms = torch.norm(emotion_embedding, dim=1, keepdim=True)
        speaker_norms = torch.norm(speaker_embedding, dim=1, keepdim=True).t()
        normalized_dot_product_matrix = dot_product_matrix / (emotion_norms * speaker_norms)
        ort_loss = torch.norm(normalized_dot_product_matrix, p='fro')**2

        cosine_sim = F.cosine_similarity(emotion_embedding.unsqueeze(2), speaker_embedding.unsqueeze(1), dim=-1)
        cosine_ort_loss = torch.norm(cosine_sim.mean(dim=-1), p='fro') ** 2

        return  0.02 * (ort_loss + cosine_ort_loss)

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 Lort_losss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.Lort_losss = Lort_losss
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

        # emotion embedding 
        self.emo_VAD_inten_proj = nn.Linear(1, 2 * spk_embed_dim, bias=True)
        self.emosty_layer_norm = nn.LayerNorm(2 * spk_embed_dim)

        self.sty_proj = nn.Linear(spk_embed_dim, spk_embed_dim, bias=True)

        self.azimuth_bins = nn.Parameter(torch.linspace(-np.pi/2, np.pi, 4), requires_grad=False)
        self.azimuth_emb = torch.nn.Embedding(4, spk_embed_dim // 2)
        self.elevation_bins = nn.Parameter(torch.linspace(np.pi/2, np.pi, 2), requires_grad=False)
        self.elevation_emb = torch.nn.Embedding(2, spk_embed_dim // 2)

        self.spk_embed_proj = nn.Linear(512, spk_embed_dim, bias=True)
        self.emo_proj = nn.Linear(768, spk_embed_dim, bias=True)

        # self.spk_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(512, 1024),
        #     Mish(),
        #     torch.nn.Linear(1024, spk_embed_dim),
        # )

        self.emo_mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            Mish(),
            torch.nn.Linear(1024, spk_embed_dim),
        )
        if self.Lort_losss:
            # print("self.Lort_losss:", self.Lort_losss)
            self.map_speaker_embedding = torch.nn.Linear(output_size, spk_embed_dim)

    def _process_embeddings(self, embedding, low_level_emo_embedding, emotion_embedding):
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        emos_proj_embed = self.emo_mlp(emotion_embedding)
        intens_embed = self.emo_VAD_inten_proj(low_level_emo_embedding[:, 0:1])
        # style_vector=style_vector.squeeze(1) 
        ele_embed = 0
        elevation = low_level_emo_embedding[:, 1:2]
        elevation_index = torch.bucketize(elevation, self.elevation_bins)
        elevation_index = elevation_index.squeeze(1)
        elevation_embed = self.elevation_emb(elevation_index)
        ele_embed = elevation_embed + ele_embed
        azi_embed = 0
        azimuth = low_level_emo_embedding[:, 2:3]   
        azimuth_index = torch.bucketize(azimuth, self.azimuth_bins)
        azimuth_index = azimuth_index.squeeze(1)
        azimuth_embed = self.azimuth_emb(azimuth_index)
        azi_embed = azimuth_embed + azi_embed # [11, 96]

        style_embed = torch.cat((ele_embed, azi_embed), dim=-1) # 192
        style_proj_embed = self.sty_proj(style_embed)  # 192

        # Softplus+
        combined_embedding = torch.cat((emos_proj_embed, style_proj_embed), dim=-1)  # 384
        emotion_embedding = F.softplus(combined_embedding)
        emosty_embed = self.emosty_layer_norm(emotion_embedding)
        emo_all_emb = (intens_embed + emosty_embed) # torch.Size([11, 384])
        embedding = torch.cat((embedding, emo_all_emb), dim=-1)  # torch.Size([11, 464])

        return embedding

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device) # torch.Size([11, 173, 80])
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['utt_embedding'].to(device) # torch.Size([11, 80])
        low_level_emo_embedding = batch['low_level_emotion_embedding'].to(device)  # low_level_emo_embedding
        emotion_embedding = batch['emotion_embedding'].to(device) # torch.Size([11, 768])

        # emo_id = batch['utt_emo'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(embedding)

        # processing emotion intense and speaker info
        emos_proj_embed = self.emo_mlp(emotion_embedding)
        if self.Lort_losss:
            spk_embedding_ort = self.map_speaker_embedding(spk_embedding)
            lort_losss = OrthogonalityLoss(spk_embedding_ort, emos_proj_embed)
        else:
            lort_losss = 0

        intens_embed = self.emo_VAD_inten_proj(low_level_emo_embedding[:, 0:1])
        # style_vector=style_vector.squeeze(1) 
        ele_embed = 0
        elevation = low_level_emo_embedding[:, 1:2]
        elevation_index = torch.bucketize(elevation, self.elevation_bins)
        elevation_index = elevation_index.squeeze(1)
        elevation_embed = self.elevation_emb(elevation_index)
        ele_embed = elevation_embed + ele_embed
        azi_embed = 0
        azimuth = low_level_emo_embedding[:, 2:3]   
        azimuth_index = torch.bucketize(azimuth, self.azimuth_bins)
        azimuth_index = azimuth_index.squeeze(1)
        azimuth_embed = self.azimuth_emb(azimuth_index)
        azi_embed = azimuth_embed + azi_embed # [11, 96]

        style_embed = torch.cat((ele_embed, azi_embed), dim=-1) # 192
        style_proj_embed = self.sty_proj(style_embed)  # 192

        # Softplus+
        combined_embedding = torch.cat((emos_proj_embed, style_proj_embed), dim=-1)  # 384
        emotion_embedding = F.softplus(combined_embedding)
        emosty_embed = self.emosty_layer_norm(emotion_embedding)
        emo_all_emb = (intens_embed + emosty_embed) # torch.Size([11, 384])
        embedding = torch.cat((spk_embedding, emo_all_emb), dim=-1)  # torch.Size([11, 464])

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device) # torch.Size([11, 173, 80])
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),  # torch.Size([11, 80, 173])
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(), # torch.Size([11, 173, 80])
            embedding,  # torch.Size([11, 464]) [11, 192]
            cond=conds
        ) 

        loss += lort_losss

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  low_level_emo_embedding,
                  emotion_embedding,
                  flow_cache):
        # check dim if batch （if batch_size == 1）
        # import pdb;pdb.set_trace()
        # if token.ndim == 2:
        #     token = token.unsqueeze(0)
        #     token_len = token_len.unsqueeze(0)
        # if prompt_token.ndim == 2:
        #     prompt_token = prompt_token.unsqueeze(0)
        #     prompt_token_len = prompt_token_len.unsqueeze(0)
        # if prompt_feat.ndim == 2:
        #     prompt_feat = prompt_feat.unsqueeze(0)
        #     prompt_feat_len = prompt_feat_len.unsqueeze(0)
        # if embedding.ndim == 2:
        #     embedding = embedding.unsqueeze(0)
        if low_level_emo_embedding.ndim == 1:
            low_level_emo_embedding = low_level_emo_embedding.unsqueeze(0)
        if emotion_embedding.ndim == 1:
            emotion_embedding = emotion_embedding.unsqueeze(0)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        emos_proj_embed = self.emo_mlp(emotion_embedding)
        print("low_level_emo_embedding:", low_level_emo_embedding)
        intens_embed = self.emo_VAD_inten_proj(low_level_emo_embedding[:, :1])

        ele_embed = 0
        elevation = low_level_emo_embedding[:, 1:2]
        elevation_index = torch.bucketize(elevation, self.elevation_bins)
        elevation_index = elevation_index 
        elevation_embed = self.elevation_emb(elevation_index)
        ele_embed = elevation_embed + ele_embed

        azi_embed = 0
        azimuth = low_level_emo_embedding[:, 2:3]
        azimuth_index = torch.bucketize(azimuth, self.azimuth_bins)
        azimuth_index = azimuth_index 
        azimuth_embed = self.azimuth_emb(azimuth_index)
        azi_embed = azimuth_embed + azi_embed # [batch_size, 96]

        style_embed = torch.cat((ele_embed, azi_embed), dim=-1) # [batch_size, 192]
        style_proj_embed = self.sty_proj(style_embed)
        if style_proj_embed.dim() == 3:
            style_proj_embed = style_proj_embed.squeeze(1)
        elif style_proj_embed.dim() == 2:
            pass

        # Softplus+
        combined_embedding = torch.cat((emos_proj_embed, style_proj_embed), dim=-1)  # [batch_size, 384]
        emotion_embedding = F.softplus(combined_embedding)
        emosty_embed = self.emosty_layer_norm(emotion_embedding)
        emo_all_emb = (intens_embed + emosty_embed) # [batch_size, 384]

        embedding = torch.cat((embedding, emo_all_emb), dim=-1)  # [batch_size, 464]

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([token.shape[0], mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2] * token.shape[0]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat, flow_cache