# 

import torch
import torch.nn.functional as F

def tpr_loss(disc_real_outputs, disc_generated_outputs, tau):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_DG = torch.median((dr - dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def mel_loss(real_speech, generated_speech, mel_transforms):
    loss = 0
    for transform in mel_transforms:
        mel_r = transform(real_speech)
        mel_g = transform(generated_speech)
        loss += F.l1_loss(mel_g, mel_r)
    return loss

def OrthogonalityLoss(speaker_embedding, emotion_embedding):
        speaker_embedding_t = speaker_embedding.t()
        dot_product_matrix = torch.matmul(emotion_embedding, speaker_embedding_t)
        emotion_norms = torch.norm(emotion_embedding, dim=1, keepdim=True)
        speaker_norms = torch.norm(speaker_embedding, dim=1, keepdim=True).t()
        normalized_dot_product_matrix = dot_product_matrix / (emotion_norms * speaker_norms)
        ort_loss = torch.norm(normalized_dot_product_matrix, p='fro')**2
        cosine_sim = F.cosine_similarity(emotion_embedding.unsqueeze(2), speaker_embedding.unsqueeze(1), dim=-1)
        cosine_ort_loss = torch.norm(cosine_sim.mean(dim=-1), p='fro') ** 2

        return  0.01 * (ort_loss + cosine_ort_loss)