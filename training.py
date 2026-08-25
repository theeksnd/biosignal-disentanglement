"""
model_training.py
Disentangled autoencoder for long-term biosignal generation — training.

Reference implementation consolidated from the original research code.
No pretrained checkpoints are distributed; this is provided so the training
procedure and loss structure are reproducible. Users adapt paths/config.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix

from model import (DisentangledGenerator, SkipEncode, LatentDiscriminator,
                   ReconDiscriminator, SubjectClassifier, freeze, unfreeze)

DEVICE = 'cuda:0'
CODE = 'disentangle_v1'
CKPT_DIR = './checkpoints'
EPOCHS = 1000
BATCH = 64
LR = 1e-4
os.makedirs(CKPT_DIR, exist_ok=True)

adversarial_loss = nn.BCELoss()

def cos_loss(x1, x2):
    return (1 - F.cosine_similarity(x1, x2, dim=-1)).mean()

# --- models ---
model = DisentangledGenerator().to(DEVICE)
discrim = LatentDiscriminator().to(DEVICE)          # Dis_l  (latent)
discrimre = ReconDiscriminator().to(DEVICE)         # Dis_r  (reconstruction)
subjectclf = SubjectClassifier().to(DEVICE)         # Cls (frozen, pre-trained)
subjectclf.load_state_dict(torch.load(f'{CKPT_DIR}/model_classifier.tm'))
freeze(subjectclf)

# --- optimisers (separate per module, per the paper) ---
enc_optim     = torch.optim.Adam(model.feature.parameters(), lr=LR, betas=(0.9, 0.999))
dec_optim     = torch.optim.Adam(list(model.decode_main.parameters()) +
                                 list(model.reconstruct.parameters()), lr=LR, betas=(0.9, 0.999))
shift_optim   = torch.optim.Adam(model.skip_latent.parameters(), lr=LR, betas=(0.9, 0.999))
discrim_optim = torch.optim.AdamW(discrim.parameters(), lr=LR)
reconsd_optim = torch.optim.Adam(discrimre.parameters(), lr=LR)

# training_generator = ...  # user provides: yields (xt, xt_p1, s) tensors

for epoch in range(EPOCHS):
    model.train()
    for i, (xt, xt_p1, s) in enumerate(training_generator):
        xt, xt_p1, s = xt.to(DEVICE), xt_p1.to(DEVICE), s.to(DEVICE)
        bs = xt.shape[0]
        valid = torch.ones((bs, 1), device=DEVICE)
        fake  = torch.zeros((bs, 1), device=DEVICE)

        enc_optim.zero_grad(); dec_optim.zero_grad()
        discrim_optim.zero_grad(); reconsd_optim.zero_grad()

        # ---- Stage 1: reconstruction + subject classification ----
        (rec_t, rec_t_), z_p, z_t, z_p_, z_t_, feature, z, p_pred, skips, skips_ = model(xt, xt_p1)
        clf_loss = F.cross_entropy(p_pred, s)
        recons_loss = F.mse_loss(rec_t, xt) + F.mse_loss(rec_t_, xt_p1)
        (recons_loss + clf_loss).backward(retain_graph=True)

        # ---- Stage 2: cyclic conditional generation (decoder frozen) ----
        freeze(model.decode_main); freeze(model.reconstruct)
        crec_tp1 = model.reconstruct(model.decode_main(torch.cat([z_p, z_t_], dim=-1), skips))
        crec_t   = model.reconstruct(model.decode_main(torch.cat([z_p_, z_t], dim=-1), skips_))
        recons_loss_cond = (F.mse_loss(crec_tp1, xt_p1) + F.mse_loss(crec_t, xt)
                            + 0.5 * (adversarial_loss(discrimre(crec_tp1, xt_p1), valid)
                                     + 0.5 * adversarial_loss(discrimre(crec_t, xt), valid)))
        recons_loss_cond.backward(retain_graph=True)

        # ---- Stage 3: shift-encoder training (feature encoder frozen) ----
        shift_optim.zero_grad()
        freeze(model.feature)
        z_t_shift, delta, skips_lat = model.skip_latent(feature.detach(), z_p, z_t)
        shift_loss_latent = cos_loss(z_t_shift, z_t_.detach())
        x_recons_shift = model.reconstruct(
            model.decode_main(torch.cat([z_p.detach() + delta, z_t_shift], dim=-1), skips_lat))
        shift_rec_loss = F.mse_loss(x_recons_shift, xt_p1)
        adver_loss     = adversarial_loss(discrim(torch.cat([z_t_shift, z_t], dim=-1)), valid)
        adver_loss_rec = adversarial_loss(discrimre(x_recons_shift, xt_p1), valid)
        shift_loss = 0.5 * (shift_loss_latent + adver_loss) + shift_rec_loss + adver_loss_rec
        shift_loss.backward(retain_graph=True)
        shift_optim.step()

        unfreeze(model.feature)
        enc_optim.step(); dec_optim.step()
        unfreeze(model.decode_main); unfreeze(model.reconstruct)

        # ---- discriminator updates ----
        discrim_loss = 0.5 * (adversarial_loss(discrim(torch.cat([z_t_shift, z_t], dim=-1).detach()), fake)
                              + adversarial_loss(discrim(torch.cat([z_t_, z_t], dim=-1).detach()), valid))
        discrim_loss.backward(); discrim_optim.step()

        rediscrim_loss = 0.5 * (adversarial_loss(discrimre(xt_p1, xt_p1), valid)
                                + adversarial_loss(discrimre(x_recons_shift.detach(), xt_p1), fake))
        rediscrim_loss.backward(); reconsd_optim.step()

    torch.save(model.state_dict(),     f'{CKPT_DIR}/model_{CODE}.tm')
    torch.save(discrim.state_dict(),   f'{CKPT_DIR}/discrim_{CODE}.tm')
    torch.save(discrimre.state_dict(), f'{CKPT_DIR}/discrimre_{CODE}.tm')
