"""
model.py — Disentangled Autoencoder for Long-Term Biosignal Generation
Continuous Long-Term ECG and PPG Generation Through Latent Disentanglement
Consolidated from the original training implementation (skipmodv5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 108              # encoder channel width at the bottleneck
SPLIT_PER_CODE = 216    # <-- 216 per code (code default) OR 108 (paper Table 1)
N_SUBJECTS = 17

# ============================ U-Net blocks ===============================
class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, normalize=True, dropout=0.0, prop=False):
        super().__init__()
        if prop:
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        else:
            layers = [nn.Conv2d(in_ch, out_ch, (3 if stride == 1 else 4, 4),
                                (stride, 2), (1, 1), bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, dropout=0.0, normalize=True, prop=False):
        super().__init__()
        if prop:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1)]
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, (3 if stride == 1 else 4),
                                         (stride, 2), 1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_ch)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip, skip_drop=False):
        x = self.model(x)
        if skip_drop:
            x_skip = F.dropout(x_skip, p=0.7)
        return torch.cat([x, x_skip], 1)


# ============================ Feature encoder (E_f) ======================
class Encode(nn.Module):
    """Feature encoder E_f: maps a feature map to a latent code + skip tensors."""
    def __init__(self):
        super().__init__()
        self.down3  = UnetDown(24, 32)
        self.down4  = UnetDown(32, 64, dropout=0.5, prop=True)
        self.down5  = UnetDown(64, 64, dropout=0.5, prop=True)
        self.down6  = UnetDown(64, 72, dropout=0.5)
        self.down7  = UnetDown(72, 84, dropout=0.5, prop=True)
        self.down8  = UnetDown(84, 84, dropout=0.5, prop=True)
        self.down9  = UnetDown(84, 96, dropout=0.5)
        self.down10 = UnetDown(96, 96, dropout=0.5)
        self.down11 = UnetDown(96, 108, dropout=0.5, normalize=False)
        self.latent = nn.Conv2d(108, 108, kernel_size=1, padding=0)

    def forward(self, x):
        d3 = self.down3(x); d4 = self.down4(d3); d5 = self.down5(d4)
        d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        d9 = self.down9(d8); d10 = self.down10(d9)
        z = self.down11(d10)
        z = self.latent(z).flatten(start_dim=1)
        z = F.normalize(z, dim=0)
        return z, (d3, d4, d5, d6, d7, d8, d9, d10)


# ============================ Skip / shift encoder (E_s, E_sh) ===========
class SkipEncode(nn.Module):
    """
    Skip-connection encoder combined with the time-shift estimation
    (E_s / E_sh): predicts the next temporal code z_t{t+1} and the shift delta.
    """
    def __init__(self):
        super().__init__()
        self.down3  = UnetDown(24, 32)
        self.down4  = UnetDown(32, 64, dropout=0.5, prop=True)
        self.down5  = UnetDown(64, 64, dropout=0.5, prop=True)
        self.down6  = UnetDown(64, 72, dropout=0.5)
        self.down7  = UnetDown(72, 84, dropout=0.5, prop=True)
        self.down8  = UnetDown(84, 84, dropout=0.5, prop=True)
        self.down9  = UnetDown(84, 96, dropout=0.5)
        self.down10 = UnetDown(96, 96, dropout=0.5)
        self.down11 = UnetDown(96, 108, dropout=0.5, normalize=False)
        self.latent = nn.Conv2d(108, 108, kernel_size=1, padding=0)

        self.mlp_fuse = nn.Sequential(
            nn.Linear(8 * 108, 108 * 4), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(108 * 4, 108 * 4), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(108 * 4, 108 * 3), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(108 * 3, 108 * 2),
        )
        self.delta_gen = nn.Sequential(
            nn.Linear(108 * 6, 108 * 3), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(108 * 3, 108 * 3), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(108 * 3, 108 * 2),
        )

    def forward(self, x, z_p, z_t):
        d3 = self.down3(x); d4 = self.down4(d3); d5 = self.down5(d4)
        d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        d9 = self.down9(d8); d10 = self.down10(d9)
        z = self.down11(d10)
        z = self.latent(z).flatten(start_dim=1)
        z_t_next = F.normalize(self.mlp_fuse(torch.cat([z, z_p, z_t], dim=-1)), dim=0)
        z_delta  = F.normalize(self.delta_gen(torch.cat([z, z_p], dim=-1)), dim=0)
        return z_t_next, z_delta, (d3, d4, d5, d6, d7, d8, d9, d10)


# ============================ Decoder (D) ===============================
class SkipDecode(nn.Module):
    def __init__(self):
        super().__init__()
        self.up0 = UnetUp(108, 96, dropout=0.5, normalize=False)
        self.up1 = UnetUp(96 * 2, 96, dropout=0.5)
        self.up2 = UnetUp(96 * 2, 84, dropout=0.5)
        self.up3 = UnetUp(84 * 2, 84, dropout=0.5, prop=True)
        self.up4 = UnetUp(84 * 2, 72, dropout=0.5, prop=True)
        self.up5 = UnetUp(72 * 2, 64, dropout=0.5)
        self.up6 = UnetUp(64 * 2, 64, dropout=0.5, prop=True)
        self.up7 = UnetUp(64 * 2, 32, dropout=0.5, prop=True)

    def forward(self, z, skip, skip_drop=False):
        d3, d4, d5, d6, d7, d8, d9, d10 = skip
        u0 = self.up0(z.view(-1, 108, 1, 4), d10, skip_drop)
        u1 = self.up1(u0, d9, skip_drop)
        u2 = self.up2(u1, d8, skip_drop)
        u3 = self.up3(u2, d7, skip_drop)
        u4 = self.up4(u3, d6, skip_drop)
        u5 = self.up5(u4, d5, skip_drop)
        u6 = self.up6(u5, d4, skip_drop)
        u7 = self.up7(u6, d3, skip_drop)
        return u7


# ============================ Full generator ===========================
class DisentangledGenerator(nn.Module):
    """
    Full model: feature extractor -> encoder (E_f) -> [z_p, z_t] split
    -> decoder (D) reconstruction, plus subject prediction head.
    The shift encoder (SkipEncode) and discriminators are instantiated
    separately during training (see train.py).
    """
    def __init__(self, split_per_code=SPLIT_PER_CODE, n_subjects=N_SUBJECTS):
        super().__init__()
        self.split_per_code = split_per_code
        self.feature = nn.Sequential(
            nn.Conv2d(2, 12, (3, 3), padding=1), nn.InstanceNorm2d(12),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Conv2d(12, 16, (4, 4), stride=2, padding=1), nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Conv2d(16, 24, (4, 4), stride=2, padding=1), nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Conv2d(24, 24, (3, 3), padding=1), nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
        )
        self.latent = Encode()
        self.decode_main = SkipDecode()
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1), nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.ConvTranspose2d(32, 24, (4, 4), stride=2, padding=1), nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.ConvTranspose2d(24, 16, 3, stride=1, padding=1), nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1), nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.ConvTranspose2d(8, 2, 3, padding=1),
        )
        self.subject = nn.Linear(split_per_code, n_subjects)

    def forward(self, x, xt):
        feat = self.feature(x)
        z, skips = self.latent(feat)
        z_p, z_t = torch.split_with_sizes(
            z, [self.split_per_code, self.split_per_code], dim=-1)

        feat_ = self.feature(xt)
        z_, skips_ = self.latent(feat_)
        z_p_, z_t_ = torch.split_with_sizes(
            z_, [self.split_per_code, self.split_per_code], dim=-1)

        rec_t  = self.reconstruct(self.decode_main(z,  skips, skip_drop=True))
        rec_t_ = self.reconstruct(self.decode_main(z_, skips, skip_drop=True))
        p_pred = F.softmax(self.subject(z_p), dim=-1)
        return (rec_t, rec_t_), z_p, z_t, z_p_, z_t_, feat, z, p_pred, skips, skips_


# ============================ Discriminators & classifier ===============
class LatentDiscriminator(nn.Module):
    """Latent-space discriminator (Dis_l), operates on concatenated codes."""
    def __init__(self, in_dim=108 * 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.4), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.mlp(x)


class ReconDiscriminator(nn.Module):
    """Reconstruction-space discriminator (Dis_r), on real/generated spectrograms."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 24, (3, 7), padding=(1, 3), stride=(4, 4)),
            nn.LeakyReLU(0.4, inplace=True), nn.Dropout(0.4),
            nn.Conv2d(24, 36, (3, 7), padding=(1, 3), stride=(4, 4)),
            nn.LeakyReLU(0.4, inplace=True), nn.Dropout(0.4),
            nn.Conv2d(36, 64, (3, 7), padding=(1, 3), stride=(4, 8)),
            nn.LeakyReLU(0.4, inplace=True), nn.Dropout(0.4),
            nn.Flatten(start_dim=1), nn.Linear(512, 1), nn.Sigmoid(),
        )
    def forward(self, x, xq):
        return self.model(torch.cat([x, xq], dim=1)).squeeze(dim=-1)


class SubjectClassifier(nn.Module):
    """Pre-trained subject classifier (Cls), frozen during main training."""
    def __init__(self, n_subjects=N_SUBJECTS):
        super().__init__()
        self.feature = nn.Sequential(
            self._c(2, 4, stride=1, normalize=False),
            self._c(4, 8, down=False), self._c(8, 12, down=False),
            self._c(12, 16, down=False), self._c(16, 24), self._c(24, 32),
            self._c(32, 48, down=False),
        )
        self.embedding = nn.Sequential(
            self._c(48, 64, dropout=0.5), self._c(64, 72, dropout=0.5),
            self._c(72, 84, dropout=0.5), self._c(84, 84, dropout=0.5),
            nn.Conv2d(84, 84, 4, 2, 1),
        )
        self.subject = nn.Linear(168, n_subjects)

    def _c(self, i, o, stride=2, normalize=True, dropout=0.0, down=True):
        if down:
            layers = [nn.Conv2d(i, o, (3 if stride == 1 else 4, 4), (stride, 2), (1, 1), bias=False)]
        else:
            layers = [nn.Conv2d(i, o, 3, padding=1)]
        if normalize: layers.append(nn.InstanceNorm2d(o))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout: layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        f = self.feature(x)
        z = self.embedding(f).flatten(start_dim=1)
        return torch.softmax(self.subject(z), dim=-1)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
