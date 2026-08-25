"""
model_conductance.py
Layer-conductance attribution of the decoder's first upsampling block.

EXPLORATORY analysis only (per the paper's Supplementary Material). Attributes
decoder output regions back to the latent code via Captum LayerConductance.
Requires: pip install captum
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import LayerConductance
from model import DisentangledGenerator

DEVICE = 'cuda:0'
CODE = 'disentangle_v1'
model = DisentangledGenerator().to(DEVICE)
model.load_state_dict(torch.load(f'./checkpoints/model_{CODE}.tm'))
model.eval()

# cond_generator = ...  # user provides one batch (xt, ...)
xt = next(iter(cond_generator))[0].to(DEVICE)
feature = model.feature(xt)
z, skips = model.latent(feature)
d3, d4, d5, d6, d7, d8, d9, d10 = skips

dim = (2, 128, 512)                       # output (channels, freq, time)
accum = torch.zeros_like(z)
conductance = LayerConductance(model.decode, model.decode_main.up0.model[0])
for c in range(dim[0]):
    for f in range(0, dim[1], 16):
        for t in range(0, dim[2], 32):
            attr = conductance.attribute(
                (z, d3, d4, d5, d6, d7, d8, d9, d10),
                target=(c, f, t), attribute_to_layer_input=True)
            accum += attr.mean(dim=0)
accum = (accum / (dim[0] * dim[1] * dim[2])).flatten(start_dim=1).detach().cpu().numpy()

plt.figure(figsize=(12, 3))
plt.imshow(accum); plt.colorbar(orientation='horizontal', pad=0.1)
plt.savefig('./conductance.png')
