"""
model_evaluation.py
Recurrent multi-step generation and CorrX evaluation.

Loads a trained model and scores multi-step continuations against ground
truth. CorrX is the peak cross-correlation ratio defined in the paper
(best-lag cross-correlation normalised by the reference autocorrelation;
no absolute value or clipping).
"""
import numpy as np
import torch
from model import DisentangledGenerator, transform

DEVICE = 'cuda:0'
CODE = 'disentangle_v1'
model = DisentangledGenerator().to(DEVICE)
model.load_state_dict(torch.load(f'./checkpoints/model_{CODE}.tm'))
model.eval()

def corrx(ref, gen):
    """Peak cross-correlation ratio (best lag), normalised by reference
    autocorrelation. No abs, no clipping — matches the paper definition."""
    ref = ref - ref.mean(); gen = gen - gen.mean()
    xc = np.correlate(gen, ref, mode='full')
    return xc.max() / (np.dot(ref, ref) + 1e-12)

@torch.no_grad()
def generate_steps(xt, n_steps=3):
    """Recurrent inference: seed xt, produce X_{t+1..t+n} by looping outputs back."""
    outputs = []
    cur = xt
    for _ in range(n_steps):
        (rec_t, _), z_p, z_t, *_ , feature, z, _, skips, _ = model(cur, cur)
        z_t_shift, delta, skips_lat = model.skip_latent(feature, z_p, z_t)
        nxt = model.reconstruct(
            model.decode_main(torch.cat([z_p + delta, z_t_shift], dim=-1), skips_lat))
        outputs.append(nxt)
        cur = nxt          # loop back
    return outputs

# eval_generator = ...  # user provides: yields (xt, targets[list of X_{t+1..t+n}], subject)
results = {}   # subject -> list of per-step CorrX
for xt, targets, subj in eval_generator:
    preds = generate_steps(xt.to(DEVICE), n_steps=len(targets))
    for step, (p, tgt) in enumerate(zip(preds, targets)):
        p = transform(p)[0]; tgt = tgt.cpu().numpy()[0]
        results.setdefault(int(subj), []).append((step, corrx(tgt, p)))

# aggregate per subject / per step -> Table 2
