# Continuous Long-Term ECG and PPG Generation Through Latent Disentanglement

## Overview

This repository contains the implementation of a novel generative framework for continuous long-term biosignal generation through latent disentanglement. The proposed framework combines autoencoder-based generation with adversarial learning to synthesise long-duration, subject-specific ECG and PPG signals via recurrent inference.

To the best of our knowledge, this is the first study to generate subject-conditioned ECG sequences of approximately 8 s — substantially beyond the single R-peak or short-segment outputs of existing generation approaches.

---

## Key Contributions

- **Disentangled Autoencoder Framework** — explicitly separates subject-specific morphology (*z_s*) from temporal dynamics (*z_d*) in the learned latent space
- **Time-Shift Encoder** (*E_sh*) — predicts the next temporal latent code, enabling recurrent signal generation
- **Recurrent Inference** — recursively synthesises long-duration sequences from a single seed segment
- **Style-Transfer Analysis** — demonstrates the transferability of learned subject codes across subjects
- **Multi-modal Validation** — demonstrated on ECG (EPHNOGRAM dataset) and PPG (PhysioNet arterial blood pressure dataset)

---
## Base Architecture

The encoder/decoder backbone employed in this framework is built upon and extended from our previously published implementation:

> **biosignalGANs** — Adversarial learning models for biological signals including artificial synthesis and modality transfer.
> [https://github.com/theekshanadis/biosignalGANs](https://github.com/theekshanadis/biosignalGANs)

The novel components introduced in this work — the shift encoder, conditional latent discriminator, and three-stage training procedure — are extensions of that base architecture and will be released in this repository upon paper acceptance.

---
## Method

The framework operates on 2D spectrogram representations (magnitude + phase, shape `[2 × 128 × 256]`) and learns to disentangle two dominant signal characteristics:

| Latent Code | Symbol | Captures |
|---|---|---|
| Subject-specific morphological code | *z_s* | Individual morphological style, consistent across time for a given subject |
| Temporal dynamics code | *z_d* | Rhythm and rate-related variations |

### Three-Stage Training

| Stage | Epochs | Active Modules | Loss Functions |
|---|---|---|---|
| Stage 1 | 100 | Encoder + Decoder | *L_re* + *L_cce* |
| Stage 2 | 150 | Encoder only (Decoder frozen) | *L_re-c* (cyclic reconstruction) |
| Stage 3 | 150 | Shift Encoder only (Encoder frozen) | *L_adv* + *L_mse* + *L_re-sh* |

### Recurrent Inference
```
Seed Xt → Ef → ft → Es → [zs, zd]
                               ↓
                         Esh → zd_{t+1}
                               ↓
                         D([zs, zd_{t+1}]) → X̂_{t+1}
                               ↓
                         Loop back → feed X̂_{t+1} as new Xt
                               ↓
                         Repeat → X̂_{t+2}, X̂_{t+3} ...
```

---

## Datasets
| Modality | Dataset | Subjects | Segment Length |
|---|---|---|---|
| ECG | [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/) | 17 | 1.2 s |
| PPG | [PhysioNet Arterial Blood Pressure](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) | 7 | 1.2 s |
---

## Citation
If you use this work please cite:
```bibtex
@article{dissanayake2026disentanglement,
  title   = {Continuous Long-Term ECG and PPG Generation Through Latent Disentanglement},
  author  = {Dissanayake, Theekshana and Fernando, Tharindu and Denman, Simon and Sridharan, Sridha and Fookes, Clinton},
  journal = {Under-Review: Computers in Biology and Medicine},
  year    = {2026}
}
```

---

## Related Work
- **biosignalGANs** (base architecture): [https://github.com/theekshanadis/biosignalGANs](https://github.com/theekshanadis/biosignalGANs)
---

## Contact
For questions or collaborations, please open an issue or contact the corresponding author: Theekshana Dissanayake, TU-Berlin, BIFOLD
