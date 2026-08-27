# Continuous Long-Term ECG and PPG Generation Through Latent Disentanglement

## Overview

This repository contains the implementation of a generative framework for continuous long-term biosignal generation through latent disentanglement. The proposed framework combines autoencoder-based generation with adversarial learning to synthesise long-duration, subject-specific ECG and PPG signals via recurrent inference.

The framework generates subject-conditioned ECG sequences of approximately 6 s (a reliable two-forward-step horizon), extending to approximately 8 s for a subset of subjects — beyond the single R-peak or short-segment outputs of prior generation approaches. We present these results as a preliminary within-recording proof of concept; see the manuscript for the full evaluation scope and limitations.

---

## Key Contributions

- **Disentangled Autoencoder Framework** — separates a subject-specific morphological code (*z_p*) from a temporal dynamics code (*z_t*) in the learned latent space
- **Time-Shift Encoder** (*E_sh*) — predicts the next temporal latent code, enabling recurrent signal generation
- **Recurrent Inference** — recursively synthesises multi-step sequences from a single seed segment
- **Style-Transfer Analysis** — explores the transferability of learned subject codes across subjects
- **Multi-modal Demonstration** — shown on ECG (EPHNOGRAM dataset) and PPG (PhysioNet arterial blood pressure dataset)

---
## Base Architecture

The encoder/decoder backbone employed in this framework is built upon and extended from our previously published implementation:

> **biosignalGANs** — Adversarial learning models for biological signals including artificial synthesis and modality transfer.
> [https://github.com/theekshanadis/biosignalGANs](https://github.com/theekshanadis/biosignalGANs)

The novel components introduced in this work — the shift encoder, conditional latent discriminator, and three-stage training procedure — are extensions of that base architecture and will be released in this repository in accordance with the journal's data and code policy.

---
## Method

The framework operates on 2D spectrogram representations (magnitude + phase, shape `[2 × 128 × 256]`) and learns to disentangle two dominant signal characteristics:

| Latent Code | Symbol | Captures |
|---|---|---|
| Subject-specific morphological code | *z_p* | Individual morphological style, consistent across time for a given subject |
| Temporal dynamics code | *z_t* | Rhythm and rate-related variations |

### Three-Stage Training

| Stage | Epochs | Active Modules | Loss Functions |
|---|---|---|---|
| Stage 1 | 100 | Encoder + Decoder | *L_re* + *L_cce* |
| Stage 2 | 150 | Encoder only (Decoder frozen) | *L_c-re* (cyclic reconstruction) |
| Stage 3 | 150 | Shift Encoder only (Encoder frozen) | *L_adv* + *L_mse* + *L_sh-re* |


---

## Datasets
| Modality | Dataset | Subjects | Segment Length |
|---|---|---|---|
| ECG | [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/) | 17 | 2.0 s |
| PPG | [PhysioNet Arterial Blood Pressure]([ https://physionet.org/content/... ](https://physionet.org/content/bp-graphene-bioimpedance/1.0.0/subject6_day1/setup01_baseline/#files-panel)) | 7 | 1.2 s |
---

## Citation
If you use this work please cite:
```bibtex
@article{dissanayake2026disentanglement,
  title   = {Continuous Long-Term ECG and PPG Generation Through Latent Disentanglement},
  author  = {Dissanayake, Theekshana and Fernando, Tharindu and Denman, Simon and Sridharan, Sridha and Fookes, Clinton},
  journal = {Under review, Computers in Biology and Medicine},
  year    = {2026}
}
```
## Implementation

This repository provides a reference implementation of the disentangled
autoencoder framework described in the paper. **Pretrained checkpoints are not
distributed**; the code is released for methodological transparency and to
allow adaptation to other datasets.

### Repository contents

| File | Description |
|---|---|
| `model.py` | Cleaned model definition — encoder (E_f), skip/shift encoder (E_s, E_sh), decoder (D), latent and reconstruction discriminators, and the subject classifier. |
| `training.py` | Cleaned three-stage training loop (reconstruction + classification → cyclic conditional generation → shift-encoder training). |
| `evaluation.py` | Cleaned recurrent multi-step generation and CorrX evaluation. |
| `conductance.py` | Cleaned layer-conductance attribution (exploratory analysis; see Supplementary Material). |
| `test_inference.py` | Test code; see Supplementary Material). |

### Latent configuration

The encoder produces a latent that is split into a subject-morphological code
(*z_p*) and a temporal-dynamics code (*z_t*), each of dimension **108**
(combined latent 216). 

---

## Related Work
- **biosignalGANs** (base architecture): [https://github.com/theekshanadis/biosignalGANs](https://github.com/theekshanadis/biosignalGANs)
---

## Contact
For questions or collaborations, please open an issue or contact the corresponding author: Theekshana Dissanayake, TU Berlin, BIFOLD
