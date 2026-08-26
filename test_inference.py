import torch
torch.manual_seed(0)

# ------------------------------------------------------------------
# Instantiate all modules (as they exist at inference time)
# ------------------------------------------------------------------
model      = DisentangledGenerator(SPLIT_PER_CODE)   # E_f, E_z, D, subject head
skip       = SkipEncode()                            # E_sh (shift encoder)
disc_l     = LatentDiscriminator(in_dim=SPLIT_PER_CODE * 2)  # Dis_l
disc_r     = ReconDiscriminator()                    # Dis_r
subject_cl = SubjectClassifier()                     # Cls (frozen at inference)

model.eval(); skip.eval(); subject_cl.eval()

# ------------------------------------------------------------------
# A single seed segment: 2-channel spectrogram [B, 2, 128, 256]
# ------------------------------------------------------------------
xt = torch.rand(1, 2, 128, 256)
print("seed xt:", xt.shape)

# ------------------------------------------------------------------
# One forward pass of the base model (reconstruction path)
# ------------------------------------------------------------------
with torch.no_grad():
    (rec_t, rec_t_), z_p, z_t, z_p_, z_t_, feat, z, p_pred, skips, skips_ = model(xt, xt)

print("\n--- base model outputs ---")
print("reconstruction rec_t :", rec_t.shape)      # expect [1, 2, 128, 256]
print("z_p (subject code)   :", z_p.shape)        # expect [1, 108]
print("z_t (temporal code)  :", z_t.shape)        # expect [1, 108]
print("subject pred p_pred  :", p_pred.shape)     # expect [1, 17]

# ------------------------------------------------------------------
# Recurrent multi-step generation: seed -> X_{t+1} -> X_{t+2} -> X_{t+3}
# ------------------------------------------------------------------
@torch.no_grad()
def generate_next(cur):
    """Given current segment, produce the next one via the shift encoder."""
    feat = model.feature(cur)
    z, skips = model.latent(feat)
    z_p, z_t = torch.split_with_sizes(z, [SPLIT_PER_CODE, SPLIT_PER_CODE], dim=-1)
    z_t_next, z_delta, skips_lat = skip(feat, z_p, z_t)      # E_sh
    nxt = model.reconstruct(
        model.decode_main(torch.cat([z_p + z_delta, z_t_next], dim=-1), skips_lat))
    return nxt

print("\n--- recurrent generation ---")
cur = xt
steps = []
for i in range(3):                       # t+1, t+2, t+3
    cur = generate_next(cur)
    steps.append(cur)
    print(f"X_t+{i+1}:", cur.shape)       # each expect [1, 2, 128, 256]

# ------------------------------------------------------------------
# Discriminators + classifier (sanity — shapes only)
# ------------------------------------------------------------------
with torch.no_grad():
    z_t_next, z_delta, skips_lat = skip(model.feature(xt),
                                        *torch.split_with_sizes(
                                            model.latent(model.feature(xt))[0],
                                            [SPLIT_PER_CODE, SPLIT_PER_CODE], dim=-1))
    d_l = disc_l(torch.cat([z_t_next, z_t], dim=-1))   # latent disc
    d_r = disc_r(steps[0], xt)                          # recon disc (gen, ref)
    s_pred = subject_cl(steps[0])                       # subject id of generation

print("\n--- discriminators / classifier ---")
print("Dis_l (latent) :", d_l.shape)     # expect [1, 1]
print("Dis_r (recon)  :", d_r.shape)     # expect [1]
print("Cls subject    :", s_pred.shape)  # expect [1, 17]

print("\nALL PATHS RAN — pipeline is self-consistent at 108-per-code.")
