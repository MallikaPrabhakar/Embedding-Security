"""LPIPS (AlexNet) between originals and reconstructions, for PIPE, Bob, GMI and KED-MI,
current schemes (facenet). PIPE recons sourced from crossdataset/celeba; GMI/KED-MI recons
from GAN_attacks (see schemes.py). Run from the metrics/ dir (in the ddpm env)."""
import os
import re
import glob
import numpy as np
import cv2
import lpips

from schemes import (SCHEMES, MODEL, MODE, DATASET, ROOT, bob_dir,
                     GAN_ATTACKS, GAN_SCHEMES, CELEBA_IMG_DIR, gan_recon_dir, gan_offset,
                     load_iden_to_celeba, load_celeba_to_image)

lpips_metric = lpips.LPIPS(net="alex")
results = []


# ---------------- Bob ----------------
for model in [MODEL]:
    for disp, bob_scheme, _ in SCHEMES:
        path_bob = bob_dir(model, bob_scheme)
        oip = sorted(glob.glob(f"{path_bob}/images/*.npy"))
        rip = sorted(glob.glob(f"{path_bob}/reconstructed_images/*.npy"))
        if not oip or not rip:
            print(f"LPIPS Bob {model} {disp}: no data", flush=True)
            continue
        oi = [lpips.im2tensor(cv2.resize(np.moveaxis(np.load(p), 0, -1), (64, 64))) for p in oip]
        ri = [lpips.im2tensor(cv2.resize(np.moveaxis(np.load(p), 0, -1), (64, 64))) for p in rip]
        s = np.array([lpips_metric.forward(oi[i], ri[i]).item() for i in range(len(oi))])
        print(f"LPIPS Bob {model} {disp}: {s.mean():.3f} ± {s.std():.3f}", flush=True)
        results.append(("Bob", model, disp, s.mean(), s.std()))


# ---------------- PIPE (recons from crossdataset/celeba) ----------------
for model in [MODEL]:
    for disp, _, npz in SCHEMES:
        oip = sorted(glob.glob(f"{ROOT}/PIPE_attack/datasets/{DATASET}/{MODE}/images/*.jpg"))
        if not os.path.exists(npz):
            print(f"LPIPS Pipe {model} {disp}: no npz", flush=True)
            continue
        oi = np.array([cv2.cvtColor(cv2.resize(cv2.imread(p), (64, 64)), cv2.COLOR_RGB2BGR) for p in oip])
        ri = np.load(npz)["arr_0"]
        s = np.array([lpips_metric.forward(lpips.im2tensor(oi[i]), lpips.im2tensor(ri[i])).item()
                      for i in range(len(oi))])
        print(f"LPIPS Pipe {model} {disp}: {s.mean():.3f} ± {s.std():.3f}", flush=True)
        results.append(("PIPE", model, disp, s.mean(), s.std()))


# ---------------- GMI / KED-MI (GAN_attacks reconstructions) ----------------
# Each recon attack_iden_<L>|<seed>.png is compared to a representative original of the same
# CelebA identity (attack_iden L -> CelebA id -> img_id_map.csv -> celeba/images).
_GAN_RE = re.compile(r"attack_iden_(\d+)\|\d+\.png$")
_iden2cid = load_iden_to_celeba()
_cid2img = load_celeba_to_image()


def _load64(path):  # BGR file -> RGB 64x64 lpips tensor (matches the PIPE path)
    return lpips.im2tensor(cv2.cvtColor(cv2.resize(cv2.imread(path), (64, 64)), cv2.COLOR_BGR2RGB))


for attack_disp, attack_sub in GAN_ATTACKS:
    orig_cache = {}                               # CelebA id -> original tensor (loaded once)
    for scheme in GAN_SCHEMES:
        recon_files = sorted(glob.glob(f"{gan_recon_dir(attack_sub, scheme)}/attack_iden_*.png"))
        if not recon_files:
            print(f"LPIPS {attack_disp} facenet {scheme}: no data", flush=True)
            continue
        off = gan_offset(attack_sub, scheme)
        vals = []
        for rp in recon_files:
            m = _GAN_RE.search(os.path.basename(rp))
            if not m:
                continue
            cid = _iden2cid.get(int(m.group(1)) - off)
            img = _cid2img.get(cid)
            if img is None:
                continue
            op = os.path.join(CELEBA_IMG_DIR, img)
            if not os.path.exists(op):
                continue
            if cid not in orig_cache:
                orig_cache[cid] = _load64(op)
            vals.append(lpips_metric.forward(orig_cache[cid], _load64(rp)).item())
        s = np.array(vals)
        print(f"LPIPS {attack_disp} facenet {scheme}: {s.mean():.3f} ± {s.std():.3f}  (n={len(s)})", flush=True)
        results.append((attack_disp, "facenet", scheme, s.mean(), s.std()))


# save a CSV of the LPIPS results in metrics/
with open("lpips_results.csv", "w") as f:
    f.write("attack,model,scheme,lpips_mean,lpips_std\n")
    for attack, model, disp, m, sd in results:
        f.write(f"{attack},{model},{disp},{m:.4f},{sd:.4f}\n")
print("wrote lpips_results.csv", flush=True)
