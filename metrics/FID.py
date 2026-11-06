"""Write 64x64 PNGs for FID: originals + reconstructions, for PIPE, Bob, GMI and KED-MI,
current schemes (facenet). metrics.sh then runs `pytorch_fid` on the folders.
Run from the metrics/ dir (in the ddpm env)."""
import numpy as np
import cv2
import glob
import os
import re

from schemes import (SCHEMES, MODEL, MODE, DATASET, ROOT, bob_dir,
                     GAN_ATTACKS, GAN_SCHEMES, CELEBA_IMG_DIR, gan_recon_dir, gan_offset,
                     load_iden_to_celeba, load_celeba_to_image)


# ---------------- Bob ----------------
for model in [MODEL]:
    for disp, bob_scheme, _ in SCHEMES:
        path_bob = bob_dir(model, bob_scheme)
        original_image_paths = sorted(glob.glob(f"{path_bob}/images/*.npy"))
        reconstructed_image_paths = sorted(glob.glob(f"{path_bob}/reconstructed_images/*.npy"))
        if not original_image_paths or not reconstructed_image_paths:
            print(f"[FID/bob] SKIP {model}/{bob_scheme} (no data)", flush=True)
            continue

        og_path = "fid/original_bob/"
        rec_path = f"fid/reconstructed_bob/{model}/{bob_scheme}"
        os.makedirs(og_path, exist_ok=True)
        os.makedirs(rec_path, exist_ok=True)
        for i, (op, rp) in enumerate(zip(original_image_paths, reconstructed_image_paths)):
            og_img = cv2.resize(np.moveaxis(np.load(op), 0, -1), (64, 64))
            rec_img = cv2.resize(np.moveaxis(np.load(rp), 0, -1), (64, 64))
            if not os.path.exists(og_path + f"{i}.png"):
                cv2.imwrite(og_path + f"{i}.png", og_img)
            cv2.imwrite(rec_path + f"/{i}.png", rec_img)
        print(f"[FID/bob] wrote {len(original_image_paths)} {model}/{bob_scheme}", flush=True)


# ---------------- PIPE (recons from crossdataset/celeba) ----------------
for model in [MODEL]:
    for disp, _, npz in SCHEMES:
        original_image_paths = sorted(glob.glob(f"{ROOT}/PIPE_attack/datasets/{DATASET}/{MODE}/images/*.jpg"))
        if not os.path.exists(npz):
            print(f"[FID/pipe] SKIP {model}/{disp} (no npz {npz})", flush=True)
            continue

        original_images = np.array([cv2.resize(cv2.imread(p), (64, 64)) for p in original_image_paths])
        reconstructed_images = np.load(npz)["arr_0"]
        reconstructed_images = np.array([cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in reconstructed_images])

        og_path = "fid/original_pipe/"
        rec_path = f"fid/reconstructed_pipe/{model}/{disp}"
        os.makedirs(og_path, exist_ok=True)
        os.makedirs(rec_path, exist_ok=True)
        for i in range(len(original_images)):
            if not os.path.exists(og_path + f"{i}.png"):
                cv2.imwrite(og_path + f"{i}.png", cv2.resize(original_images[i], (64, 64)))
            cv2.imwrite(rec_path + f"/{i}.png", cv2.resize(reconstructed_images[i], (64, 64)))
        print(f"[FID/pipe] wrote {len(original_images)} {model}/{disp}", flush=True)


# ---------------- GMI / KED-MI (GAN_attacks reconstructions) ----------------
# Recons named attack_iden_<L>|<seed>.png; map L -> CelebA id -> a representative original.
# Originals go to a shared fid/original_gan/, one PNG per CelebA identity (FID is distributional,
# so the real set is just the attacked identities' originals; recons are all 5000 per scheme).
_GAN_RE = re.compile(r"attack_iden_(\d+)\|\d+\.png$")
_iden2cid = load_iden_to_celeba()
_cid2img = load_celeba_to_image()
og_path = "fid/original_gan/"
os.makedirs(og_path, exist_ok=True)

for attack_disp, attack_sub in GAN_ATTACKS:
    src = attack_disp.lower()                      # gmi / kedmi
    for scheme in GAN_SCHEMES:
        recon_files = sorted(glob.glob(f"{gan_recon_dir(attack_sub, scheme)}/attack_iden_*.png"))
        if not recon_files:
            print(f"[FID/{src}] SKIP facenet/{scheme} (no data)", flush=True)
            continue
        rec_path = f"fid/reconstructed_{src}/facenet/{scheme}"
        os.makedirs(rec_path, exist_ok=True)
        off = gan_offset(attack_sub, scheme)
        n = 0
        for i, rp in enumerate(recon_files):
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
            if not os.path.exists(og_path + f"{cid}.png"):     # one original per identity
                cv2.imwrite(og_path + f"{cid}.png", cv2.resize(cv2.imread(op), (64, 64)))
            cv2.imwrite(rec_path + f"/{i}.png", cv2.resize(cv2.imread(rp), (64, 64)))
            n += 1
        print(f"[FID/{src}] wrote {n} facenet/{scheme}", flush=True)
