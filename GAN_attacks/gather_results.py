#!/usr/bin/env python
"""Score the saved GMI/DMI reconstructions and write GAN_attacks/RESULTS_SUMMARY.md.

For every scheme it reports, per reconstructed image, the **ASR-L2**: a reconstruction
succeeds if the FaceNet *evaluation* embedding's euclidean_l2 (L2 of unit-normalised
embeddings) to the target identity's train-time embeddings is < 1.1027 (the Facenet512
threshold) -- scored by an independent FaceNet evaluator, not by any scheme's own classifier.

The scheme's trained-classifier test accuracy (parsed from its checkpoint filename) is also
reported.

  CUDA_VISIBLE_DEVICES=<g> python gather_results.py
"""
import os, sys, re, glob, collections
import torch
import torch.nn.functional as F
import PIL.Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))   # GAN_attacks
REPO = os.path.abspath(os.path.join(HERE, ".."))    # Embedding-Security
BIDO = os.path.join(HERE, "BiDO")
# facenet_pytorch lives at the repo root; BiDO/model.py only adds it via a
# cwd-relative '../..', so put both on sys.path here to stay cwd-independent.
sys.path.insert(0, REPO)
sys.path.insert(0, BIDO)
import model as bido_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 1.1027                                                  # Facenet512 euclidean_l2
EVAL_TAR = os.path.join(BIDO, "eval_model", "FaceNet_pytorch.tar")  # embedding evaluator
CKPT_DIR = os.path.join(BIDO, "target_model/celeba/reg/facenet")    # per-scheme classifiers (acc tag)
ORIG_DIR = os.path.join(REPO, "datasets/pp/celeba/images")
TRAIN_FILE = os.path.join(HERE, "attack_dataset/CelebA/trainset_jpg.txt")  # embedding gallery
FNAME_RE = re.compile(r"attack_iden_(\d+)\|(\d+)\.png$")
_to_tensor = transforms.ToTensor()
N_REF = 10


# scheme -> (gmi dir, gmi offset, dmi dir, dmi offset). The old unprotected GMI run saved gt+1.
def _dir(attack, prot): return os.path.join(HERE, attack, "attack_res/celeba/reg/facenet", f"{prot}_d", "all")
SCHEMES = [
    ("unprotected",               _dir("GMI", "unprotected"),               1, _dir("DMI", "unprotected"),               0),
    ("facialfe_e8",               _dir("GMI", "facialfe_e8"),               0, _dir("DMI", "facialfe_e8"),               0),
    ("facialfe_leech",            _dir("GMI", "facialfe_leech"),            0, _dir("DMI", "facialfe_leech"),            0),
    ("facialfe_leech_randomised", _dir("GMI", "facialfe_leech_randomised"), 0, _dir("DMI", "facialfe_leech_randomised"), 0),
    ("ourfe",                     _dir("GMI", "ourfe"),                     0, _dir("DMI", "ourfe"),                     0),
]


def low2high(img, dim):
    img = F.interpolate(img, size=(dim, dim), mode="bilinear", align_corners=False)
    return (img * 255.0 - 127.5) / 128.0


def load_imgs(paths, img_dir=""):
    return torch.stack([_to_tensor(PIL.Image.open(os.path.join(img_dir, p)).convert("RGB")) for p in paths])


def load_state(model, tar):
    sd = torch.load(tar, map_location="cpu"); sd = sd.get("state_dict", sd)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    return model.to(DEVICE).eval()


def ckpt_for(prot):
    """Locate a scheme's trained-classifier checkpoint (only its filename accuracy tag is used)."""
    g = sorted(glob.glob(os.path.join(CKPT_DIR, f"FaceNet_reg_facenet_*_{prot}_d.tar")))
    return g[-1] if g else None


def acc_of(tar):
    return float(re.search(r"facenet_([0-9.]+)_", os.path.basename(tar)).group(1)) if tar else None


def evaluate(recon_dir, offset, E, galleries):
    """Return embed_asr_per_img in %, or None if no recons."""
    recon = collections.defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(recon_dir, "attack_iden_*.png"))):
        m = FNAME_RE.search(os.path.basename(fp))
        if m:
            recon[int(m.group(1)) - offset].append(fp)
    if not recon:
        return None
    emb_hits = tot = 0
    with torch.no_grad():
        for iden, paths in recon.items():
            if iden not in galleries:
                continue
            imgs = load_imgs(paths).to(DEVICE)
            # ASR-L2: FaceNet evaluator, min unit-norm L2 to the identity's gallery < threshold
            re_emb = F.normalize(E(low2high(imgs, 160)), dim=1)
            emb_hits += int((torch.cdist(re_emb, galleries[iden]).min(dim=1).values < THRESHOLD).sum())
            tot += len(paths)
    if tot == 0:
        return None
    return 100.0 * emb_hits / tot


def main():
    # embedding evaluator (independent FaceNet, classify=False, 512-d)
    E = load_state(bido_model.InceptionResnetV1().to(DEVICE), EVAL_TAR)

    # per-label gallery (training images) for the embedding ASR
    gal_names = collections.defaultdict(list)
    with open(TRAIN_FILE) as f:
        for line in f:
            p = line.split()
            if len(p) == 2 and len(gal_names[int(p[1])]) < N_REF:
                gal_names[int(p[1])].append(p[0])
    galleries = {}
    with torch.no_grad():
        for iden, names in gal_names.items():
            galleries[iden] = F.normalize(E(low2high(load_imgs(names, ORIG_DIR).to(DEVICE), 160)), dim=1)

    rows = []
    for prot, gdir, goff, ddir, doff in SCHEMES:
        acc = acc_of(ckpt_for(prot))                            # the scheme's classifier test acc
        g_emb = evaluate(gdir, goff, E, galleries)
        d_emb = evaluate(ddir, doff, E, galleries)
        rows.append((prot, acc, g_emb, d_emb))
        print(f"{prot}: acc={acc} GMI(emb={g_emb}) DMI(emb={d_emb})")

    def f(x): return "N/A" if x is None else f"{x:.2f}"
    md = ["# GAN-attack results on CelebA / FaceNet (per-image embedding ASR)", "",
          "ASR-L2: a reconstruction succeeds if the FaceNet *evaluation* embedding's euclidean_l2",
          "(unit-normalised) to the target identity's train-time embedding is < 1.1027 (the Facenet512",
          "threshold). \n",
          "Classifier acc is the scheme's trained-classifier test accuracy.", "",
          "| scheme | classifier acc (%) | GMI ASR-L2 | DMI ASR-L2 |",
          "|---|---|---|---|"]
    for prot, acc, ge, de in rows:
        md.append(f"| {prot} | {f(acc)} | {f(ge)} | {f(de)} |")
    md += ["", "To (re)generate the table from the saved reconstructions:", "",
           "```bash", "bash scripts/aggregate_results.sh 0          # writes RESULTS_SUMMARY.md", "```"]
    out = os.path.join(HERE, "RESULTS_SUMMARY.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
