"""Shared scheme config for the PIPE / Bob / GAN LPIPS + FID metrics (5 schemes).

Each entry: (display name, Bob database scheme dir, PIPE reconstruction npz).
PIPE reconstructions are sourced from the unified crossdataset/celeba layout
(unprotected & facialfe_e8 @25k, the rest @50k). facenet only (Bob current-scheme
recons + our PIPE recons are facenet).
"""
import os
import dotenv
dotenv.load_dotenv()  # for PROJECT_ROOT
ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# PIPE recons: the clean crossdataset/celeba tree -> {ckpt}/{scheme}/celeba/facenet
XD = f"{ROOT}/PIPE_attack/results"
MODEL = 'facenet'
MODE = 'test'
NPZ = f"{MODE}_samples_1000x64x64x3.npz"
DATASET = "celeba"

SCHEMES = [
    # display                     Bob scheme dir                              PIPE recon npz (crossdataset/celeba)
    ("unprotected",               "unprotected_none",                          f"{XD}/ckpt_025000/unprotected/celeba/{MODEL}/{NPZ}"),
    ("facialfe_e8",               "facialfe_e8_none",                          f"{XD}/ckpt_025000/facialfe_e8/celeba/{MODEL}/{NPZ}"),
    ("facialfe_leech",            "facialfe_leech_centered_none",              f"{XD}/ckpt_050000/facialfe_leech/celeba/{MODEL}/{NPZ}"),
    ("facialfe_leech_randomised", "facialfe_leech_randomised_centered_none",   f"{XD}/ckpt_050000/facialfe_leech_randomised/celeba/{MODEL}/{NPZ}"),
    ("ourfe",                     "ourfe_none",                                f"{XD}/ckpt_050000/ourfe/celeba/{MODEL}/{NPZ}"),
]

def bob_dir(model, bob_scheme):
    return f"{ROOT}/Bob_attack/experiments/DataGen/databases/{DATASET}/{model}/{bob_scheme}/{MODE}"


# ---------------- GAN attacks (GMI / KED-MI), GAN_attacks/ ----------------
# Reconstructions: GAN_attacks/{GMI,DMI}/attack_res/celeba/reg/facenet/<scheme>_d/all/
#   attack_iden_<L>|<seed>.png  (1000 attack identities x 5 seeds = 5000 each).
# The attack_iden L (0-999) is NOT the CelebA id; map it with attack_iden_to_celeba_id.txt,
# then pick a representative original image for that CelebA id from img_id_map.csv.
GAN = f"{ROOT}/GAN_attacks"
GAN_ATTACKS = [("GMI", "GMI"), ("KEDMI", "DMI")]          # (display label, attack_res subdir)
GAN_SCHEMES = ["unprotected", "facialfe_e8", "facialfe_leech",
               "facialfe_leech_randomised", "ourfe"]      # schemes GAN_attacks actually ran
CELEBA_IMG_DIR = f"{ROOT}/datasets/pp/{DATASET}/images"


def gan_recon_dir(attack_subdir, scheme):
    return f"{GAN}/{attack_subdir}/attack_res/{DATASET}/reg/facenet/{scheme}_d/all"


def gan_offset(attack_subdir, scheme):
    """true attack_iden = filename_number - offset. The old unprotected GMI run saved
    attack_iden_{gt+1}; every other run (and all KED-MI) saved {gt}. Mirrors gather_results.py."""
    return 1 if (attack_subdir == "GMI" and scheme == "unprotected") else 0


def load_iden_to_celeba():
    """attack_iden (0-999) -> CelebA identity, from attack_iden_to_celeba_id.txt."""
    m = {}
    with open(f"{GAN}/attack_dataset/CelebA/attack_iden_to_celeba_id.txt") as f:
        for line in f:
            a, c = line.split()
            m[int(a)] = int(c)
    return m


def load_celeba_to_image():
    """CelebA identity -> a representative original image filename (first by name in img_id_map.csv)."""
    import csv
    m = {}
    with open(f"{ROOT}/datasets/pp/{DATASET}/img_id_map.csv") as f:
        for row in csv.DictReader(f):
            m.setdefault(int(row["id"]), row["img"])
    return m
