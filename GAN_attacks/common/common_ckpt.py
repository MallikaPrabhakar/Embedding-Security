"""Shared checkpoint helper for BiDO-trained classifiers, used by GMI/ and DMI/.

Locates the trained *unprotected* FaceNet classifier, whose test-accuracy tag is
embedded in its filename (e.g. FaceNet_reg_facenet_79.52_unprotected_d.tar). That
checkpoint is the evaluator E for every attack and, for the `unprotected` scheme,
the target T as well -- so the attacks glob for it and read the accuracy back from
the filename instead of hardcoding it. The default ckpt_dir is relative to the
attack's working directory (GMI/ or DMI/), which sit alongside BiDO/.
"""
import os
import glob
import re


def unprot_ckpt(ckpt_dir="../BiDO/target_model/celeba/reg/facenet"):
    matches = sorted(glob.glob(os.path.join(ckpt_dir, "FaceNet_reg_facenet_*_unprotected_d.tar")))
    if not matches:
        raise FileNotFoundError(
            "No unprotected classifier checkpoint (FaceNet_reg_facenet_*_unprotected_d.tar) "
            f"in {ckpt_dir}")
    path = matches[-1]
    acc = float(re.search(r"facenet_([0-9.]+)_unprotected_d", os.path.basename(path)).group(1))
    return path, acc
