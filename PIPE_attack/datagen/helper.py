import os
import numpy as np
import pandas as pd

def build_pairs(dataset, model, mode='train', samples=0, seed=0, unprotected_dir=None):
    """Genuine/impostor pairs from the train & test_seen (mode = train) or test & test2 (mode = test) splits."""
    folder = unprotected_dir or f'../datasets/unprotected/{dataset}/{model}'
    img_id = pd.read_csv(f'../datasets/{dataset}/img_id_map.csv')

    def emb_path(label, img):
        # celeba/lfw store zero-padded 6-digit names; casia stores unpadded -> fall back.
        base = f"{folder}/{label}/embeddings"
        stem = str(img[:-4])
        padded = f"{base}/{stem.zfill(6)}.npy"
        return padded if os.path.exists(padded) else f"{base}/{stem}.npy"

    def split_map(label):
        sub = img_id[img_id['label'] == label]
        return {id_: emb_path(label, img) for img, id_ in zip(sub['img'], sub['id'])}

    if mode == 'train':
        f1, f2 = split_map('train'), split_map('test_seen')
    elif mode == 'test':
        f1, f2 = split_map('test'), split_map('test2')
    else:
        raise ValueError(f'invalid mode {mode}; expected train or test')
    ids = sorted(set(f1.keys()) & set(f2.keys()))
    if not ids:
        raise FileNotFoundError(f'no shared {mode} identities for {dataset}/{model}')

    rng = np.random.RandomState(seed)
    same_pairs = [[f1[i], f2[i]] for i in ids]
    diff_pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            diff_pairs.append([f1[ids[i]], f2[ids[j]]])

    if samples:
        k = samples // 2
        same_pairs = [same_pairs[t] for t in np.sort(rng.choice(len(same_pairs), min(k, len(same_pairs)), replace=False))]
        diff_pairs = [diff_pairs[t] for t in np.sort(rng.choice(len(diff_pairs), min(k, len(diff_pairs)), replace=False))]
    return same_pairs, diff_pairs

