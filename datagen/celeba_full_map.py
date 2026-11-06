"""Build img_id_map_full.csv: a full CelebA split

Pulls *all* images of every identity from identity_CelebA.csv (unlike
img_id_map.csv, which keeps a single representative image per identity — too few
for triplet loss + PKSampler, which need K images per identity per batch), then
labels each image using the existing img_id_map.csv split:

  * 'test'      — every image whose identity is a held-out test identity
                  (id has label 'test' in img_id_map.csv; unseen at train time).
  * 'test_seen' — the specific held-out images (img has label 'test_seen' in
                  img_id_map.csv; seen identities, unseen images).
  * 'train'     — everything else (all remaining images of the seen identities).


Note: identity_CelebA.csv references all ~202k CelebA images, but only the
subset cropped into <images_dir> can actually be loaded. The script reports how
many referenced images are missing on disk; pass --require_existing to drop them.
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CELEBA = os.path.join(REPO, 'datasets', 'pp', 'celeba')


def build_args():
    ap = argparse.ArgumentParser(
        description='Create img_id_map_full.csv from identity_CelebA.csv and img_id_map.csv')
    ap.add_argument('--out_csv', default=os.path.join(CELEBA, 'img_id_map_full.csv'))
    ap.add_argument('--images_dir', default=os.path.join(CELEBA, 'images'),
                    help='Used to report (and, with --require_existing, filter by) '
                         'on-disk image availability.')
    ap.add_argument('--require_existing', action='store_true',
                    help='Keep only rows whose image exists under --images_dir.')
    return ap.parse_args()


def main():
    args = build_args()

    ident = pd.read_csv(os.path.join(CELEBA, 'identity_CelebA.csv'))  # columns: img, id
    split = pd.read_csv(os.path.join(CELEBA, 'img_id_map.csv'))       # columns: img, id, label
    ident_images = set(ident['img'].values)
    # add augmented images to ident from split
    for idx, row in split.iterrows():
        if row['img'] not in ident_images:
            ident = pd.concat([ident, pd.DataFrame({'img': [row['img']], 'id': [row['id']]})], ignore_index=True)

    print(split[ split['label'] == 'test2' ].sort_values('img'), flush=True)

    test_ids = set(split.loc[split['label'].isin(['test', 'test2']), 'id'])
    test_seen_imgs = set(split.loc[split['label'] == 'test_seen', 'img'])

    # Relabel every CelebA image. test_seen images belong to seen (train) ids,
    # so they start as 'train'; applying test (by id) then test_seen / test2
    # (by img) is order-independent: test_seen ids are disjoint from test, and
    # test2 imgs sit inside test ids (a specific second image per held-out id).
    ft = ident[['img', 'id']].copy()
    ft['label'] = 'train'
    ft.loc[ft['id'].isin(test_ids), 'label'] = 'test'
    ft.loc[ft['img'].isin(test_seen_imgs), 'label'] = 'test_seen'

    def label_summary(df):
        order = ['train', 'test_seen', 'test']
        counts = df['label'].value_counts()
        return ', '.join(f'{lab}={int(counts.get(lab, 0))}' for lab in order)

    print(f'labels: {label_summary(ft)}  (total {len(ft)} rows, '
          f'{ft["id"].nunique()} identities)', flush=True)

    # On-disk availability: only images cropped into images_dir can be loaded.
    if os.path.isdir(args.images_dir):
        present = set(os.listdir(args.images_dir))
        exists = ft['img'].isin(present)
        by = ft.assign(_e=exists).groupby('label')['_e'].agg(['sum', 'size'])
        for lab in ('train', 'test_seen', 'test'):
            if lab in by.index:
                print(f'  on disk [{lab:<9}]: {int(by.loc[lab, "sum"])}/'
                      f'{int(by.loc[lab, "size"])}', flush=True)
        if args.require_existing:
            ft = ft.loc[exists]
            print(f'--require_existing: kept {len(ft)} rows ({label_summary(ft)})', flush=True)
    elif args.require_existing:
        raise SystemExit(f'--require_existing set but images_dir not found: {args.images_dir}')

    ft.to_csv(args.out_csv, index=False)
    print(f'wrote {len(ft)} rows -> {args.out_csv}', flush=True)


if __name__ == '__main__':
    main()
