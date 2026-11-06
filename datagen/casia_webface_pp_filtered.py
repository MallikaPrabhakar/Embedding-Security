# imports
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('../')
from facenet_pytorch import MTCNN
import cv2 
from error_handling import pad_and_resize_image

# %%
mtcnn = MTCNN(keep_all=False, device='cuda:0')
np.bool = bool
import mxnet as mx

output_dir = "../datasets/raw/casia_webface"
img_dir = "../datasets/raw/casia_webface/images"

write_dir = "../datasets/pp/casia_webface/"
os.makedirs(write_dir, exist_ok=True)
os.makedirs(os.path.join(write_dir, 'images'), exist_ok=True)  # MTCNN only makedirs on a detected face; ensure images/ exists for the no-face fallback
img_id_map = pd.DataFrame(columns=['img', 'id', 'label'])


# Create output folder
os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)


rec_path = "../datasets/raw/faces_webface_112x112/train.rec"
idx_path = "../datasets/raw/faces_webface_112x112/train.idx"

record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
label = -1
total = 9576+996
print(record)
count = 0
test_seen_count = 0
for idx in tqdm(record.keys):
        # print(idx)
        item = record.read_idx(idx)
        header, img = mx.recordio.unpack(item)
        # print(header)
        if header.flag != 0:
            continue
        new_label = header.label
        if new_label != label:
            label = new_label
            if count < total - 1000:
                folder = 'train'
            else:
                folder = 'test'
            img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [str(header.id)+".jpg"], 'id': [int(header.label)], 'label': [folder]})], ignore_index=True)
            img_arr = mx.image.imdecode(img).asnumpy().astype(np.uint8)
            count+=1
            if os.path.exists(os.path.join(write_dir, 'images', f'{header.id}.jpg')):
                print(f"Image {header.id} already exists, skipping.")
                continue
            img_cropped = mtcnn(img_arr, save_path=write_dir+f'images/{header.id}.jpg')

            if img_cropped is None:
                img_cropped = pad_and_resize_image(img_arr, (160, 160))
                cv2.imwrite(write_dir+f'images/{header.id}.jpg', cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
        else:
            continue


print(count)

# %%
# test_seen + test2 splits (merged from make_casia_test_seen.py + extract_casia_test2.py).
N_TEST_SEEN = 1000
train_ids = set(int(x) for x in img_id_map[img_id_map['label'] == 'train']['id'])
test_ids = set(int(x) for x in img_id_map[img_id_map['label'] == 'test']['id'])

def align_and_save(img_arr, out_name):
    pp_path = write_dir + f'images/{out_name}'
    cropped = mtcnn(img_arr, save_path=pp_path)      # 160x160 aligned crop
    if cropped is None:                              # MTCNN found no face -> pad+resize fallback
        out = pad_and_resize_image(img_arr, (160, 160))
        cv2.imwrite(pp_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

# RecordIO groups images by identity, so the 2nd image of each id appears right after its 1st.
seen_first, ts_picked, t2_picked = set(), set(), set()
ts_rows, t2_rows = [], []
for idx in tqdm(record.keys):
    header, img = mx.recordio.unpack(record.read_idx(idx))
    if header.flag != 0:
        continue
    lab = int(header.label)
    if lab not in seen_first:                        # 1st image == the existing train/test image
        seen_first.add(lab)
        continue
    out_name = f'{int(header.id)}.jpg'
    if lab in train_ids and lab not in ts_picked and len(ts_picked) < N_TEST_SEEN:
        align_and_save(mx.image.imdecode(img).asnumpy().astype(np.uint8), out_name)
        ts_picked.add(lab)
        ts_rows.append({'img': out_name, 'id': lab, 'label': 'test_seen'})
    elif lab in test_ids and lab not in t2_picked:
        align_and_save(mx.image.imdecode(img).asnumpy().astype(np.uint8), out_name)
        t2_picked.add(lab)
        t2_rows.append({'img': out_name, 'id': lab, 'label': 'test2'})
print(f'casia_webface: wrote {len(t2_rows)} test2 + {len(ts_rows)} test_seen rows + images (no embeddings)', flush=True)

# append test2 then test_seen (matching the existing pp/img_id_map.csv layout) and re-save
extra = pd.DataFrame(t2_rows + ts_rows, columns=['img', 'id', 'label'])
img_id_map = pd.concat([img_id_map[~img_id_map['label'].isin(['test2', 'test_seen'])], extra], ignore_index=True)
img_id_map.to_csv(os.path.join(write_dir, 'img_id_map.csv'), index=False)
record.close()