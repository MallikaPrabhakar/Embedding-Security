# %%
# imports
import os
import numpy as np
import sys
sys.path.append('../')
from facenet_pytorch import MTCNN
import glob 
import cv2
import pandas as pd
from error_handling import pad_and_resize_image

# %%
folder_name = 'lfw'
mtcnn = MTCNN(keep_all=False, device='cuda:0')

# %%
path= '../datasets/'
d_raw = path+'raw/lfw/'

os.makedirs(f'{path}pp/{folder_name}/images', exist_ok=True)

# %%
lfw_all_names = pd.read_csv(f'{d_raw}lfw_allnames.csv')
names = lfw_all_names['name'].tolist()
names.sort()
id_images_dict = {}
img_id_map = pd.DataFrame(columns=['img', 'id', 'label'])
for i in range(len(lfw_all_names)):
    name = lfw_all_names.iloc[i]['name']
    id_images_dict[name] = lfw_all_names.iloc[i]['images']
# print(id_images_dict)

files = []
image_path = f'{d_raw}lfw-deepfunneled/lfw-deepfunneled/'

seen_count = 0
for i in range(len(names)):
    name = names[i]
    image = f'{name}_0001.jpg'
    if i >= len(names) - 1000:
        img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [image], 'id': [i], 'label': ['test']})], ignore_index=True)
    else:
        img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [image], 'id': [i], 'label': ['train']})], ignore_index=True)
    files.append(image_path + name + '/' + image)
    if seen_count < 1000 and id_images_dict[names[i]] > 1 and i < len(names)- 1000:
        seen_count += 1
        image = f'{names[i]}_0002.jpg'
        img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [image], 'id': [i], 'label': ['test_seen']})], ignore_index=True)
        files.append(image_path + names[i] + '/' + image)

print(f'Found {len(files)} files', flush=True)
img_id_map.to_csv(f'{path}pp/{folder_name}/img_id_map.csv', index=False)
# %%
dir = f'{path}pp/{folder_name}'
i = 0
for file in files:
    i+=1
    if (i % 100 == 0): print(i, flush=True)
    if os.path.exists(dir + '/images/' + file.split('/')[-1]):
        print('Skipping', file.split('/')[-1], flush=True)
        continue
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    write_path = f'{dir}/images/' + file.split('/')[-1]
    img_cropped = mtcnn(img, save_path=write_path)
    if img_cropped is None:
        img_cropped = pad_and_resize_image(img, (160, 160))
        cv2.imwrite(write_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))


# test2 split: a SECOND image of each TEST identity (merged from make_lfw_test2.py).

seed = 0
# pick a random non-_0001 image for each test identity (fixed seed)
rng = np.random.RandomState(seed)
cands = []
for _, r in img_id_map[img_id_map['label'] == 'test'].iterrows():
    name = r['img'][:-len('_0001.jpg')]              # test img is "{name}_0001.jpg"
    others = sorted(p for p in glob.glob(f'{image_path}{name}/{name}_*.jpg')
                    if not p.endswith('_0001.jpg'))
    if not others:                                   # only 1 image for this identity -> skip
        continue
    pick = others[rng.randint(len(others))]
    cands.append((int(r['id']), pick, os.path.basename(pick)))
print(f'{folder_name}: {len(cands)} test identities have a 2nd image (test2 candidates)', flush=True)

test2_rows = []
for k, (idv, src, out_name) in enumerate(cands):
    pp_path = f'{dir}/images/{out_name}'
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    img_cropped = mtcnn(img, save_path=pp_path)      # 160x160 aligned crop
    if img_cropped is None:                          # MTCNN found no face -> pad+resize fallback
        img_cropped = pad_and_resize_image(img, (160, 160))
        cv2.imwrite(pp_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
    test2_rows.append({'img': out_name, 'id': idv, 'label': 'test2'})
    if (k + 1) % 100 == 0:
        print(f'  {k+1}/{len(cands)}', flush=True)

# append test2 rows to the in-memory img_id_map and re-save (pp)
new = pd.DataFrame(test2_rows, columns=['img', 'id', 'label'])
img_id_map = pd.concat([img_id_map[img_id_map.label != 'test2'], new], ignore_index=True)
img_id_map.to_csv(f'{dir}/img_id_map.csv', index=False)
print(f'{folder_name}: wrote {len(new)} test2 rows + images (no embeddings)', flush=True)