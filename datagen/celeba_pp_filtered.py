# %%
# imports
import os
import pandas as pd
import sys
sys.path.append('../')
from facenet_pytorch import MTCNN
import cv2 
from error_handling import pad_and_resize_image

# %%
mtcnn = MTCNN(keep_all=False, device='cuda:0')

# %%
read_path = '../datasets/raw/CelebA/'
path = '../datasets/pp/celeba/'
os.makedirs(f'{path}images', exist_ok=True)


# %%
all_img_id_maps = pd.read_csv(f'{read_path}Anno/identity_CelebA.txt', delim_whitespace=True, header=None, names=['img', 'id'])

ids = all_img_id_maps['id'].unique()
id_img_map = dict()
for id in ids:
    images = list(all_img_id_maps[all_img_id_maps['id'] == id]['img'].values)
    images.sort()
    id_img_map[id] = images[0]

id_img_map = pd.DataFrame.from_dict(id_img_map, orient='index', columns=['img'])
id_img_map.index.name = 'id'
id_img_map.reset_index(inplace=True)
clipped_id_img_map = id_img_map[:-1000]

# %%
img_id_map = pd.DataFrame(columns=['img', 'id', 'label'])
for row in clipped_id_img_map.itertuples():
    img = row[2]
    id = int(row[1])
    img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [img], 'id': [id], 'label': ['train']})], ignore_index=True)

top_9177_ids = set(clipped_id_img_map['id'])
top_9177_imgs = set(clipped_id_img_map['img'])
train = top_9177_imgs
images_to_process = train
not_seen_ids = set(all_img_id_maps['id']) - top_9177_ids
test = []
not_seen_ids = list(not_seen_ids)
not_seen_ids.sort()
for id in not_seen_ids:
    img = id_img_map[id_img_map['id'] == id]['img'].values[0]
    test.append(img)
    img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [img], 'id': [id], 'label': ['test']})], ignore_index=True)
images_to_process = images_to_process.union(set(test))

# %%
# get images for all ids in all_img_id_maps
all_imgs = dict()
for i in range(1,10178):
    imgs = all_img_id_maps[all_img_id_maps['id'] == i]['img'].values
    all_imgs[i] = imgs

# %%
count = 0
test_seen = []
top_9177_ids = list(top_9177_ids)
top_9177_ids.sort()
for id in top_9177_ids:
    if count==1000:
        break
    imgs = all_imgs[id]
    if len(imgs) > 1: 
        if imgs[0] in top_9177_imgs:
            img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [imgs[1]], 'id': [id], 'label': ['test_seen']})], ignore_index=True)
            test_seen.append(imgs[1])
        count+=1
# print(len(img_id_map))
images_to_process = images_to_process.union(set(test_seen))

## add another image for each test id
count = 0
test2 = []
image_count = 202600
for id in not_seen_ids:
    if count==1000:
        break
    imgs = all_imgs[id]
    if len(imgs) > 1: 
        if imgs[0] in test:
            img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [imgs[1]], 'id': [id], 'label': ['test2']})], ignore_index=True)
            test2.append(imgs[1])
            count+=1
    else :
        # 3 images do not have a second image, we augment the dataset with a horizontally flipped version of the existing image for these ids, and label them as test2 as well
        augmented_img = cv2.imread(f'{read_path}Img/img_align_celeba/{imgs[0]}')
        augmented_img = cv2.flip(augmented_img, 1) # horizontal flip
        cv2.imwrite(f'{read_path}Img/img_align_celeba/{image_count}.jpg', augmented_img)
        img_id_map = pd.concat([img_id_map, pd.DataFrame({'img': [f'{image_count}.jpg'], 'id': [id], 'label': ['test2']})], ignore_index=True)
        test2.append(f'{image_count}.jpg')
        image_count+=1
        count+=1

images_to_process = images_to_process.union(set(test2))
print(count, flush=True)


print("PIPE, Bob attack images", len(images_to_process), flush=True)
img_id_map.to_csv(f'{path}img_id_map.csv', index=False)

# %%
# additional images needed for GMI, KED-MI
gan_attack_imgs = pd.read_csv('../GAN_attacks/attack_dataset/CelebA/all_filenames_jpg.csv', header=None, names=['img'])

print("Gan attack images", len(gan_attack_imgs), flush=True)
gan_attack_imgs = list(gan_attack_imgs['img'].values)
gan_attack_imgs = set(gan_attack_imgs)
# %%
print("common images", len(images_to_process.intersection(set(gan_attack_imgs))), flush=True)
images_to_process = images_to_process.union(set(gan_attack_imgs))
images_to_process = list(images_to_process); print(len(images_to_process), flush=True)
images_to_process.sort()
print("Total images to process", len(images_to_process), flush=True)
# %%
i= 0
for file in images_to_process:
    i+=1
    if (i % 100 == 0): print(i, file, flush=True)
    if os.path.exists(path + '/images/' + file.split('/')[-1]):
        print('Skipping', file.split('/')[-1], flush=True)
        continue
    img_path = f'{read_path}Img/img_align_celeba/{file}'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    write_path = f'{path}/images/' + file.split('/')[-1]
    img_cropped = mtcnn(img, save_path=write_path)
    if img_cropped is None:
        img_cropped = pad_and_resize_image(img, (160, 160))
        cv2.imwrite(write_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
