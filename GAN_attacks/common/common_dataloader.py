"""Common dataloader shared by BiDO/, GMI/ and DMI/ (CelebA only).

Consolidates the previously-duplicated `ImageFolder` class. The standardization
and resizing that used to differ between dataloader.py and dataloader_normalize.py
(and between BiDO/GMI/DMI) are reconciled into two flags, so each directory's
utils.py keeps its own behaviour by passing the appropriate values:

    standardize=True  -> append (x*255-127.5)/128   (maps [0,1] -> ~[-1,1], FaceNet convention)
    resize=True       -> Resize((112, 112))         (FaceNet / IR50 input size)

Each per-directory dataloader.py / dataloader_normalize.py is a thin shim that
re-exports ImageFolder from this module. This module has NO `import utils`,
removing the old utils<->dataloader circular import.
"""
import os
import PIL
from PIL import ImageFile
import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

RE_SIZE = 112  # FaceNet / IR50 input size


class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, mode, standardize=True, resize=True):
        self.args = args
        self.mode = mode
        self.standardize = standardize
        self.resize = resize
        self.img_path = args["dataset"]["img_path"]
        self.model_name = args["dataset"]["model_name"]
        self.processor = self.get_processor()
        self.image_list, self.label_list = self.get_list(file_path)
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        if self.mode != "gan":
            print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        with open(file_path, "r") as f:
            for line in f.readlines():
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
                name_list.append(img_name)
        return name_list, label_list

    def load_img(self, path):
        img = PIL.Image.open(os.path.join(self.img_path, path))
        img = img.convert('RGB')
        return img

    def get_processor(self):
        proc = []
        if self.resize:
            proc.append(transforms.Resize((RE_SIZE, RE_SIZE)))
        if self.mode == "train":
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
        proc.append(transforms.ToTensor())  # [0,255] PIL -> [0,1] tensor
        if self.standardize:
            proc.append(transforms.Lambda(lambda x: (x * 255.0 - 127.5) / 128.0)) # [0,1] -> ~[-1,1], FaceNet convention
        return transforms.Compose(proc)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img = self.processor(self.load_img(img_name))
        label = self.label_list[index]
        # img_name (img_id) is needed by the seeded PIPE FE schemes (facialfe_leech_randomised,
        # ourfe); label stays the CE target for classifier training.
        return img, img_name, label

    def __len__(self):
        return self.num_img
