import json, time, torch, sys
import torch.nn.functional as F
import torchvision.utils as tvls
from datetime import datetime
import dataloader

device = "cuda"


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()

def init_dataloader(args, file_path, batch_size=64, mode="gan"):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    # standardize to ~[-1,1] but no resize (preserves old GMI/dataloader.py behaviour)
    data_set = dataloader.ImageFolder(args, file_path, mode, standardize=True, resize=False)

    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle_flag,
                                              num_workers=2,
                                              pin_memory=True)

    interval = time.time() - tf
    print('Initializing data loader took %ds' % interval)
    return data_set, data_loader


def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')


# normalize=False: G is sigmoid (already in [0,1]), so save the raw output as-is. normalize=True
# would per-image min-max contrast-stretch it, making the saved PNG diverge from the float tensor
# the attack scored (the ~1% offline-vs-log ASR gap); the only residual is 8-bit PNG rounding.
def save_tensor_images(images, filename, nrow=None, normalize=False):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)


def low2high(img, dim=160):
    """Upsample G(z) (64x64, [0,1]) to the target/eval FaceNet input size.

    We use F.interpolate instead of transforms.ToPILImage() to remain differentiable. 
    There should be no detaching here. 
    We also standardise to ~[-1,1] to match the preprocessing that the target model 
    was trained with (BiDO dataloader applies (x*255-127.5)/128; facenet eval expects the same range)."""
    img = F.interpolate(img, size=(dim, dim), mode='bilinear', align_corners=False)
    img = (img * 255.0 - 127.5) / 128.0
    return img
