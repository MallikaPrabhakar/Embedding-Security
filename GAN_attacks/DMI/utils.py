import sys
import json, time, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
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

def init_dataloader(args, file_path, batch_size=64, mode="gan", iterator=False):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    # resize + standardize to ~[-1,1] (preserves old DMI/dataloader.py behaviour)
    data_set = dataloader.ImageFolder(args, file_path, mode, standardize=True, resize=True)

    if iterator:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_flag,
                                                  drop_last=True,
                                                  num_workers=2,
                                                  pin_memory=True).__iter__()
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_flag,
                                                  drop_last=True,
                                                  num_workers=2,
                                                  pin_memory=True)

    return data_set, data_loader

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


# normalize=False: G is sigmoid (already in [0,1]), so save the raw output as-is. normalize=True
# would per-image min-max contrast-stretch it, making the saved PNG diverge from the float tensor
# the attack scored (the ~1% offline-vs-log ASR gap); the only residual is 8-bit PNG rounding.
def save_tensor_images(images, filename, nrow=None, normalize=False):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)


def low2high(img, dim=160, standardize=True):
    """Resize an image to `dim` and (optionally) standardize to ~[-1,1].

    We use F.interpolate instead of transforms.ToPILImage() to remain differentiable.
    There should be no detaching here.

    standardize=True applies (x*255-127.5)/128, matching the preprocessing the target/eval
    models were trained with (BiDO dataloader; facenet expects the same range). Use this for
    RAW G(z) (sigmoid output in [0,1]) before feeding T/E, or to standardize raw fakes before
    the discriminator.

    standardize=False only resizes. Use this for inputs that are ALREADY standardized
    (e.g. images straight from the dataloader, which already applies (x*255-127.5)/128) to
    avoid double standardization."""
    img = F.interpolate(img, size=(dim, dim), mode='bilinear', align_corners=False)
    if standardize:
        img = (img * 255.0 - 127.5) / 128.0
    return img


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


# define "soft" cross-entropy with pytorch tensor operations
def softXEnt(input, target):
    targetprobs = nn.functional.softmax(target, dim=1)
    logprobs = nn.functional.log_softmax(input, dim=1)

    return -(targetprobs * logprobs).sum() / input.shape[0]


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
