import numpy as np
import torch, random, sys, json, time, copy, os
import dataloader
from torch.utils.data import sampler
from collections import defaultdict

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

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def init_dataloader(args, file_path, mode="gan"):
    tf = time.time()
    model_name = args['dataset']['model_name']
    bs = args[model_name]['batch_size']
    # normalized variant: resize + standardize to ~[-1,1] (preserves old dataloader_normalize.py)
    data_set = dataloader.ImageFolder(args, file_path, mode, standardize=True, resize=True)

    if mode == "train":
        sampler = RandomIdentitySampler(data_set, args[model_name]['batch_size'], args['dataset']['instance'])
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  sampler=sampler,
                                                  batch_size=bs,
                                                  num_workers=args['dataset']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  shuffle=False,
                                                  batch_size=bs,
                                                  num_workers=args['dataset']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=True)

    interval = time.time() - tf
    print('Initializing data loader took {:.2f}'.format(interval))
    return data_loader


class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def save_checkpoint(state, directory, filename):
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
