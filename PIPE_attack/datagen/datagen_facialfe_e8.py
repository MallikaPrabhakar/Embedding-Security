# %%
import numpy as np
import os
import glob
import argparse

T_FACENET = 7.25

# %%
def ss_gen(w, id=None):
    c = decode(w)
    s = c-w
    return s, c

def ss_rec(w_, s, id=None):
    c_ = w_ + s
    c = decode(c_)
    w = c - s
    return w

def decode_d8(x):
    # unique decoding of D8
    x_ = np.round(x)
    if np.sum(x_)%2 == 0:
        return x_
    else:
        i = np.argmin(np.abs(x - x_))
        ei = np.zeros_like(x_)
        ei[i] = 1
        # print('here')
        if x_[i] > x[i]:
            return x_ + ei
        else:
            return x_ - ei

def decode_e8(x):
    # E8 = D8 ∪ (D8 + 1/2). For the second coset, decode the shifted point in D8
    # then add 1/2 back so x2 is the actual nearest E8 point (not a bare D8 point).
    x1 = decode_d8(x)
    x2 = decode_d8(x - 0.5) + 0.5
    # print("x1:", x1, "x2:", x2)
    # print(distance(x, x1), distance(x, x2))
    return x1 if distance(x, x1) < distance(x, x2) else x2

def decode2(x):
    # decode e8 m times m = 512//8
    m = len(x)
    val = np.zeros_like(x)
    for _ in range(m//8):
        # print(_)
        xi = decode_e8(x[_ * 8:(_ + 1) * 8])
        val[_ * 8:(_ + 1) * 8] = xi
    return val


def decode(x):
    return decode2(x)

def distance(x, y):
    # we actually need voronoi distance but since it is claimed to be an estimation of L2 distance, we use L2 distance
    return np.linalg.norm(x - y) # l2


# %%
def main():
    # load argvalues
    parser = argparse.ArgumentParser(description='Generate FacialFE-E8 embeddings')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--folders', nargs='+', default=None,
                        help='splits to process (default: train test_seen test test2)')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    read_folder = f'../datasets/unprotected/{dataset}/{model}/'
    ts = [T_FACENET]  # for facenet

    folders = args.folders if args.folders is not None else ['train', 'test_seen', 'test', 'test2']
    for t in ts:
        write_folder = f'../datasets/facialfe_e8/{dataset}/{model}_invscale{t}/'
        for folder in folders:
            write_path = write_folder + folder + '/embeddings/'
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            embedding_list = glob.glob(read_folder + folder + '/embeddings/*.npy')
            print(read_folder + folder + '/embeddings/*.npy')
            for embedding_path in embedding_list:
                w = np.load(embedding_path) /t
                s, c = ss_gen(w)
                np.save(write_path + os.path.basename(embedding_path), s)


if __name__ == '__main__':
    main()
