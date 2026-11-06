import utils
from utils import *

from generator import *
from discri import *

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch, time, os, statistics
import numpy as np
from generator import Generator
from argparse import ArgumentParser
import torch.nn.functional as F

import sys

sys.path.append('../BiDO')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import model
from common_ckpt import unprot_ckpt  # shared: locate unprotected ckpt + read acc from filename

device = "cuda"


# Map each target label (0-999) to a representative training-image filename, so the SEEDED FE
# schemes derive that image's per-IMAGE codeword from the filename during recovery. Non-seeded
# FE ignore img_id; unprotected T ignores its 2nd arg.
_LABEL2NAME = {}
def _iden_names(iden, train_file="../attack_dataset/CelebA/trainset_jpg.txt"):
    if not _LABEL2NAME:
        with open(train_file) as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    _LABEL2NAME.setdefault(int(p[1]), p[0])
    return [_LABEL2NAME[int(l)] for l in iden.tolist()]


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


def dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=1500, clip_range=1,
                   improved=False, num_seeds=5, verbose=False):
    iden = iden.view(-1).long().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    tf = time.time()

    # NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.zeros(bs, 100), requires_grad=True)

    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)

    for i in range(iter_times):
        z = reparameterize(mu, log_var)
        fake = G(z)
        #* DG was trained on standardized inputs (k+1_gan_vib standardizes both reals and
        #* fakes), so standardize raw G(z) here too before the discriminator.
        if improved == True:
            _, label = D(utils.low2high(fake, dim=64, standardize=True))
        else:
            label = D(utils.low2high(fake, dim=64, standardize=True))

        #* T's classifier head was trained on 112x112 inputs, so resize before feeding into T. (Keep native size for E below)
        t_dim = 112
        if args.prot == "unprotected":
            out = T(utils.low2high(fake, dim=t_dim))
        else:
            out = T(utils.low2high(fake, dim=t_dim), _iden_names(iden))

        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()

        if improved:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            Prior_Loss = - label.mean()

        Iden_Loss = criterion(out, iden)

        Total_Loss = Prior_Loss + lamda * Iden_Loss

        Total_Loss.backward()
        solver.step()

        z = torch.clamp(z.detach(), -clip_range, clip_range).float()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i + 1) % 500 == 0 and verbose:
            fake_img = G(z.detach())

            if args.dataset == 'celeba':
                #* Modify evaluator model output format
                img_dim = 112  # E is the facenet classifier, trained at 112
                eval_prob = E(utils.low2high(fake_img, dim=img_dim))
                # eval_prob = E(utils.low2high(fake_img))[-1]
            else:
                eval_prob = E(fake_img)[-1]

            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
            print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.4f}".format(i + 1, Prior_Loss_val,
                                                                                                Iden_Loss_val, acc))

    if verbose:
        interval = time.time() - tf
        print("Time:{:.2f}".format(interval))

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, num_seeds))

    for random_seed in range(num_seeds):
        tf = time.time()
        z = reparameterize(mu, log_var)
        fake = G(z)

        # save the realized latents for this seed/batch (regen + exact rescore later)
        torch.save({'z': z.detach().cpu(), 'iden': iden.detach().cpu()},
                   os.path.join(args.z_dir, "z_seed{}_iden{:03d}-{:03d}.pt".format(
                       random_seed + 1, iden[0].item(), iden[-1].item())))

        if args.dataset == 'celeba':
            #* Modify evaluator model output format
            img_dim = 112  # E is the facenet classifier, trained at 112
            eval_prob = E(utils.low2high(fake, dim=img_dim))
            # eval_prob = E(utils.low2high(fake))[-1]
        else:
            eval_prob = E(fake)[-1]

        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()
            sample = fake[i]
            save_tensor_images(sample.detach(),
                               os.path.join(args.save_img_dir, "attack_iden_{:03d}|{}.png".format(gt, random_seed + 1)))

            if eval_iden[i].item() == gt:
                seed_acc[i, random_seed] = 1
                cnt += 1
                best_img = G(z)[i]
                save_tensor_images(best_img.detach(), os.path.join(args.success_dir,
                                                                   "attack_iden_{:03d}|{}.png".format(gt,
                                                                                                      random_seed + 1)))
            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1

        interval = time.time() - tf
        if verbose:
            print("Time:{:.2f}\tSeed:{}\tAcc:{:.4f}\t".format(interval, random_seed, cnt * 100.0 / bs))
        res.append(cnt * 100.0 / bs)
        res5.append(cnt5 * 100.0 / bs)

        torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.stdev(res)
    acc_var5 = statistics.stdev(res5)

    if verbose:
        print(f"Acc:{acc:.4f}\tAcc_5:{acc_5:.4f}\tAcc_var:{acc_var:.4f}\tAcc_var5:{acc_var5:.4f}")

    return acc, acc_5, acc_var, acc_var5


if __name__ == "__main__":

    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba')
    parser.add_argument('--defense', default='reg', help='reg')
    parser.add_argument('--iter', default=5000, type=int)
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    parser.add_argument('--verbose', action='store_true', help='') 
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 's') 
    parser.add_argument('--prot', type=str, help="Protection scheme (the FE scheme dir name)", default = 'unprotected')
    parser.add_argument('--embed_model', type=str, help="Embedding Model (facenet)", default = 'facenet')
    args = parser.parse_args()

    z_dim = 100
    ############################# mkdirs ##############################
    log_path = os.path.join(args.save_img_dir, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    #* Change log filename
    log_file = f"attack_{args.dataset}_{args.defense}_{args.embed_model}_{args.prot}_{args.sameR}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'a+')

    #* Make specific dir to save results
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense, args.embed_model)
    save_model_dir = os.path.join(save_model_dir, f'{args.prot}_{args.sameR}')

    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense, args.embed_model)
    save_res_dir = os.path.join(args.save_img_dir, f'{args.prot}_{args.sameR}')
    os.makedirs(save_res_dir, exist_ok=True)

    args.save_img_dir = save_res_dir
    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)

    # Persist the optimized latents (sibling 'z/' dir, like GMI) so G(z) can be regenerated and
    # rescored exactly from the float tensor -- no PNG round-trip, no quantization/normalize noise.
    args.z_dir = os.path.join(save_res_dir, 'z')
    os.makedirs(args.z_dir, exist_ok=True)

    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)

    ############################# mkdirs ##############################
    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    stage = loaded_args["dataset"]["stage"]
    if args.embed_model == 'facenet':
        loaded_args["dataset"]["model_name"] = "FaceNet"
    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        #* Evaluator model E: use the TRAINED unprotected-facenet classifier as the evaluator
        #* (real 1000-class head, so argmax(E(x)) is a valid identity prediction). Locate it by
        #* globbing the checkpoint rather than hardcoding its accuracy tag.
        if args.embed_model == 'facenet':
            E = model.InceptionResnetV1Unprotected(pretrained="vggface2", classify=True, num_classes=1000, device=device)
            E = torch.nn.DataParallel(E).to('cuda')
            path_E, unprot_acc = unprot_ckpt()
            ckp_E = torch.load(path_E)
            E.load_state_dict(ckp_E['state_dict'])
            print("Load evaluation model successfully!")
        else:
            raise ValueError("Invalid embed_model argument")

        if args.defense == 'reg':
            #* Change target model
            num_classes = 1000
            if args.embed_model == 'facenet':
                if args.prot == "unprotected":
                    #* Unprotected embeds (differentiable: backbone + classifier head). Target ==
                    #* evaluator here, so reuse the globbed unprotected-classifier accuracy.
                    ac = unprot_acc
                    T = model.InceptionResnetV1Unprotected(pretrained="vggface2", classify=True, num_classes=num_classes, device=device)
                else:
                    #* All FE/protected schemes go through the unified PipeFE wrapper, selected
                    #* by scheme name: facialfe_e8 | facialfe_leech | facialfe_leech_randomised |
                    #* ourfe. NOTE: the FE step is non-differentiable for ourfe, so Iden_Loss cannot backprop to z for these schemes.
                    ac = args.acc
                    T = model.InceptionResnetV1PipeFE(scheme=args.prot, sameR=(args.sameR == 's'),
                                                      pretrained="vggface2", classify=True,
                                                      num_classes=num_classes, device=device,
                                                      dataset=args.dataset, differentiable=True)
                model_tar = f"{model_name}_{args.defense}_{args.embed_model}_{ac:.2f}_{args.prot}_{args.sameR}.tar"

            print(f"Target model is {model_tar}")
            path_T = os.path.join(args.model_path, args.dataset, args.defense, args.embed_model, model_tar)

            T = torch.nn.DataParallel(T).to('cuda')
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            path_G = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            print("Load G from:", path_G)
            path_D = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)
            print("Load D from:", path_D)

        G = Generator(z_dim)
        G = torch.nn.DataParallel(G).to('cuda')
        D = MinibatchDiscriminator()
        D = torch.nn.DataParallel(D).to('cuda')

        ckp_G = torch.load(path_G)
        G.load_state_dict(ckp_G['state_dict'], strict=False)
        ckp_D = torch.load(path_D)
        D.load_state_dict(ckp_D['state_dict'], strict=False)

        ############         attack     ###########
        aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0

        # Evaluate on all 1000 identities
        ids = 1000
        times = 20
        ids_per_time = ids // times
        iden = torch.from_numpy(np.arange(ids_per_time))
        for idx in range(times):
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            print(f"Attack on ids ids {idx*ids_per_time} to {(idx+1)*ids_per_time}")
            acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100,
                                                iter_times=args.iter, clip_range=1, improved=args.improved_flag,
                                                num_seeds=5, verbose=args.verbose)

            iden = iden + ids_per_time
            aver_acc += acc / times
            aver_acc5 += acc5 / times
            aver_var += var / times
            aver_var5 += var5 / times
            res = aver_acc, aver_acc5, aver_var, aver_var5
            # np.save(os.path.join(args.save_img_dir, args.dataset, args.defense, f"res_batch{idx}.npy"), res)
            np.save(os.path.join(save_res_dir, f"res_batch{idx}.npy"), res)

        res = aver_acc, aver_acc5, aver_var, aver_var5
        np.save(os.path.join(save_res_dir, "res.npy"), res)
                
        print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(
            aver_acc,
            aver_acc5,
            aver_var,
            aver_var5))


