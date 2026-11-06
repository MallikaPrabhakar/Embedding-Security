import torch, os, time, random, generator, discri
import numpy as np
import torch.nn as nn
import statistics
from argparse import ArgumentParser

device = "cuda"

import sys

sys.path.append('../BiDO')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import model, utils
from utils import save_tensor_images
from common_ckpt import unprot_ckpt  # shared: locate unprotected ckpt + read acc from filename

# Map each target label (0-999) to a representative training-image filename, so the SEEDED FE
# schemes (facialfe_leech_randomised, ourfe) derive that image's per-image codeword/keys from the
# filename at attack time. Non-seeded FE ignore img_id; unprotected T ignores its 2nd arg.
_LABEL2NAME = {}
def _iden_names(iden, train_file="../attack_dataset/CelebA/trainset_jpg.txt"):
    if not _LABEL2NAME:
        with open(train_file) as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    _LABEL2NAME.setdefault(int(p[1]), p[0])
    return [_LABEL2NAME[int(l)] for l in iden.tolist()]


def inversion(args, G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500,
              clip_range=1, num_seeds=5, verbose=False):
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)

    res = []
    res5 = []
    for random_seed in range(num_seeds):
        tf = time.time()

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        z = torch.randn(bs, 100).cuda().float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).cuda().float()

        # Train inversion model
        for i in range(iter_times):
            fake = G(z) # [bs, 3, 64, 64], values between [0,1]
            label = D(fake)
            if args.dataset == 'celeba':
                #* T's classifier head was trained on 112x112 inputs, so resize before feeding into T. (Keep native size for E below)
                t_dim = 112
                if args.prot == "unprotected":
                    out = T(utils.low2high(fake, dim=t_dim))
                else:
                    out = T(utils.low2high(fake, dim=t_dim), _iden_names(iden))
            else:
                out = T(fake)[-1]
            

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()
            Iden_Loss = criterion(out, iden)
            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + (- momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if verbose:
                if (i + 1) % 500 == 0:
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
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        acc))

        # Generate recovered images, compute accuracy
        fake = G(z)
        # Persist the optimized latents (tiny) so G(z) can be regenerated + rescored later,
        # exactly, without re-optimizing. Keyed by seed + the id-range of this batch.
        if args.dataset == 'celeba':
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
                               os.path.join(args.save_img_dir,
                                            "attack_iden_{:03d}|{}.png".format(gt, random_seed + 1)))

            if eval_iden[i].item() == gt:
                cnt += 1
                flag[i] = 1
                best_img = G(z)[i]
                save_tensor_images(best_img.detach(),
                                   os.path.join(args.success_dir,
                                                "attack_iden_{:03d}|{}.png".format(gt, random_seed + 1)))

            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1

        res.append(cnt * 100.0 / bs)
        res5.append(cnt5 * 100.0 / bs)
        torch.cuda.empty_cache()
        interval = time.time() - tf
        print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 100.0 / bs))

    acc = statistics.mean(res)
    acc_5 = statistics.mean(res5)
    acc_var = statistics.stdev(res)
    acc_var5 = statistics.stdev(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tacc_var5{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5


if __name__ == '__main__':
        
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba')
    parser.add_argument('--defense', default='reg', help='reg')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--iter', default=3000, type=int)
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 's') 
    parser.add_argument('--prot', type=str, help="Protection scheme (the FE scheme dir name)", default = 'unprotected')
    parser.add_argument('--embed_model', type=str, help="Embedding Model (facenet)", default = 'facenet')
    parser.add_argument('--acc', type=float, default=0.0, help="classifier acc tag for the model_tar filename (FE schemes; unprotected uses 79.52)")

    args = parser.parse_args()

    ############################# mkdirs ##############################
    #* Add logging
    log_path = os.path.join(args.save_img_dir, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    #* Change log filename
    log_file = f"attack_{args.dataset}_{args.defense}_{args.embed_model}_{args.prot}_{args.sameR}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'a+')

    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense, args.embed_model)
    #* Make specific dir to save results
    save_res_dir = os.path.join(args.save_img_dir, f'{args.prot}_{args.sameR}')
    os.makedirs(save_res_dir, exist_ok=True)
    args.save_img_dir = save_res_dir

    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)
    args.z_dir = os.path.join(args.save_img_dir, 'z')   # optimized latents (for regen + rescore)
    os.makedirs(args.z_dir, exist_ok=True)
    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)

    ############################# mkdirs ##############################

    if args.dataset == 'celeba':
        #* Change target model name
        # model_name = "VGG16"
        if args.embed_model == 'facenet':
            model_name = "FaceNet"
        
        num_classes = 1000

        #* Evaluator model E: use the TRAINED unprotected-facenet classifier (79.52) as the
        #* evaluator. It has a real 1000-class head, so argmax(E(x)) is a valid identity
        #* prediction 
        if args.embed_model == 'facenet':
            E = model.InceptionResnetV1Unprotected(pretrained="vggface2", classify=True, num_classes=num_classes, device=device)
            E = nn.DataParallel(E).cuda()
            e_path, unprot_acc = unprot_ckpt()
            ckp_E = torch.load(e_path)
            E.load_state_dict(ckp_E['state_dict'])
            print("Load evaluation model successfully!")
        #     print("Load evaluation model successfully!")
        else:
            raise ValueError("Invalid embed_model argument")

        g_path = "./GAN_models/celeba/celeba_G_300.tar"
        G = generator.Generator()
        G = nn.DataParallel(G).cuda()
        ckp_G = torch.load(g_path)
        G.load_state_dict(ckp_G['state_dict'], strict=False)
        print("Load GAN model G successfully!")

        d_path = "./GAN_models/celeba/celeba_D_300.tar"
        D = discri.DGWGAN()
        D = nn.DataParallel(D).cuda()
        ckp_D = torch.load(d_path)
        D.load_state_dict(ckp_D['state_dict'], strict=False)
        print("Load GAN model D successfully!")

        
        if args.defense == 'reg':
            #* Change target model
            if args.embed_model == 'facenet':
                if args.prot == "unprotected":
                    #* Unprotected embeds (differentiable: backbone + classifier head).
                    #* Reuse the evaluator's unprotected checkpoint accuracy (globbed above).
                    ac = unprot_acc
                    T = model.InceptionResnetV1Unprotected(pretrained="vggface2", classify=True, num_classes=num_classes, device=device)
                else:
                    #* All FE/protected schemes go through the unified PipeFE wrapper, selected
                    #* by scheme name: facialfe_e8 | facialfe_leech | facialfe_leech_randomised |
                    #* ourfe. differentiable=True uses the straight-through torch FE
                    #* (pipe_fe_torch) so Iden_Loss can backprop to z (forward stays bit-exact).
                    ac = args.acc
                    T = model.InceptionResnetV1PipeFE(scheme=args.prot, sameR=(args.sameR == 's'),
                                                        pretrained="vggface2", classify=True,
                                                        num_classes=num_classes, device=device,
                                                        dataset=args.dataset, differentiable=True)
                model_tar = f"{model_name}_{args.defense}_{args.embed_model}_{ac:.2f}_{args.prot}_{args.sameR}.tar"

            
            print(f"Target model is {model_tar}")
            path_T = os.path.join(args.model_path, args.dataset, args.defense, args.embed_model, model_tar)
            
            T = nn.DataParallel(T).cuda()
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            #* Evaluate on all 1000 identities
            res_all = []
            ids = 1000
            times = 20
            ids_per_time = ids // times
            iden = torch.from_numpy(np.arange(ids_per_time))
            for idx in range(times):
                print("--------------------- Attack batch [%s]------------------------------" % idx)
                print(f"Attack on ids ids {idx*ids_per_time} to {(idx+1)*ids_per_time}")
                res = inversion(args, G, D, T, E, iden, lr=2e-2, iter_times=2000, verbose=True)
                res_all.append(res)
                iden = iden + ids_per_time
                np.save(os.path.join(save_res_dir, f"res_all_batch{idx}.npy"), res_all)

            res = np.array(res_all).mean(0)
            np.save(os.path.join(save_res_dir, "res_all.npy"), res_all)
            np.save(os.path.join(save_res_dir, "res.npy"), res)
        
            print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
            

