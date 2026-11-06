import os, sys
import time
import utils
import torch
from utils import *
from torch.autograd import grad
import torch.nn.functional as F
from discri import *
from generator import *
from argparse import ArgumentParser

sys.path.append('../BiDO')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import model
from common_ckpt import unprot_ckpt  # shared: locate unprotected ckpt + read acc from filename

device = "cuda"


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]

    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


if __name__ == "__main__":
        
    parser = ArgumentParser(description='Step1: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba')
    parser.add_argument('--defense', default='reg', help='reg')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 's') 
    parser.add_argument('--prot', type=str, help="Protection scheme (the FE scheme dir name)", default = 'unprotected')
    parser.add_argument('--embed_model', type=str, help="Embedding Model (facenet)", default = 'facenet')
    args = parser.parse_args()

    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)


    ############################# mkdirs ##############################
    #* Make specific dir to save results
    save_res_dir = os.path.join(args.root_path, f'{args.prot}_{args.sameR}_{args.embed_model}')
    os.makedirs(save_res_dir, exist_ok=True)

    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense, args.embed_model)
    save_model_dir = os.path.join(save_model_dir, f'{args.prot}_{args.sameR}')
    os.makedirs(save_model_dir, exist_ok=True)
    # save_img_dir = "./improvedGAN/imgs_improved_{}".format(args.dataset)
    save_img_dir = os.path.join(save_res_dir, "imgs_improved_{}".format(args.dataset))
    os.makedirs(save_img_dir, exist_ok=True)

    #* Make log file
    log_file = os.path.join(save_model_dir, f"train_gan_{args.prot}_{args.sameR}_{args.embed_model}.txt")
    utils.Tee(log_file, 'a+')
    ############################# mkdirs ##############################

    file_path = loaded_args['dataset']['gan_file_path']
    stage = loaded_args['dataset']['stage']
    lr = loaded_args[stage]['lr']
    batch_size = loaded_args[stage]['batch_size']
    z_dim = loaded_args[stage]['z_dim']
    epochs = loaded_args[stage]['epochs']
    n_critic = loaded_args[stage]['n_critic']
    n_classes = loaded_args["dataset"]["n_classes"]

    if args.embed_model == 'facenet':
        loaded_args["dataset"]["model_name"] = "FaceNet"
    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        if args.defense == 'reg':
            #* Modify target model
            num_classes = 1000
            if args.embed_model == 'facenet':
                if args.prot == "unprotected":
                    #* Unprotected embeds (differentiable: backbone + classifier head). Glob the
                    #* unprotected classifier and read its acc tag rather than hardcoding it.
                    _, ac = unprot_ckpt()
                    T = model.InceptionResnetV1Unprotected(pretrained="vggface2", classify=True, num_classes=num_classes, device=device)
                else:
                    #* All FE/protected schemes go through the unified PipeFE wrapper, selected
                    #* by scheme name: facialfe_e8 | facialfe_leech | facialfe_leech_randomised |
                    #* ourfe. NOTE: the FE step is non-differentiable (CPU numpy
                    #* under no_grad). KED-MI uses T only for pseudo-labels (y=argmax(T(imgs))),
                    #* which does not need a gradient, so PipeFE is fine here.
                    ac = args.acc
                    T = model.InceptionResnetV1PipeFE(scheme=args.prot, sameR=(args.sameR == 's'),
                                                      pretrained="vggface2", classify=True,
                                                      num_classes=num_classes, device=device, dataset=args.dataset)
                model_tar = f"{model_name}_{args.defense}_{args.embed_model}_{ac:.2f}_{args.prot}_{args.sameR}.tar"
            
            print(f"Target model is {model_tar}")
            path_T = os.path.join(args.model_path, args.dataset, args.defense, args.embed_model, model_tar)

            T = nn.DataParallel(T).cuda()
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

    print("Saving to Gpath", Gpath)
    print("Saving to Dpath", Dpath)
    print("---------------------Training [%s]------------------------------" % stage)

    if args.dataset == 'celeba':
        G = Generator(z_dim)
        DG = MinibatchDiscriminator()


    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()

    #* Rename to dataloader_init
    _, dataloader_init = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

    step = 0
    for epoch in range(0, epochs):
        start = time.time()

        _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

        for i, (imgs, names, iden) in enumerate(dataloader_init):
            current_iter = epoch * len(dataloader_init) + i + 1
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            #* Ignore identity label
            x_unlabel, _, _ = next(unlabel_loader1)
            x_unlabel2, _, _ = next(unlabel_loader2)

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            #* Modify target model input/output format
            # y_prob = T(imgs)[-1]
            if args.prot == "unprotected":
                y_prob = T(imgs)
            else:
                y_prob = T(imgs, names)  # img_id (image name) drives the seeded FE schemes

            y = torch.argmax(y_prob, dim=1).view(-1)

            #* Downsize to 64 dim for discriminator. Reals (imgs/x_unlabel) come from the
            #* dataloader ALREADY standardized -> standardize=False (resize only). Fakes are
            #* raw G(z) in [0,1] -> standardize=True, so DG sees reals and fakes on one scale.
            _, output_label = DG(utils.low2high(imgs, dim=64, standardize=False))
            _, output_unlabel = DG(utils.low2high(x_unlabel, dim=64, standardize=False))
            _, output_fake = DG(utils.low2high(f_imgs, dim=64, standardize=True))

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                - torch.mean(log_sum_exp(output_unlabel))
                                + torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            acc = torch.mean((output_label.max(1)[1] == y).float())

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(utils.low2high(f_imgs, dim=64, standardize=True))
                #* Downsize to 64 dim for discriminator (x_unlabel2 already standardized)
                mom_unlabel, _ = DG(utils.low2high(x_unlabel2, dim=64, standardize=False))

                mom_gen = torch.mean(mom_gen, dim=0)
                mom_unlabel = torch.mean(mom_unlabel, dim=0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, dg_loss, g_loss,
                                                                                   acc))

        if epoch + 1 >= 100:
            torch.save({'state_dict': G.state_dict()}, Gpath)
            torch.save({'state_dict': DG.state_dict()}, Dpath)

        if (epoch + 1) % 5 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(),
                               os.path.join(save_img_dir,
                                            f"improved_{args.dataset}_img_{args.defense}_{epoch + 1}.png"),
                               nrow=8)
