import torch, os, engine
import utils
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy

device = "cuda"
from csv_logger import CSVLogger, plot_csv
from model import InceptionResnetV1Unprotected, InceptionResnetV1PipeFE
import time

def main(args, loaded_args, trainloader, testloader):
    n_classes = loaded_args["dataset"]["n_classes"]
    if args.embed_model == 'facenet':
        model_name = "FaceNet"
    # model_name = loaded_args["dataset"]["model_name"]
    weight_decay = loaded_args[model_name]["weight_decay"]
    momentum = loaded_args[model_name]["momentum"]
    n_epochs = loaded_args[model_name]["epochs"]
    lr = loaded_args[model_name]["lr"]
    milestones = loaded_args[model_name]["adjust_epochs"]

    if args.dataset == 'celeba':
        lr = 1e-2
        n_epochs = 50
            
        if args.embed_model == "facenet":
            #* Target model = facenet backbone (frozen) + classifier head trained on the
            #* (protected) embedding. Unprotected keeps the 512-d head; FE schemes go through
            #* the unified PipeFE wrapper, whose logits head is auto-sized to the FE output.
            if args.prot == 'unprotected':
                net = InceptionResnetV1Unprotected(pretrained="vggface2", classify=True, num_classes=n_classes, device=device)
                T_path = os.path.join("target_model/FaceNet_pytorch_classify_untrained.tar")
            else:
                # facialfe_e8 | facialfe_leech | facialfe_leech_randomised | ourfe
                net = InceptionResnetV1PipeFE(scheme=args.prot, sameR=(args.sameR == 's'),
                                              pretrained="vggface2", classify=True, num_classes=n_classes, device=device)
                T_path = os.path.join(f"target_model/FaceNet_pytorch_classify_{args.prot}_untrained.tar")

        print("Training model ", T_path)
        
        net = torch.nn.DataParallel(net).to(device)
        ckp_E = torch.load(T_path)
        net.load_state_dict(ckp_E['state_dict'], strict=False)
        for p in net.parameters():
            p.requires_grad_(False)
        logits_params = [p for name, p in net.named_parameters() if "logits" in name]
        for p in logits_params:
            p.requires_grad_(True)

        optimizer = torch.optim.SGD(params=logits_params,
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            nesterov=True
                            )
    

    scheduler = MultiStepLR(optimizer, milestones, gamma=0.2)

    criterion = nn.CrossEntropyLoss().cuda()
    # net = torch.nn.DataParallel(net).to(device)

    ################## viz ######################
    args.output_dir = os.path.join(args.model_dir, args.dataset, args.defense, args.embed_model)
    os.makedirs(args.output_dir, exist_ok=True)
    #* Modify logger to log time, under new dir
    epoch_fieldnames = ['global_iteration', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'time']
    epoch_logger = CSVLogger(every=1,
                             fieldnames=epoch_fieldnames,
                             filename=os.path.join(model_path_subdir, 'epoch_log.csv'),
                             resume=0)
    print(f"Logging to {epoch_logger.filename}")
    ################## viz ######################
    best_acc = -1
    tf = time.time()
    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
        
        train_loss, train_acc = engine.train_reg(net, criterion, optimizer, trainloader)
        test_loss, test_acc = engine.test_reg(net, criterion, testloader)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = deepcopy(net)

        scheduler.step()

        ################################### viz ####################################
        if epoch % 1 == 0:
            interval = time.time() - tf
            epoch_logger.writerow({
                'global_iteration': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'time': interval,
            })
            plot_csv(epoch_logger.filename, os.path.join(args.output_dir, f'{args.prot}_{args.sameR}/', 'epoch_plots.jpeg'))
        ################################### viz ####################################

    print("best acc:", best_acc)
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
    }, model_path, "{}_{}_{}_{:.2f}_{}_{}.tar".format(model_name, args.defense, args.embed_model, best_acc, args.prot, args.sameR))


    
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train reg')
    parser.add_argument('--dataset', default='celeba', help='celeba')
    parser.add_argument('--defense', default='reg', help='reg')
    parser.add_argument('--root_path', default='./', help='')
    parser.add_argument('--config_dir', default='./config', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    parser.add_argument('--output_dir', default='./target_model/celeba/reg', help='')
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 'd') 
    parser.add_argument('--prot', type=str, help="Protection scheme (the FE scheme dir name)", default = 'unprotected')
    parser.add_argument('--embed_model', type=str, help="Embedding Model (facenet)", default = 'facenet')

    args = parser.parse_args()
    model_path = os.path.join(args.model_dir, args.dataset, args.defense, args.embed_model)
    os.makedirs(model_path, exist_ok=True)
    #* Make specific subdir to save models
    model_path_subdir = os.path.join(model_path, f'{args.prot}_{args.sameR}')
    os.makedirs(model_path_subdir, exist_ok=True)
    
    print("Saving model to ", model_path)
    print("Saving to ", model_path_subdir)
    file = os.path.join(args.config_dir, args.dataset + ".json")
    loaded_args = utils.load_json(json_file=file)

    train_file = loaded_args['dataset']['train_file']
    test_file = loaded_args['dataset']['test_file']
    testloader = utils.init_dataloader(loaded_args, test_file, mode="test")
    trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
    

    main(args, loaded_args, trainloader, testloader)

