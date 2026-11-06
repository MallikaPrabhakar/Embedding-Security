import torch, time
from util import AverageMeter, accuracy
from tqdm import tqdm

device = "cuda"


def train_reg(model, criterion, optimizer, trainloader):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, names, iden) in pbar:
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        #* Only output (bs, 1000) sized out_digit
        #* Give indens as input as well
        out_digit = model(inputs, names)  # 2nd arg = img_id (image name) for the FE seed; ignored by unprotected
        cross_loss = criterion(out_digit, iden)
        # triplet_loss = triplet(feats, iden)
        loss = cross_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        bs = inputs.size(0)
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item(), bs)

        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return losses.avg, top1.avg

def test_reg(model, criterion, testloader):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)

    with torch.no_grad():
        for batch_idx, (inputs, names, iden) in pbar:
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)
            #* Only output (bs, 1000) sized out_digit
            #* Give indens as input as well
            out_digit = model(inputs, names)  # 2nd arg = img_id (image name) for the FE seed; ignored by unprotected
            cross_loss = criterion(out_digit, iden)

            loss = cross_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

    return losses.avg, top1.avg

