import sys
import argparse
import json
import os
import pdb
import time
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch import topk
from torch.autograd import Variable
from training.datasets.classifier_dataset import DeepFakeClassifierDataset, collate_function
from torch.nn.modules.loss import BCEWithLogitsLoss
from training.tools.config import load_config
from training.tools.utils import create_optimizer, AverageMeter, read_annotations, Progbar, predict_set, evaluate
from training.transforms.albu import create_train_transforms,create_val_transforms
from training.zoo import classifiers
from apex import amp

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE',default='configs/b7.json',help='path to configuration file')
    arg('--train_txt',type=str,default='/data/lvyb/data/pub_test/train_val_test/dfdc_dfv2_ff++_timit/train.txt')
    arg('--val_txt', type=str,default='/data/lvyb/data/pub_test/train_val_test/dfdc_dfv2_ff++_timit/val.txt')
    arg('--workers', type=int, default=6, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output_dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='classifier_')
    arg('--label_smoothing', type=float, default=0.01)
    arg('--logdir', type=str, default='logs')
    arg('--zero_score', action='store_true', default=False)
    arg('--from_zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg('--freeze_epochs', type=int, default=0)
    arg("--local_rank", default=0, type=int)
    arg("--seed", default=777, type=int)
    arg("--opt_level", default='O1', type=str)
    arg("--test_every", type=int, default=1)
    arg("--no_hardcore", action="store_true")
    arg("--debug",type=int,default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug:
        pdb.set_trace()
    #分布训练和设置gpuid：我选择单卡！
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    conf = load_config(args.config)
    model = classifiers.__dict__[conf['network']](encoder=conf['encoder'])

    model = model.cuda()
    if args.distributed:
        model = convert_syncbn_model(model)
    ohem = conf.get("ohem_samples", None)
    reduction = "mean"
    if ohem:
        reduction = "none"

    loss_function = BCEWithLogitsLoss()
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)
    bce_best = 100
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = DeepFakeClassifierDataset(
        annotations=read_annotations(args.train_txt),
        mode="train",
        balance=True,
        hardcore=True,#not args.no_hardcore,
        label_smoothing=args.label_smoothing,
        transforms=create_train_transforms(conf["size"]),
        normalize=conf.get("normalize", None))

    data_val = DeepFakeClassifierDataset(
        annotations=read_annotations(args.val_txt),
        mode="val",
        balance=False,
        transforms=create_val_transforms(conf["size"]),
        normalize=conf.get("normalize", None))

    val_data_loader = DataLoader(
        data_val,
        batch_size=batch_size * 2,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_function)

    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + '/' + conf.get("prefix", args.prefix) + conf['encoder'] + "_" + str(args.fold))
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    bce_best = checkpoint.get('bce_best', 0)
            print("=> loaded checkpoint '{}' (epoch {}, bce_best {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['bce_best']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale='dynamic')

    snapshot_name = "{}{}_{}_{}".format(conf.get("prefix", args.prefix), conf['network'], conf['encoder'], args.fold)

    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True,find_unused_parameters=True)
    else:
        model = DataParallel(model).cuda()

    data_val.reset_seed(1, args.seed)

    max_epochs = conf['optimizer']['schedule']['epochs']
    for epoch in range(start_epoch, max_epochs):
        data_train.reset_seed(epoch, args.seed)
        train_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
            train_sampler.set_epoch(epoch)
        # 前freeze_epoch轮不参与训练
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            model.module.encoder.eval()
            for p in model.module.encoder.parameters():
                p.requires_grad = False
        else:
            model.module.encoder.train()
            for p in model.module.encoder.parameters():
                p.requires_grad = True

        train_data_loader = DataLoader(
            data_train,
            batch_size=batch_size,
            num_workers=args.workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_function)
        print(len(train_data_loader.dataset))
        train_epoch(current_epoch, loss_function, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank, args.debug)
        model = model.eval()

        if args.local_rank == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, args.output_dir + '/' + snapshot_name + "_last")
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, args.output_dir + snapshot_name + "_{}".format(current_epoch))
            if (epoch + 1) % args.test_every == 0:
                bce_best = validate(args, val_data_loader, bce_best, model,
                                        snapshot_name=snapshot_name,
                                        current_epoch=current_epoch,
                                        summary_writer=summary_writer,
                                        conf=conf)
        current_epoch += 1


def validate(args, data_val, bce_best, model, snapshot_name, current_epoch, summary_writer,conf):
    print("Test phase")
    if args.debug:
        pdb.set_trace()
    model = model.eval()
    probs, gt_labels, names = predict_set([model],data_val,{'run_type':'val','debug':args.debug})
    matrix = evaluate(gt_labels, probs > conf['pos_th'], probs)
    bce = matrix['bce']

    #bce, probs, targets = validate(model, data_loader=data_val)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/bce', float(bce), global_step=current_epoch)
        if bce < bce_best:
            print("Epoch {} improved from {} to {}".format(current_epoch, bce_best, bce))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'bce_best': bce,
                }, args.output_dir + snapshot_name + "_best_dice")
            bce_best = bce
            with open("predictions_{}.json".format(args.fold), "w") as f:
                json.dump({"probs": probs.tolist(), "targets": gt_labels.tolist()}, f)

        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, args.output_dir + snapshot_name + "_last")
        print("Epoch: {} bce: {}, bce_best: {}".format(current_epoch, bce, bce_best))
        print(matrix)
    return bce_best


def train_epoch(current_epoch, loss_function, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank, debug):
    #存储平均值
    progbar = Progbar(len(train_data_loader.dataset), stateful_metrics=['epoch', 'config','lr'])
    batch_time = AverageMeter()
    end = time.time()
    losses = AverageMeter()
    fake_losses = AverageMeter()
    real_losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    print("training epoch {}".format(current_epoch))
    model.train()

    for i, (labels, imgs, img_path) in enumerate(train_data_loader):
        numm = imgs.shape[0]
        optimizer.zero_grad()
        imgs = imgs.reshape((-1,imgs.size(-3),imgs.size(-2), imgs.size(-1)))
        imgs = Variable(imgs, requires_grad=True).cuda()

        labels = labels.reshape(-1)
        labels = labels.cuda().float()
        out_labels = model(imgs)

        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5

        ohem = conf.get("ohem_samples", None)
        if torch.sum(fake_idx * 1) > 0:
            fake_loss = loss_function(out_labels[fake_idx].reshape(-1), labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = loss_function(out_labels[real_idx].reshape(-1), labels[real_idx])
        #挑选出最大的n个计算损失
        if ohem:
            fake_loss = topk(fake_loss, k=min(ohem, fake_loss.size(0)), sorted=False)[0].mean()
            real_loss = topk(real_loss, k=min(ohem, real_loss.size(0)), sorted=False)[0].mean()

        loss = (fake_loss + real_loss) / 2
        losses.update(loss.item(), imgs.size(0))
        fake_losses.update(0 if fake_loss == 0 else fake_loss.item(), imgs.size(0))
        real_losses.update(0 if real_loss == 0 else real_loss.item(), imgs.size(0))
        summary_writer.add_scalar('train/loss', loss.item(), global_step=i + current_epoch * max_iters)
        summary_writer.add_scalar('train/lr',float(scheduler.get_lr()[-1]), global_step=i + current_epoch * max_iters)

        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()

        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if (i == max_iters - 1) or debug:
            break
        progbar.add(numm, values=[('epoch', current_epoch),
                                          ('loss', losses.avg),
                                          ("lr",float(scheduler.get_lr()[-1])),
                                          ("f",fake_losses.avg),
                                          ("r",real_losses.avg)])

    pdb.set_trace()
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        #summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()
