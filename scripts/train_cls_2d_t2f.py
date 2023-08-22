#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_cls.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/2 19:16   Bot Zhao      1.0         None
"""

# import lib
import time
# from progress.bar import Bar as Bar
import torch, h5py
from torch import optim, nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, Dataset
import pdb
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/zhang_istbi/data_disk/zhaobt/projects/tumor_cls/")
from nets import Losses
from utils import data_generator
from nets import unet, resnet
from utils import file_io
from utils import logger
from utils import model_io


def train(cfg, epoch, model, optimizer, labeled_train_loader, mixUp_dataset, use_cuda):
    global loss_value, losses
    model.train()
    criterion = cfg.loss.losses[0]
    ec_seg = nn.CrossEntropyLoss(weight=torch.tensor([0.01, 0.99])).cuda()
    dice_loss = Losses.DiceLoss().cuda()
    timer, sum_loss, loss_1, loss_2, loss_3 = logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter(), \
                                              logger.AverageMeter(), logger.AverageMeter(),
    generater = iter(mixUp_dataset)
    for i, data in enumerate(labeled_train_loader):
        st = time.time()
        # if i < 580:
        #     continue
        if cfg.train.multi_task:
            x, mask, label = data
            lam = np.random.beta(cfg.train.alpha, cfg.train.alpha)
            x2, mask2, label2 = next(generater)
            if label.size()[0] != label2.size()[0]:
                break
            if cfg.net.seg_class == 2:
                label[label != 0] = 1
                label2[label2 != 0] = 1
            x = lam * x + (1 - lam) * x2
            if use_cuda:
                x = x.cuda()
                label, label2 = label.cuda(), label2.cuda()
                mask, mask2 = mask.cuda(), mask2.cuda()
            seg_pred, pred = model(x)
            # import pdb
            # pdb.set_trace()
            loss_value1 = lam * criterion(pred, label) + (1 - lam) * criterion(pred, label2)
            loss_value2 = lam * ec_seg(seg_pred, mask) + (1 - lam) * ec_seg(seg_pred, mask2)
            loss_value3 = lam * dice_loss(seg_pred, mask) + (1 - lam) * dice_loss(seg_pred, mask2)

            loss_value = loss_value1 + 0.1 * loss_value2 + 0.1 * loss_value3
            losses = [loss_value1.item(), loss_value2.item(), loss_value3.item()]
            # import pdb
            # pdb.set_trace()
            loss_1.update(loss_value1.item())
            loss_2.update(loss_value2.item())
            loss_3.update(loss_value3.item())
            sum_loss.update(loss_value.item())
            msg = "epoch: {},  iter:{} / all:{},  total: {:.4f}, ec_loss_cls: {:.4f}, ec_loss_seg: {:.4f}, " \
                  "dice_loss_seg: {:.4f}".format(epoch, i, len(labeled_train_loader), sum_loss.avg,
                                                 loss_1.avg, loss_2.avg, loss_3.avg)
            print(msg)
        else:
            raise Exception("Error!")

        # if i>100:
        #     break
        try:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            loss_value.backward()
            # pdb.set_trace()
            optimizer.step()
            # pdb.set_trace()
        except RuntimeError:
            print("==>RuntimeError")
            continue
        timer.update(time.time() - st)

    return sum_loss.avg, loss_1.avg, loss_2.avg, loss_3.avg,


def acc(pred, y_true):
    res = 0
    for p, t in zip(pred, y_true):
        if p == t:
            res += 1
    return res / len(pred)


def val_model(cfg, writer, epoch, model, val_dataset, use_cuda):
    # implemented
    model.eval()
    if use_cuda:
        model.cuda()
    dices = logger.AverageMeter()
    pred = []
    y_true = []
    dice_loss = Losses.DiceLoss().cuda()
    criterion = cfg.loss.losses[0].cuda()
    loss_ec_cls = logger.AverageMeter()
    for idx, data in enumerate(val_dataset):
        # if idx>21:
        #     break
        img, tumor, label = data
        if use_cuda:
            img = img.cuda()
            tumor = tumor.cuda()
            label = label.cuda()
        seg, pred_label = model(img)
        #
        # import pdb
        # pdb.set_trace()
        dice = dice_loss(seg, tumor)
        ec_loss = criterion(pred_label, label)

        dices.update(dice.item())
        loss_ec_cls.update(ec_loss.item())

        # pred_seg = torch.max(seg, dim=1)[1].detach().cpu().numpy()
        # tumor = tumor.detach().cpu().numpy()
        # if idx % 50 == 0:
        #     fig = file_io.draw_fig_2d(img.detach().cpu().numpy(), pred_seg, tumor)
        #     writer.add_figure("val/fig" + str(idx), fig, epoch)
        # except Exception:
        #     print(Exception)
        # import pdb
        # pdb.set_trace()

        pred.append(torch.max(pred_label, dim=1)[1].item())

        y_true.append(label.item())
    print("==>VAL: loss_ec_cls:{},  dice: {}, acc: {}".format(loss_ec_cls.avg, dices.avg, acc(pred, y_true)))
    return loss_ec_cls.avg, dices.avg, acc(pred, y_true)


def main(cfg):
    if cfg.net.name == "Unet_cls":
        model = unet.UnetPlusPlus_Cls_v2(cfg.net.encoder_block, cfg.net.seg_class, cfg.net.cls_class, )
        labeled_f = h5py.File(cfg.data.train_txt, 'r')
        labeled_f_val = h5py.File(cfg.data.val_txt, 'r')

        # 标签是四个类别，这里只做二分类
        try:
            if cfg.net.cls_class != 2:
                # import pdb
                # pdb.set_trace()
                val_labels = labeled_f_val['label'][:]
                train_labels = labeled_f['label'][:]
            else:
                temp = np.zeros_like(labeled_f['label'][:])
                temp[labeled_f['label'][:] == cfg.net.cls_label] = 1
                train_labels = temp.copy()
                temp = np.zeros_like(labeled_f_val['label'][:])
                temp[labeled_f_val['label'][:] == cfg.net.cls_label] = 1
                val_labels = temp.copy()
            print("==>setting target label {}".format(cfg.net.cls_label))
        except:
            train_labels = labeled_f['label'][:]
            train_labels[train_labels != 1] = 0
            val_labels = labeled_f_val['label'][:]
            val_labels[val_labels != 1] = 0
            pass
        train_cls_dataset = data_generator.cls_generator_2D_2mod(labeled_f['t2f'],
                                                                 labeled_f['t1p'],
                                                                 labeled_f['mask'], train_labels,
                                                                 rotate_degree=cfg.data.random_rotate,
                                                                 noise_sigma=cfg.data.random_Noise, )
        train_dataLoader = DataLoader(train_cls_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=16)
        mixUp_dataset = DataLoader(train_cls_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=16)

        val_cls_dataset = data_generator.cls_generator_2D_2mod(labeled_f_val['t2f'],
                                                               labeled_f_val['t1p'],
                                                               labeled_f_val['mask'], val_labels,
                                                               rotate_degree=0,
                                                               noise_sigma=(0, 1e-9), )
        val_dataLoader = DataLoader(val_cls_dataset, batch_size=1, shuffle=False)
    else:
        raise Exception("We have not implemented this model %s".format(cfg.net.name))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu
    # train_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.train_txt), batch_size=cfg.train.batch_size,
    #                                shuffle=True)
    # val_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.val_txt), batch_size=1, shuffle=False)
    if not os.path.isdir(cfg.general.model_path):
        os.makedirs(cfg.general.model_path)
    if not os.path.isdir(cfg.general.log_path):
        os.makedirs(cfg.general.log_path)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0, )
    if torch.cuda.is_available():
        use_cuda = True
        model = model.cuda()
    else:
        use_cuda = False

    if cfg.train.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7)
    elif cfg.train.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)

    if cfg.train.load_model is not None:
        start_epoch = model_io.reload_ckpt(cfg.train.load_model, model, optimizer, scheduler=scheduler,
                                           use_cuda=use_cuda)
        cfg.train.start_epoch = start_epoch
    writer = SummaryWriter(cfg.general.log_path)
    temp = np.inf
    for epoch in range(cfg.train.start_epoch + 1, cfg.train.epochs + 1):
        loss_sum, loss1, loss2, loss3 = train(cfg, epoch, model, optimizer, train_dataLoader, mixUp_dataset, use_cuda)
        writer.add_scalar("train/Sum_Loss", loss_sum, epoch)
        writer.add_scalar("train/EC_loss_CLS", loss1, epoch)
        writer.add_scalar("train/EC_loss_Seg", loss2, epoch)
        writer.add_scalar("train/Dice_loss_Seg", loss3, epoch)
        with torch.no_grad():
            print("==> Validating! ")
            ec_loss, val_dice, acc_value = val_model(cfg, writer, epoch, model, val_dataLoader, use_cuda)
            writer.add_scalar("val/ec_cls", ec_loss, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            writer.add_scalar("val/acc_value", acc_value, epoch)
        scheduler.step()
        print("learning rate", optimizer.param_groups[0]['lr'])
        if epoch % 20 == 0:
            if not os.path.isdir(cfg.general.model_path):
                os.makedirs(cfg.general.model_path)
            torch.save(dict(epoch=epoch,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict()),
                       f=os.path.join(cfg.general.model_path, str(epoch) + "_model.pth"))

        if val_dice < temp:
            torch.save(dict(epoch=epoch,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict()),
                       f=os.path.join(cfg.general.model_path, "best_model.pth"))
            temp = val_dice

    torch.save(dict(epoch=cfg.train.epochs + 1,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict()),
               f=os.path.join(cfg.general.model_path, "final_model.pth"))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Brain Training')
    parser.add_argument('-i', '--config', default=None,
                        help='model config (default: Unet)')
    arguments = parser.parse_args()
    config = file_io.load_module_from_disk(arguments.config)
    cfg = config.cfg
    print(cfg.data.train_txt)
    main(cfg)
