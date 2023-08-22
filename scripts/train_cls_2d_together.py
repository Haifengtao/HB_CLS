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

    weight = torch.tensor([1 / 0.2 / 18.6, 1 / 0.44 / 18.6, 1 / 0.184 / 18.6, 1 / 0.17 / 18.6]).cuda()
    weight1 = torch.tensor([1 - 1 / 0.2 / 18.6, 1 / 0.2 / 18.6]).cuda()
    weight2 = torch.tensor([1 - 1 / 0.44 / 18.6, 1 / 0.44 / 18.6]).cuda()
    weight3 = torch.tensor([1 - 1 / 0.184 / 18.6, 1 / 0.184 / 18.6]).cuda()
    weight4 = torch.tensor([1 - 1 / 0.17 / 18.6, 1 / 0.17 / 18.6]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction="mean").cuda()
    criterion1 = torch.nn.CrossEntropyLoss(weight=weight1, reduction="mean").cuda()
    criterion2 = torch.nn.CrossEntropyLoss(weight=weight2, reduction="mean").cuda()
    criterion3 = torch.nn.CrossEntropyLoss(weight=weight3, reduction="mean").cuda()
    criterion4 = torch.nn.CrossEntropyLoss(weight=weight4, reduction="mean").cuda()

    ec_seg = nn.CrossEntropyLoss(weight=torch.tensor([0.01, 0.99])).cuda()
    dice_loss = Losses.DiceLoss().cuda()
    timer, sum_loss, loss_1, loss_2, loss_3 = logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter(), \
                                              logger.AverageMeter(), logger.AverageMeter(),
    cls_loss_1, cls_loss_2, cls_loss_3, cls_loss_4 = logger.AverageMeter(), logger.AverageMeter(), \
                                                     logger.AverageMeter(), logger.AverageMeter()

    generater = iter(mixUp_dataset)
    for i, data in enumerate(labeled_train_loader):
        st = time.time()
        # if i < 580:
        #     continue
        if cfg.train.multi_task:
            x, mask, label, label_1, label_2, label_3, label_4 = data
            lam = torch.tensor(np.random.beta(cfg.train.alpha, cfg.train.alpha))
            x2, mask2, label2, label2_1, label2_2, label2_3, label2_4 = next(generater)

            if label.size()[0] != label2.size()[0]:
                break
            if cfg.net.seg_class == 2:
                label[label != 0] = 1
                label2[label2 != 0] = 1
            x = lam * x + (1 - lam) * x2
            if use_cuda:
                x = x.cuda()
                label, label_1, label_2, label_3, label_4 = label.cuda(), label_1.cuda(), label_2.cuda(), \
                                                            label_3.cuda(), label_4.cuda()
                label2, label2_1, label2_2, label2_3, label2_4 = label2.cuda(), label2_1.cuda(), label2_2.cuda(), \
                                                                 label2_3.cuda(), label2_4.cuda()
                mask, mask2 = mask.cuda(), mask2.cuda()
            seg_pred, pred, pred1, pred2, pred3, pred4 = model(x)
            # import pdb
            # pdb.set_trace()
            loss_value1 = lam * criterion(pred, label) + (1 - lam) * criterion(pred, label2)
            # import pdb
            # pdb.set_trace()
            cls2_loss_1 = lam * criterion1(pred1, label_1) + (1 - lam) * criterion1(pred1, label2_1)
            cls2_loss_2 = lam * criterion2(pred2, label_2) + (1 - lam) * criterion2(pred2, label2_2)
            cls2_loss_3 = lam * criterion3(pred3, label_3) + (1 - lam) * criterion3(pred3, label2_3)
            cls2_loss_4 = lam * criterion4(pred4, label_4) + (1 - lam) * criterion4(pred4, label2_4)

            loss_value2 = lam * ec_seg(seg_pred, mask) + (1 - lam) * ec_seg(seg_pred, mask2)
            loss_value3 = lam * dice_loss(seg_pred, mask) + (1 - lam) * dice_loss(seg_pred, mask2)

            loss_value = loss_value1 + cls2_loss_1 + cls2_loss_1 + cls2_loss_1 + cls2_loss_1 \
                         + 0.1 * loss_value2 + 0.1 * loss_value3

            # import pdb
            # pdb.set_trace()
            loss_1.update(loss_value1.item())
            loss_2.update(loss_value2.item())
            loss_3.update(loss_value3.item())

            cls_loss_1.update(cls2_loss_1.item())
            cls_loss_2.update(cls2_loss_2.item())
            cls_loss_3.update(cls2_loss_3.item())
            cls_loss_4.update(cls2_loss_4.item())

            sum_loss.update(loss_value.item())
            msg = "epoch: {},  iter:{} / all:{},  total: {:.4f}, ec_loss_cls: {:.4f}, " \
                  "cls_loss_1: {:.4f}, cls_loss_2: {:.4f}, cls_loss_3: {:.4f}, cls_loss_4: {:.4f}," \
                  "ec_loss_seg: {:.4f}, " \
                  "dice_loss_seg: {:.4f}".format(epoch, i, len(labeled_train_loader), sum_loss.avg,
                                                 loss_1.avg, cls_loss_1.avg, cls_loss_2.avg, cls_loss_3.avg,
                                                 cls_loss_4.avg, loss_2.avg, loss_3.avg)
            print(msg)
        else:
            raise Exception("Error!")

        try:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        except RuntimeError:
            print("==>RuntimeError")
            continue
        timer.update(time.time() - st)

    return sum_loss.avg, loss_1.avg, cls_loss_1.avg, cls_loss_2.avg, cls_loss_3.avg, cls_loss_4.avg, \
           loss_2.avg, loss_3.avg


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
    preds, pred1s, pred2s, pred3s, pred4s = [], [], [], [], []
    y_true, y_true1, y_true2, y_true3, y_true4 = [], [], [], [], []
    dice_loss = Losses.DiceLoss().cuda()
    criterion = cfg.loss.losses
    loss_ec_cls = logger.AverageMeter()
    loss_1, loss_2, loss_3, loss_4 = logger.AverageMeter(), logger.AverageMeter(), \
                                     logger.AverageMeter(), logger.AverageMeter()
    for idx, data in enumerate(val_dataset):
        # if idx>21:
        #     break
        img, tumor, label, label_1, label_2, label_3, label_4 = data
        if use_cuda:
            img = img.cuda()
            tumor = tumor.cuda()
            label, label_1, label_2, label_3, label_4 = label.cuda(), label_1.cuda(), label_2.cuda(), \
                                                        label_3.cuda(), label_4.cuda()
        seg, pred_label, pred1, pred2, pred3, pred4 = model(img)
        #
        # import pdb
        # pdb.set_trace()
        dice = dice_loss(seg, tumor)
        ec_loss = criterion[0](pred_label, label)
        ec_loss1 = criterion[1](pred1, label_1)
        ec_loss2 = criterion[2](pred2, label_2)
        ec_loss3 = criterion[3](pred3, label_3)
        ec_loss4 = criterion[4](pred4, label_4)

        dices.update(dice.item())
        loss_ec_cls.update(ec_loss.item())
        loss_1.update(ec_loss1.item())
        loss_2.update(ec_loss2.item())
        loss_3.update(ec_loss3.item())
        loss_4.update(ec_loss4.item())

        pred_seg = torch.max(seg, dim=1)[1].detach().cpu().numpy()
        tumor = tumor.detach().cpu().numpy()
        if idx % 50 == 0:
            fig = file_io.draw_fig_2d(img.detach().cpu().numpy(), pred_seg, tumor)
            writer.add_figure("val/fig" + str(idx), fig, epoch)
        # except Exception:
        #     print(Exception)
        # import pdb
        # pdb.set_trace()

        preds.append(torch.max(pred_label, dim=1)[1].item())
        pred1s.append(torch.max(pred1, dim=1)[1].item())
        pred2s.append(torch.max(pred2, dim=1)[1].item())
        pred3s.append(torch.max(pred3, dim=1)[1].item())
        pred4s.append(torch.max(pred4, dim=1)[1].item())
        y_true.append(label.item())
        y_true1.append(label_1.item())
        y_true2.append(label_2.item())
        y_true3.append(label_3.item())
        y_true4.append(label_4.item())
    print("==>VAL: loss_ec_cls:{},  dice: {}, acc: {},"
          " acc1: {}, acc2: {}, acc3: {}, acc4: {}".format(loss_ec_cls.avg, dices.avg, acc(preds, y_true),
                                                           acc(pred1s, y_true1), acc(pred2s, y_true2),
                                                           acc(pred3s, y_true3), acc(pred4s, y_true4)))
    return loss_ec_cls.avg, dices.avg, acc(preds, y_true), acc(pred1s, y_true1), \
           acc(pred2s, y_true2), acc(pred3s, y_true3), acc(pred4s, y_true4)


def main(cfg):
    if cfg.net.name == "Unet_cls":
        model = unet.UnetPlusPlus_Cls_V2(encoder_name="resnet18", in_channels=1, classes=2, aux_params=dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.2,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=2,  # define number of output labels
        ))
        labeled_f = h5py.File(cfg.data.train_txt, 'r')
        labeled_f_val = h5py.File(cfg.data.val_txt, 'r')

        # 标签是四个类别，这里只做二分类
        val_labels = labeled_f_val['label'][:]
        train_labels = labeled_f['label'][:]

        train_cls_dataset = data_generator.cls_generator_2D_4cls(labeled_f['img'],
                                                                 labeled_f['mask'], train_labels,
                                                                 rotate_degree=cfg.data.random_rotate,
                                                                 noise_sigma=cfg.data.random_Noise, )
        train_dataLoader = DataLoader(train_cls_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=16)
        mixUp_dataset = DataLoader(train_cls_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=16)

        val_cls_dataset = data_generator.cls_generator_2D_4cls(labeled_f_val['img'],
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
        loss_sum, loss1, cls_loss_1, cls_loss_2, cls_loss_3, cls_loss_4, loss2, loss3 = train(cfg, epoch, model,
                                                                                              optimizer,
                                                                                              train_dataLoader,
                                                                                              mixUp_dataset, use_cuda)
        writer.add_scalar("train/Sum_Loss", loss_sum, epoch)
        writer.add_scalar("train/EC_loss_CLS", loss1, epoch)
        writer.add_scalar("train/EC_loss_CLS_1", cls_loss_1, epoch)
        writer.add_scalar("train/EC_loss_CLS_2", cls_loss_2, epoch)
        writer.add_scalar("train/EC_loss_CLS_3", cls_loss_3, epoch)
        writer.add_scalar("train/EC_loss_CLS_4", cls_loss_4, epoch)
        writer.add_scalar("train/EC_loss_Seg", loss2, epoch)
        writer.add_scalar("train/Dice_loss_Seg", loss3, epoch)
        with torch.no_grad():
            print("==> Validating! ")
            ec_loss, val_dice, acc_value, acc1, acc2, acc3, acc4 = val_model(cfg, writer, epoch, model, val_dataLoader,
                                                                             use_cuda)
            writer.add_scalar("val/ec_cls", ec_loss, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            writer.add_scalar("val/acc_value", acc_value, epoch)
            writer.add_scalar("val/acc_value1", acc1, epoch)
            writer.add_scalar("val/acc_value2", acc2, epoch)
            writer.add_scalar("val/acc_value3", acc3, epoch)
            writer.add_scalar("val/acc_value4", acc4, epoch)
        scheduler.step()
        print("learning rate", optimizer.param_groups[0]['lr'])
        if epoch % 5 == 0:
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
