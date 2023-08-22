#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/17 15:52   Bot Zhao      1.0         None
"""

# import lib
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/2 19:14   Bot Zhao      1.0         None
"""

# import lib
from easydict import EasyDict as edict
import torch
import os

__C = edict()
cfg = __C

__C.general = {}
__C.general.log_path = "./cpts/cls/resnest101_2d_cls_seg_fold4/log"
__C.general.model_path = "./cpts/cls/resnest101_2d_cls_seg_fold4/model"


__C.data = {}
__C.data.train_txt = r"../../dataset/tumor_cls/cross_val_h5_dataset/fold4_2d_mix_train.h5"
__C.data.val_txt = r"../../dataset/tumor_cls/cross_val_h5_dataset/fold4_2d_mix_test.h5"
__C.data.random_rotate = 30
__C.data.random_Noise = (0.001, 0.005)
# __C.data.test_txt = r"./data/version_mix/fold1/test.txt"

__C.net = {}
__C.net.name = "Unet_cls"
__C.net.inchannels = 1
__C.net.seg_class = 2
__C.net.cls_class = 2
__C.net.encoder_block = "resnext101_32x16d"


__C.train = {}
__C.train.load_model = "./cpts/cls/resnest101_2d_cls_seg_fold4/model/75_model.pth"
__C.train.start_epoch = 0
__C.train.epochs = 130
__C.train.batch_size = 16
__C.train.save_epoch = 10
__C.train.learning_rate = 0.0001
__C.train.warmRestart = 1
__C.train.mixup = True
__C.train.aug_data = True
__C.train.multi_task = True
__C.train.alpha = 0.5
__C.train.gpu = '1'


__C.loss = {}

os.environ['CUDA_VISIBLE_DEVICES'] = __C.train.gpu
weight = torch.tensor([0.62, 0.38]).cuda()
__C.loss.losses = [torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")]

__C.logger = {}
__C.logger.title = []
__C.logger.save_path = __C.general.log_path
__C.logger.save_name = "train_log_cls.csv"
__C.logger.file_name = []

# __C.infer = {}
# __C.infer.data_log="/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_summary_v1.csv"
# __C.infer.save_suffix = "_pred_mask_baseling.nii.gz"

# __C.test = {}
# __C.test.model = "/share/inspurStorage/home1/zhaobt/PycharmProjects/tumor_cls/cpts/cls/multi_task_mixup_2_fold1/model_new" \
#                  "/1000_model.pth"
# __C.test.save_dir = "./cpts/cls/multi_task_mixup_2_fold1/res_model_new.json"