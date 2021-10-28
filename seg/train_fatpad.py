# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import pandas as pd
import denseUnet

from read_fatpad import zongimage_train, zongimage_validate, zongimage_test

arch_names = list(denseUnet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Dense_Unet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="None",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, model, criterion, optimizer, epoch, train_image, train_label, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    train_num = len(train_image) // args.batch_size
    train_image_epoch = np.zeros(shape=(args.batch_size, 1, 384, 384))
    train_label_epoch = np.zeros(shape=(args.batch_size, 1, 384, 384))

    for i in range(int(train_num)):
        # suijishu = np.random.randint(0, len(train_image), args.batch_size)
        for j in range(args.batch_size):
            # 固定采样
            train_image_epoch[j, 0, :, :] = train_image[i * args.batch_size + j]
            train_label_epoch[j, 0, :, :] = train_label[i * args.batch_size + j]

            # 随机采样
            # train_image_epoch[j, :, :, :] = train_image[suijishu[j]]
            # train_label_epoch[j, :, :, :] = train_label[suijishu[j]]
        input = train_image_epoch.copy()
        target = train_label_epoch.copy()
        input = torch.from_numpy(input).cuda().float()
        target = torch.from_numpy(target).cuda().float()

        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = dice_coef(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = dice_coef(output, target)
            '''
            loss = 0
            iou = 0
            for i_dice in range(output.shape[1]):
                loss = loss + criterion(output[:, i_dice, :, :], target[:, i_dice, :, :]) / output.shape[1]
                iou = iou + dice_coef(output[:, i_dice, :, :], target[:, i_dice, :, :]) / output.shape[1]
            '''
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, model, criterion, validate_image, validate_label):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        test_num = len(validate_image) // args.batch_size
        test_image_epoch = np.zeros(shape=(args.batch_size, 1, 384, 384))
        train_label_epoch = np.zeros(shape=(args.batch_size, 1, 384, 384))
        for i in range(int(test_num)):
            # suijishu = np.random.randint(0, len(validate_image), args.batch_size)
            for j in range(args.batch_size):
                test_image_epoch[j, 0, :, :] = validate_image[i * args.batch_size + j]
                train_label_epoch[j, 0, :, :] = validate_label[i * args.batch_size + j]
            input = test_image_epoch.copy()
            target = train_label_epoch.copy()
            input = torch.from_numpy(input).cuda().float()
            target = torch.from_numpy(target).cuda().float()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = dice_coef(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = dice_coef(output, target)
                '''
                loss = 0
                iou = 0

                for i_dice in range(output.shape[1]):
                    loss = loss + criterion(output[:, i_dice, :, :], target[:, i_dice, :, :]) / output.shape[1]
                    iou = iou + dice_coef(output[:, i_dice, :, :], target[:, i_dice, :, :]) / output.shape[1]
                '''
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()
    # args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' % (args.dataset, args.arch)
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    # define loss function (criterion) BCEwithlogitsloss:将BCE与sigmoid和为一步
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % args.arch)
    model = denseUnet.__dict__[args.arch](args)
    '''
    state_dict = torch.load('models/%s/model_train_loss_bce_dice.pth' % args.name)  # 模型
    # 直接丢弃不需要的模块
    state_dict.pop('outconv.weight')
    state_dict.pop('outconv.bias')
    model.load_state_dict(state_dict, strict=False)
    '''
    model = model.cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    trigger = 0

    # 读取训练数据集
    train_image, train_label = zongimage_train()

    suiji_zhongzi_train = random.sample(range(0, len(train_image)), len(train_image))
    suiji_train_image = train_image.copy()
    suiji_train_label = train_label.copy()
    for i in range(len(train_image)):
        suiji_train_image[i] = train_image[suiji_zhongzi_train[i]].copy()
        suiji_train_label[i] = train_label[suiji_zhongzi_train[i]].copy()
    np.savez('datasets_fatpad_train', suiji_train_image=suiji_train_image, suiji_train_label=suiji_train_label)  # 保存乱序后的数据集

    validate_image, validate_label = zongimage_validate()

    suiji_zhongzi_validate = random.sample(range(0, len(validate_image)), len(validate_image))
    suiji_validate_image = validate_image.copy()
    suiji_validate_label = validate_label.copy()
    for i in range(len(validate_image)):
        suiji_validate_image[i] = validate_image[suiji_zhongzi_validate[i]].copy()
        suiji_validate_label[i] = validate_label[suiji_zhongzi_validate[i]].copy()
    np.savez('datasets_fatpad_validate', suiji_validate_image=suiji_validate_image,
             suiji_validate_label=suiji_validate_label)  # 保存乱序后的数据集

    '''
    datasets_jinggu_train = np.load('datasets_fatpad_train.npz')  # 加载train datasets
    suiji_train_image = datasets_jinggu_train['suiji_train_image']
    suiji_train_label = datasets_jinggu_train['suiji_train_label']

    datasets_jinggu_validate = np.load('datasets_fatpad_validate.npz')  # 加载validate datasets
    suiji_validate_image = datasets_jinggu_validate['suiji_validate_image']
    suiji_validate_label = datasets_jinggu_validate['suiji_validate_label']
    '''
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        # train for one epoch
        train_log = train(args, model, criterion, optimizer, epoch, suiji_train_image, suiji_train_label)
        # evaluate on validation set
        val_log = validate(args, model, criterion, suiji_validate_image, suiji_validate_label)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model_train_fatpad.pth' % args.name)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
