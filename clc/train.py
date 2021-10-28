import torch
import torch.nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models._3d import resnet
import utils
from dataset import DatasetList3d
import argparse
from config import _C as cfg   #导入基本设置
import time
import os
import torch.nn.functional as F

device = torch.device('cuda') #设置设备GPU

from read_fatpad import people_image_train, zongimage_validate, people_label #获取裁剪后图像与对应的标签
train_dir = 'D:\BaiduNetdiskDownload/clc_data' #训练集目录
train_validate = 300
train_data = ['tkr case1', 'tkr contro2l', 'tkr contro3l', 'IOA case1', 'IOA case2', 'IOA case3', 'IOA case4', 'IOA control1', 'IOA control2', 'IOA control3']
validate_data = ['tkr case2', 'tkr contro4', 'IOA case5', 'IOA contro4']
test = ['tkr case3', 'tkr control 1', 'IOA case6', 'IOA contro5']

def main(cfg):
    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    init_num_classes = 4 #四分类

    net = resnet.resnet50(num_classes=init_num_classes)#.to(device)
    # net = resnet.resnet101(num_classes=2, shortcut_type='B', spatial_size=spatial_size, sample_count=sample_count).to(device)
    net = torch.nn.DataParallel(net, device_ids=None)  # 数据并行计算

    if cfg.MODEL.pretrained_path:  # 判断是否要载入预训练模型，这里设置为不加载
        print('loading pretrained model {}'.format(cfg.MODEL.pretrained_path))
        pretrain = torch.load(cfg.MODEL.pretrained_path)
        net.load_state_dict(pretrain['state_dict'])

    net.module.fc = torch.nn.Linear(net.module.fc.in_features, init_num_classes)    # 全连接层
    net.module.fc = net.module.fc#.cuda()

    #criterion = torch.nn.CrossEntropyLoss()    # 定义损失函数
    criterion = F.binary_cross_entropy_with_logits
    # criterion = torch.nn.NLLLoss(ignore_index=-1)
    optimizer = get_optimizer(net, cfg)      # 定义优化器

    max_acc = 0
    for epoch in range(cfg.TRAIN.tr_num_epochs): #默认为20epoch
        net, avg_acc, avg_loss = train(net, optimizer, criterion, epoch + 1, cfg) # 训练
        #val_avg_loss, val_avg_acc = val(net, criterion, epoch + 1, cfg) # 验证

        #print(cfg.TRAIN.best_acc_cfg)
        #checkpoint(net, cfg, epoch + 1)
        if avg_acc > max_acc:
            max_acc = avg_acc
            torch.save(
                net,
                '{}/model_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch)) #如果训练准确率新高，保存模型
            print("=> saved best model")
        print('epoch:',epoch,'train_loss:',avg_loss,'train_acc:',avg_acc) #展示当前epoch的训练集损失、准确率、验证集损失、准确率

    print('Finished Training') #结束训练


def get_optimizer(net, cfg): #设置优化器，有sgd和adam两种（默认为SGD）
    optim = ''
    if cfg.TRAIN.tr_optim.lower() == 'sgd':
        optim = torch.optim.SGD(
            net.parameters(),
            lr=cfg.TRAIN.tr_lr,
            momentum=cfg.TRAIN.tr_momentum,
            weight_decay=cfg.TRAIN.tr_weight_decay)
    elif cfg.TRAIN.tr_optim.lower() == 'adam':
        optim = torch.optim.Adam(
            net.parameters(),
            lr=cfg.TRAIN.tr_lr,
            weight_decay=cfg.TRAIN.tr_weight_decay)

    return optim


def adjust_learning_rate(optimizer, cur_iter, cfg): #调整学习率
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.tr_lr_pow)
    cfg.TRAIN.running_lr = cfg.TRAIN.tr_lr * scale_running_lr

    optimizer_encoder = optimizer
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr


def train(net, optimizer, criterion, epoch, tmp_cfg): # 训练：输入参数：网络，损失函数，epoch，默认参数
    net.train()

    cum_loss = 0.0
    cum_acc = 0.0
    c = 0
    sheets2 = people_label()    #标签表, ndarry格式
    jishu = 0
    for eni in range(cfg.TRAIN.tr_epoch_num_iters):
        #for i, data in enumerate(dataloader, 0):
        #files_peoples = os.listdir(train_dir)
        for i in range(train_validate):
            inputs = np.zeros(shape=(1, 1, 32, 224, 224))   #输入维度为32*224*224
            people_num = sheets2[i, 0]    #是哪个人（编号）
            people_leg = sheets2[i, 1]    #是哪条腿（1/2）
            labels = np.zeros([1, 4])
            if sheets2[i, 2] == 0:     #设置标签（0123,4个等级，one-hot编码）
                labels[0, 0] = 1
            elif sheets2[i, 2] == 1:
                labels[0, 1] = 1
            elif sheets2[i, 2] == 2:
                labels[0, 2] = 1
            elif sheets2[i, 2] == 3:
                labels[0, 3] = 1

            if not os.path.exists(os.path.join(train_dir, str(people_num))):     #如果没有这个人，则跳过这一次循环，否则读取相应的序列
                print(i)
                continue
            else:
                if people_leg == 1:
                    xulie = str(people_num) + '-00m-SAG_IW_TSE_RIGHT' #右腿
                    people_dir = os.path.join(train_dir, str(people_num), xulie)
                if people_leg == 2:
                    xulie = str(people_num) + '-00m-SAG_IW_TSE_LEFT' #左腿
                    people_dir = os.path.join(train_dir, str(people_num), xulie)
                inputs_np = people_image_train(people_dir) #读入图像裁剪数据
                jishu = jishu + 1
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels, _ = data
            #输入数据、标签存到cuda中
            inputs[0, 0, :, :, :] = inputs_np
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).float()
            #inputs = torch.from_numpy(inputs).cuda().float()
            #labels = torch.from_numpy(labels).cuda().float()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) #计算损失

            optimizer.zero_grad()   #清空过往梯度；
            loss.backward()     #反向传播，计算当前梯度；
            optimizer.step()      #根据梯度更新网络参数

            softmax = torch.nn.Softmax()
            outputs = softmax(outputs) #经过softmax后真正的输出
            # print(outputs)
            # print(labels)
            # acc1 = utils.compute_accuracy(
            #     outputs,
            #     labels,
            #     augmentation=False,
            #     topk=(1, 1))
            _, pred = outputs.topk(1, 1, True, True)
            _, pred_target = labels.topk(1, 1, True, True)
            if pred == pred_target:
                acc1=1
            else:
                acc1=0
            cum_loss = cum_loss + loss
            cum_acc = cum_acc + acc1

    print(jishu) #训练结束
    avg_loss = cum_loss / (jishu * cfg.TRAIN.tr_epoch_num_iters)
    avg_acc = cum_acc / (jishu * cfg.TRAIN.tr_epoch_num_iters)

    return net, avg_acc, avg_loss

    #get_best_acc(float(avg_acc), cfg, 0)


def val(net, criterion, epoch, cfg): #验证函数;输入： 网络，损失函数，epoch，默认参数
    net.eval()
    with torch.no_grad():
        cum_loss = 0.0
        cum_acc = 0.0
        c = 0
        sheets3 = people_label()  # 标签表, ndarry格式
        jishu = 0
        for eni in range(cfg.TRAIN.tr_epoch_num_iters):
            # for i, data in enumerate(dataloader, 0):
            #files_peoples = os.listdir(train_dir)
            for i in range(100): #在训练集使用的300个后面采用100个做验证
                inputs = np.zeros(shape=(1, 1, 32, 224, 224))
                people_num = sheets3[i + train_validate, 0]  # 是哪个人
                people_leg = sheets3[i + train_validate, 1]  # 是哪条腿
                labels = np.zeros([1, 4])
                if sheets3[i + train_validate, 2] == 0:  # 设置标签
                    labels[0, 0] = 1
                elif sheets3[i + train_validate, 2] == 1:
                    labels[0, 1] = 1
                elif sheets3[i + train_validate, 2] == 2:
                    labels[0, 2] = 1
                elif sheets3[i + train_validate, 2] == 3:
                    labels[0, 3] = 1

                if not os.path.exists(os.path.join(train_dir, str(people_num))):  # 如果没有这个人，则跳过这一次循环，否则读取相应的序列
                    print(i)
                    continue
                else:
                    if people_leg == 1:
                        xulie = str(people_num) + '-00m-SAG_IW_TSE_RIGHT'
                        people_dir = os.path.join(train_dir, str(people_num), xulie)
                    if people_leg == 2:
                        xulie = str(people_num) + '-00m-SAG_IW_TSE_LEFT'
                        people_dir = os.path.join(train_dir, str(people_num), xulie)
                    inputs_np = people_image_train(people_dir)
                    jishu = jishu + 1
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels, _ = data
                inputs[0, 0, :, :, :] = inputs_np
                #inputs = torch.from_numpy(inputs).cuda().float()
                #labels = torch.from_numpy(labels).cuda().float()
                inputs = torch.from_numpy(inputs).float()
                labels = torch.from_numpy(labels).float()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                softmax = torch.nn.Softmax()
                outputs = softmax(outputs)

                # acc1 = utils.compute_accuracy(
                #     outputs,
                #     labels,
                #     augmentation=False,
                #     topk=(1, 1))
                _, pred = outputs.topk(1, 1, True, True)
                _, pred_target = labels.topk(1, 1, True, True)
                if pred == pred_target:
                    acc1 = 1
                else:
                    acc1 = 0
                cum_loss = cum_loss + loss
                cum_acc = cum_acc + acc1

        print(jishu)
        avg_loss = cum_loss / (jishu * cfg.TRAIN.tr_epoch_num_iters)
        avg_acc = cum_acc / (jishu * cfg.TRAIN.tr_epoch_num_iters)
        return avg_loss, avg_acc



def get_best_acc(acc, cfg, idx):
    if acc >= cfg.TRAIN.best_acc[idx]:
        cfg.TRAIN.tmp_acc[idx] = acc
        cfg.TRAIN.best_acc_cfg[idx] = True

def checkpoint(net, cfg, epoch):
    if cfg.TRAIN.ckpt_interval < 0:
        if cfg.TRAIN.best_acc_cfg[0] and cfg.TRAIN.best_acc_cfg[1] and\
                -cfg.TRAIN.ckpt_interval <= epoch:
            if -cfg.TRAIN.ckpt_interval != epoch:
                print('removing previous model...')
                os.remove('{}/model_epoch_{}_best.pth'\
                          .format(cfg.CKPT_DIR, cfg.TRAIN.best_acc_cfg[2]))
                cfg.TRAIN.best_acc_cfg[2] = epoch

            print('Saving best model...')
            dict_model = net.state_dict()
            torch.save(
                dict_model,
                '{}/model_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch))

            cfg.TRAIN.best_acc = cfg.TRAIN.tmp_acc

        cfg.TRAIN.best_acc_cfg[0] = False
        cfg.TRAIN.best_acc_cfg[1] = False

    elif epoch % cfg.TRAIN.ckpt_interval == 0 or epoch == cfg.TRAIN.tr_num_epochs:
        print('Saving checkpoints...')
        dict_model = net.state_dict()
        torch.save(
            dict_model,
            '{}/model_epoch_{}.pth'.format(cfg.CKPT_DIR, epoch))


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch 3D Classification Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/DCM_RP2-resnet50_3d_CV5_rsp224.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--logs",
        default="./logs",
        help="path to logs dir"
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    log_file = os.path.join(args.logs, cur_time+'.log')

    #logger = utils.setup_logger(distributed_rank=0, filename=log_file)

    if not os.path.isdir(cfg.CKPT_DIR):
        os.makedirs(cfg.CKPT_DIR)

    with open(os.path.join(cfg.CKPT_DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    main(cfg)
