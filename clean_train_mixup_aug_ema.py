import time
import argparse
import collections
from logging import root
import numpy as np
from wideresnet import WideResNet
from lenet import LeNet
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import PLL_Trainer
from collections import OrderedDict
import random
import tensorboard_logger as tb
import os
import dataset as dataset
from torch import nn
from torch.backends import cudnn
import copy
from tqdm import tqdm
from torch.autograd import Variable
from itertools import cycle
import torch.nn.functional as F


def main(config: ConfigParser):
    tb_logger = tb.Logger(logdir=config.tb_logger, flush_secs=2)
    logger = config.get_logger('train')

    ens_example_loss, ens_prediction, train_labels = None, None, None

    if config['data_loader']['type'] == 'CIFAR10DataLoader':
        sup_idx, pll_idx, pll_label = generate_idx(config, ens_example_loss, ens_prediction, train_labels=train_labels)
        sup_train_loader, _, test = dataset.mycifar10_dataloaders(config['data_loader']['args']['data_dir'],rate=0.4, sup_index=sup_idx, pll_index=pll_idx, pll_label=pll_label)

    else:
        raise NotImplementedError("todo cifar100")

    model = WideResNet(34, config['classes'], widen_factor=10, dropRate=0.0)
    model = torch.nn.DataParallel(model).cuda()
    if True: # ema
        ema_model = WideResNet(34, config['classes'], widen_factor=10, dropRate=0.0)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()  
        ema_model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config['dpll']['lr'], momentum=0.9, weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    
    cudnn.benchmark = True
    # init confidence

    # Train loop
    best_acc = 0.0
    best_ema_acc = 0.0
    for epoch in tqdm(range(0, config['dpll']['epoch'])):
        # training
        trainloss = DPLL_train(sup_train_loader, None, model, optimizer, epoch, config=config, ema_model = ema_model)
        # lr_step
        scheduler.step()
        # evaluate on validation set
        valacc, valloss = validate(test, model, criterion, epoch, logger=logger)
        ema_acc, ema_loss = validate(test, ema_model, criterion, epoch, logger=logger)
        best_acc = max(best_acc, valacc)
        best_ema_acc = max(best_ema_acc, ema_acc)
        
        log = {
            "epoch": epoch,
            "loss": trainloss,
            "learning rate": optimizer.param_groups[0]['lr'],
            "valacc": valacc,
            "valloss": valloss,
            "emaacc":ema_acc,
            "emaloss":ema_loss
        }
        for key, value in log.items():
            logger.info('    {:15s}: {}'.format(str(key), value))
        tb_logger.log_value("val loss", valloss, epoch)
        tb_logger.log_value("val acc", valacc, epoch)
        tb_logger.log_value("ema loss", ema_loss, epoch)
        tb_logger.log_value("ema acc", ema_acc, epoch)
    logger.info('best acc is ', best_acc)
    logger.info('best ema_acc is ', best_ema_acc)

def DPLL_train(sup_train_loader, pll_train_loader, model, optimizer, epoch, config=None, ema_model=None):
    """
        Run one train epoch
    """
    losses = AverageMeter()

    model.train()
    for sup_train in tqdm(sup_train_loader):

        # sup_loss is single label supervised loss
        sup_loss = sup_train_loss(model=model, sup_train=sup_train,config=config)
        # measure data loading time 

        optimizer.zero_grad()
        sup_loss.backward()
        optimizer.step()

        # todo add ema
        with torch.no_grad():
            ema_model_update(model, ema_model, 0.999)

        # update confidence
        losses.update(sup_loss.item(), sup_train[0].size(0))
        # measure elapsed time


    return losses.avg

def sup_train_loss(model=None, sup_train=None, config=None):
    
    x_aug0, x_aug1, x_aug2, y, part_y, index = sup_train
    x_aug1 = x_aug1.cuda()
    part_y = part_y.cuda()
    # aug 1, part_y
    # one-hot -> single label
    part_y = torch.argmax(part_y, dim=1).reshape(-1,)
    inputs, targets_a, targets_b, lam = mixup_data(x_aug1, part_y, config['dpll']['alpha'])
    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
    outputs = model(inputs)
    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
    return loss

def generate_idx(config=None, ens_example_loss=None, ens_prediction=None, train_labels=None):
    if ens_example_loss is None or ens_prediction is None or train_labels is None:
        ens_example_loss = np.load('./data/worse_label/loss.npy')
        ens_prediction = np.load('./data/worse_label/prediction.npy')
        train_labels = torch.load('./data/CIFAR-10_human.pt')
        train_labels = train_labels['worse_label']
    topk = config['topk']
    ens = config['ensemble']
    clean_samples = int(config['clean_select'] * config['samples'])
    l = np.sum(ens_example_loss[300-ens:,:], axis=0)
    pre = np.sum(ens_prediction[300-ens:,::], axis=0)
    arg_idx = np.argsort(l)
    sup_idx = arg_idx[:clean_samples]
    pll_idx = arg_idx[clean_samples:]
    pll_label = np.zeros_like(pre)
    for i,p in enumerate(pre):
        if i in sup_idx:
            pll_label[i, train_labels[i]] = 1
        else:
            label = torch.topk(torch.tensor(p), topk)[1].numpy()
            pll_label[i, label] = 1
    return sup_idx, pll_idx, pll_label

def validate(valid_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss().cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)  

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--lr_op', '--learning_rate_overparametrization'], type=float, target=('optimizer_overparametrization', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--instance', '--instance'], type=bool, target=('trainer', 'instance')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--key', '--comet_key'], type=str, target=('comet','api')),
        CustomArgs(['--offline', '--comet_offline'], type=str, target=('comet','offline')),
        CustomArgs(['--std', '--standard_deviation'], type=float, target=('reparam_arch','args','std')),
        CustomArgs(['--malpha', '--mixup_alpha'], type=float, target=('mixup','alpha')),
        CustomArgs(['--consist', '--ratio_consistency'], type=float, target=('train_loss','args','ratio_consistency')),
        CustomArgs(['--balance', '--ratio_balance'], type=float, target=('train_loss','args','ratio_balance')),
        CustomArgs(['--reg', '--ratio_reg'], type=float, target=('train_loss','args','ratio_reg')),
    ]
    config = ConfigParser.get_instance(args, options)
    

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    main(config)

