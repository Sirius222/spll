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




# def log_params(conf: OrderedDict, parent_key: str = None):
#     for key, value in conf.items():
#         if parent_key is not None:
#             combined_key = f'{parent_key}-{key}'
#         else:
#             combined_key = key

#         if not isinstance(value, OrderedDict):
#             mlflow.log_param(combined_key, value)
#         else:
#             log_params(value, combined_key)


def main(config: ConfigParser):
    tb_logger = tb.Logger(logdir=config.tb_logger, flush_secs=2)
    logger = config.get_logger('train')

    # model = config.initialize('arch', module_arch)
    model = None
    # ens_example_loss, ens_prediction, train_labels = sop_train(config=config, model=model, logger=logger, tb_logger=tb_logger)
    ens_example_loss, ens_prediction, train_labels = None, None, None

    dpll(config=config, logger=logger,model=model, ens_example_loss=ens_example_loss, ens_prediction=ens_prediction, train_labels=train_labels)



def sop_train(config=None, model=None, logger=None, tb_logger=None):
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'] 
    )
    valid_data_loader = data_loader.split_validation()


    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    reparametrization_net = None

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)

    config['train_loss']['args']['num_examp'] = num_examp

    train_loss = config.initialize('train_loss', module_loss)

    # test_loss
    val_loss = config.initialize('val_loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = [{'params': [p for p in model.parameters() if  getattr(p, 'requires_grad', False)]}]
    reparam_params = [{'params': [train_loss.u, train_loss.v], 'lr': config['optimizer_overparametrization']['args']['lr'], 'weight_decay': config['optimizer_overparametrization']['args']['weight_decay']}]
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    optimizer_overparametrization = config.initialize('optimizer_overparametrization', torch.optim, reparam_params)


    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    lr_scheduler_overparametrization = None


    trainer = PLL_Trainer(model, reparametrization_net, train_loss, metrics, optimizer, optimizer_overparametrization,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      lr_scheduler_overparametrization = lr_scheduler_overparametrization,
                      val_criterion=val_loss,
                      tb_logger=tb_logger)


    trainer.train()


    noise_or_not = data_loader.train_dataset.noise_or_not
    
    np.save(config.pll_label/'ens_prediction', trainer.ens_prediction) # 300, 50000, 10
    np.save(config.pll_label/'ens_example_loss', trainer.ens_example_loss) # 300, 50000
    np.save(config.pll_label/'noise_or_not', noise_or_not)
    return trainer.ens_example_loss, trainer.ens_prediction, data_loader.train_dataset.train_labels

def dpll(config=None, logger=None,model=None, ens_example_loss=None, ens_prediction=None, train_labels=None):
    if config['data_loader']['type'] == 'CIFAR10DataLoader':
        sup_idx, pll_idx, pll_label = generate_idx(config, ens_example_loss, ens_prediction, train_labels=train_labels)
        sup_train_loader, pll_train_loader, test = dataset.mycifar10_dataloaders(config['data_loader']['args']['data_dir'],rate=0.4, sup_index=sup_idx, pll_index=pll_idx, pll_label=pll_label)

    else:
        raise NotImplementedError("todo cifar100")

    if model is None:
        model = WideResNet(34, config['classes'], widen_factor=10, dropRate=0.0)


    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config['dpll']['lr'], momentum=0.9, weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    cudnn.benchmark = True
    # init confidence
    confidence = copy.deepcopy(pll_train_loader.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    # Train loop
    for epoch in tqdm(range(0, config['dpll']['epoch'])):
        # training
        trainloss = DPLL_train(sup_train_loader, pll_train_loader, model, optimizer, epoch, consistency_criterion, confidence, config=config)
        # lr_step
        scheduler.step()
        # evaluate on validation set
        valacc, valloss = validate(test, model, criterion, epoch, logger=logger)
        log = {
            "loss": trainloss,
            "learning rate": optimizer.param_groups[0]['lr'],
            "valacc": valacc,
            "valloss": valloss
        }
        for key, value in log.items():
                logger.info('    {:15s}: {}'.format(str(key), value))
        

def DPLL_train(sup_train_loader, pll_train_loader, model, optimizer, epoch, consistency_criterion, confidence, config=None):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    # assume len(sup_train_loader) < len(pll_train_loader)
    if len(sup_train_loader) < len(pll_train_loader):
        sup_train_loader = cycle(sup_train_loader)
    elif len(sup_train_loader) > len(pll_train_loader):
        pll_train_loader = cycle(pll_train_loader)
    else:
        pass


    for sup_train, pll_train in tqdm(zip(sup_train_loader, pll_train_loader)):
        (x_aug0, x_aug1, x_aug2, y, part_y, index) = pll_train

        # sup_loss is single label supervised loss
        sup_loss = sup_train_loss(model=model, sup_train=sup_train,config=config)
        # measure data loading time 
        data_time.update(time.time() - end)
        # partial label
        part_y = part_y.float().cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        y_pred_aug0 = model(x_aug0)
        # augmentation1
        x_aug1 = x_aug1.cuda()
        y_pred_aug1 = model(x_aug1)
        # augmentation2
        x_aug2 = x_aug2.cuda()
        y_pred_aug2 = model(x_aug2)

        y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

        # consist loss
        consist_loss0 = consistency_criterion(y_pred_aug0_probas_log, torch.tensor(confidence[index]).float().cuda())
        consist_loss1 = consistency_criterion(y_pred_aug1_probas_log, torch.tensor(confidence[index]).float().cuda())
        consist_loss2 = consistency_criterion(y_pred_aug2_probas_log, torch.tensor(confidence[index]).float().cuda())
        # supervised loss
        super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0, dim=1)) * (1 - part_y), dim=1))
        # dynamic lam
        lam = min((epoch / 100) * config['dpll']['lam'], config['dpll']['lam'])

        # Unified loss
        final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + super_loss + sup_loss

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # update confidence
        confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index, config=config)

        losses.update(final_loss.item(), x_aug0.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    return losses.avg


def generate_idx(config=None, ens_example_loss=None, ens_prediction=None, train_labels=None):
    if ens_example_loss is None or ens_prediction is None or train_labels is None:
        ens_example_loss = np.load('./data/worse_label/loss.npy')
        ens_prediction = np.load('./data/worse_label/prediction.npy')
        train_labels = torch.load('/home/sirius/ght/LMNL-2022/task-classification/spll/data/CIFAR-10_human.pt')
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


def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index, config=None):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(config['classes'], 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu().numpy()



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



