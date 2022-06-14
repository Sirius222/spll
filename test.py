import argparse
from functools import reduce
from multiprocessing import reduction
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        # get train dataset else false
        training=True,
        num_workers=2
    )
    noise_or_not = data_loader.train_dataset.noise_or_not

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['val_loss'])
    loss_fn = getattr(module_loss, config['val_loss']['type'])(reduce=False, ignore_index=-1).cuda()

    # # todo: change this for test
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    # print(str(config['metrics']))
    # metric_fns = getattr(module_metric, str(config['metrics']))


    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    num_classes = config['num_classes']
    total_example = config['total_example']
    total_metrics = torch.zeros(num_classes)
    import numpy as np
    example_loss= np.zeros_like(noise_or_not, dtype=float)
    k = config['select_clean'] * total_example
    k = int(k)
    k_noise = int((1.-config['select_noise']) * total_example)


    with torch.no_grad():
        for i, (data, target, idx, gt) in enumerate(tqdm(data_loader)):


            data, target, gt = data.to(device), target.to(device), gt.to(device)

            output = model(data)
            # computing loss, metrics on test set
            loss = loss_fn(output, target)

            for pi, cl in zip(idx, loss):
                example_loss[pi] = cl.cpu().data.item()

            loss = loss.mean()
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
            # for i in range(1, 1+num_classes):
                total_metrics[i] += metric(output, target) * batch_size
                # total_metrics[i] += metric_fns(output, gt, k=i) * batch_size

    _arg_index = np.argsort(example_loss)


    clean_acc = np.sum(noise_or_not[_arg_index[:k]]) / float(k)
    noise_acc = np.sum(noise_or_not[_arg_index[k_noise:]]) / float(total_example - k)

    # n_samples = len(data_loader.sampler)
    n_samples = total_example
    log = {'loss': total_loss / n_samples}
    log.update({"noise_acc": 1-noise_acc})
    log.update({"clean_acc": clean_acc})
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume',
                default='./experiment/models/cifar10_PreActResNet18_real_worse_label/0601_043410/model_best.pth', 
                type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.get_instance(args, '')
    #config = ConfigParser(args)
    main(config)
