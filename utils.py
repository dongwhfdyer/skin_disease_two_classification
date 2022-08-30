import shutil
from pathlib import Path

import torch
# import adabound
import os
import logging


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def get_optimizer(model, args):
    parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params': param, 'lr': args.lr * args.lr_fc_times})
        else:
            parameters.append({'params': param, 'lr': args.lr})
    # parameters = model.parameters()
    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters,
                               # model.parameters(),
                               args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters,
                                   # model.parameters(),
                                   args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(parameters,
                                # model.parameters(),
                                args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    # elif args.optimizer == 'radam':
    #     return RAdam(parameters, lr=args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay)

    else:
        raise NotImplementedError


def save_checkpoint(state, is_best, single=True, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if single:
        fold = ''
    else:
        fold = str(state['fold']) + '_'
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    if is_best:
        model_name = 'model_' + str(state['epoch']) + '_' + str(int(round(state['train_acc'] * 100, 0))) + '_' + str(int(round(state['acc'] * 100, 0))) + '.pth'
        model_path = os.path.join(checkpoint, fold + model_name)
        torch.save(state['state_dict'], model_path)
        logger.info("update best model")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    # top1 accuracy
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)  # 返回最大的k个结果（按最大到小排序）

    pred = pred.t()  # 转置

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / batch_size)

    return res


def create_model_logger(logger_name, log_file, log_level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # 建立一个filehandler来把日志记录在文件里，级别为debug以上
    # if log file not exists, create it
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 设置日志格式
    formatter_1 = logging.Formatter("[%(levelname)s] - %(filename)s - %(lineno)d: %(message)s")
    # formatter_2 = logging.Formatter("[%(levelname)s] - %(filename)s - %(lineno)d: %(message)s")
    formatter_3 = logging.Formatter("%(message)s")
    ch.setFormatter(formatter_3)
    fh.setFormatter(formatter_1)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def advancedLogger():
    logger_name = "global"
    log_file = "rubb/global.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # 建立一个filehandler来把日志记录在文件里，级别为debug以上
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 设置日志格式
    FORMATTER_1 = logging.Formatter("[%(levelname)s] - %(filename)s - %(lineno)d: %(message)s")
    FORMATTER_2 = logging.Formatter("[%(levelname)s] - %(filename)s - %(lineno)d: %(message)s")
    FORMATTER_3 = logging.Formatter("%(message)s")
    FORMATTER_4 = logging.Formatter("[%(levelname)s] - %(asctime)s - %(filename)s - %(lineno)d: %(message)s")
    ch.setFormatter(FORMATTER_3)
    fh.setFormatter(FORMATTER_4)
    # 将相应的handler添加在logger对象中
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = advancedLogger()


def clear_empty_checkpoints_folder():
    checkpoint_folder = Path(r"checkpoints")
    for folder in checkpoint_folder.iterdir():
        if folder.is_dir():
            # if no pth file, then delete this folder
            if not any(folder.glob('*.pth')):
                shutil.rmtree(folder)
                logger.info("delete empty folder: {}".format(folder))


if __name__ == '__main__':
    # logger.info("hello world")
    clear_empty_checkpoints_folder()
    pass
