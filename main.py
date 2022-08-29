""" 
@ author: Qmh
@ file_name: main.py
@ time: 2019:11:20:11:24
"""
import logging
import shutil
import time
import re

import yaml
from tensorboardX import SummaryWriter
from pathlib import Path

import numpy as np

from args import args
import torch
import torch.nn as nn
import models
import data_gen
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
from transform import get_transforms
import os
from build_net import make_model, make_regression_model
from utils import get_optimizer, AverageMeter, save_checkpoint, accuracy, create_model_logger, logger
import torchnet.meter as meter
import pandas as pd
from sklearn import metrics

torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()

best_acc = 0


def main():
    global best_acc

    time_now = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

    checkpoints_path = Path(args.checkpoint) / (args.save_prefix + "_" + str(time_now))
    tensorboard_log_path = Path("tensorboard_log") / time_now
    if args.if_resume:
        logger.warning("Resume from checkpoint: %s", Path(args.model_path))
        checkpoints_path = Path(args.model_path).parent
        if (Path("tensorboard_log") / checkpoints_path.name).exists():
            tensorboard_log_path = Path("tensorboard_log") / checkpoints_path.name

    create_folders(checkpoints_path)
    model_logger = create_model_logger(logger_name="model_logger", log_file=checkpoints_path / "training.log")
    writer = SummaryWriter(str(tensorboard_log_path))

    logger.info(f'Arguments: {args}')
    logger.info(f'Checkpoints will be saved to {checkpoints_path}')

    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    train_set = data_gen.Dataset(root=args.train_txt_path, transform=transformations['val_train'])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_set = data_gen.ValDataset(root=args.val_txt_path, transform=transformations['val_test'])
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.if_regression:
        model = make_regression_model(args.model_path)
    else:
        model = make_model(args)

    # ---------kkuhn-block------------------------------ saving training options and model architecture
    opt_yaml_save_path = "opt.yaml"
    model_info_save_path = "model.json"
    if args.if_resume:
        opt_yaml_save_path = "resume_" + str(time_now) + "_" + opt_yaml_save_path
        model_info_save_path = "resume_" + str(time_now) + "_" + model_info_save_path
    with open(checkpoints_path / opt_yaml_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
    with open(checkpoints_path / model_info_save_path, 'w') as f:
        f.write(model.__repr__())
    # ---------kkuhn-block------------------------------

    if use_cuda:
        model.cuda()

    if args.if_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if use_cuda:
        criterion = criterion.cuda()

    optimizer = get_optimizer(model, args)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)  # todo: if use this scheduler?

    # load checkpoint
    start_epoch = args.start_epoch
    if args.if_resume:
        # re format 1_5519_5322.pth
        epoch_acc_pth_pattern = re.compile(r'model_(\d){1,4}_(\d){2}_(\d){2}\.pth$')
        re_pth_path = epoch_acc_pth_pattern.search(args.model_path)
        if re_pth_path:
            pth_path = re_pth_path.group()
            start_epoch, _, best_acc = pth_path[:-4].split('_')[1:4]  # e.g. args.model_path = model_1_5519_5322.pth
            start_epoch = int(start_epoch) + 1
            best_acc = float(best_acc) / 100
            logger.info(f'Resume from epoch {start_epoch} with best acc {best_acc}')
        else:
            logger.warning(f'Cannot resume from {args.model_path}')

        # model.module.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, args.epochs):
        logger.info('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        # logger.info('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        if args.if_regression:
            train_loss, train_MAE, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        else:
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)

        if args.if_regression:
            test_loss, val_MAE, val_acc = val_for_regression(val_loader, model, criterion, epoch, use_cuda)
        else:
            test_loss, val_acc = val(val_loader, model, criterion, epoch, use_cuda)

        scheduler.step(test_loss)
        if args.if_regression:
            model_logger.info(f'train_loss:{train_loss:.8f}\t val_loss:{test_loss:.8f}\t train_MAE:{train_MAE:.3f}\t train_acc:{train_acc:.3f}\t val_MAE:{val_MAE:.3f}\t val_acc:{val_acc:.3f}')
        else:
            model_logger.info(f'train_loss:{train_loss:.8f}\t val_loss:{test_loss:.8f}\t train_acc:{train_acc:.3f} \t val_acc:{val_acc:.3f}')
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', test_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        if args.if_regression:
            writer.add_scalar("val_MAE", val_MAE, epoch)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        logger.info(f'Memory used: {mem} GB')
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            'fold': 0,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, single=True, checkpoint=checkpoints_path)

    print("best acc = ", best_acc)


loss_bias = 33.0  # todo:  loss is too big, so we add loss_bias to avoid it.


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    losses = AverageMeter()
    train_meter = AverageMeter()
    if args.if_regression:
        accept_item = 0
        all_items = len(train_loader.dataset)
    for (inputs, targets) in tqdm(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # kuhn edited
        if args.if_regression:
            targets = torch.reshape(targets, (-1, 1))
            # inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        # logger.debug(f'outputs:{outputs * 150.0}')
        if args.if_regression:
            loss = criterion(outputs, targets / loss_bias)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        if args.if_regression:
            outputs_ = outputs * loss_bias
            error = torch.abs(outputs_ - targets) - 10
            accept_item += torch.sum(error <= 0).item()
            mean_absolute_error = metrics.mean_absolute_error(outputs_.detach().cpu().numpy(), targets.detach().cpu().numpy())
            train_meter.update(mean_absolute_error.item(), inputs.size(0))
        else:
            acc = accuracy(outputs.data, targets.data)
            train_meter.update(acc.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
    if args.if_regression:
        train_acc = accept_item / all_items
        return losses.avg, train_meter.avg, train_acc
    else:
        return losses.avg, train_meter.avg


def val_for_regression(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    val_MAE = AverageMeter()

    model.eval()
    accept_item = 0
    all_items = len(val_loader.dataset)
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            targets = torch.reshape(targets, (-1, 1))
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets / loss_bias)
            outputs_ = outputs * loss_bias
            error = torch.abs(outputs_ - targets) - 10
            # count the element that is less than 1
            accept_item += torch.sum(error <= 0).item()

            mean_absolute_error = metrics.mean_absolute_error(outputs_.cpu().numpy(), targets.cpu().numpy())
            # numpy.array to float
            # logger.info("mean_absolute_error: ", mean_absolute_error)

            losses.update(loss.item(), inputs.size(0))
            val_MAE.update(mean_absolute_error.item(), inputs.size(0))
    val_acc = accept_item / all_items
    return losses.avg, val_MAE.avg, val_acc


def val(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()  # 将模型设置为验证模式
    # 混淆矩阵
    with torch.no_grad():
        confusion_matrix = meter.ConfusionMeter(args.num_classes)
        for _, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            targets_onehot = torch.zeros(targets.size(0), args.num_classes).cuda()
            targets_onehot.scatter_(1, targets.view(-1, 1), 1)
            targets_onehot = targets_onehot.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            confusion_matrix.add(outputs.data.squeeze(), targets_onehot.data.long().squeeze())
            acc1 = accuracy(outputs.data, targets.data)

            # compute accuracy by confusion matrix
            # cm_value = confusion_matrix.value()
            # acc2 = 0
            # for i in range(args.num_classes):
            #     acc2 += 100. * cm_value[i][i]/(cm_value.sum())

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            val_acc.update(acc1.item(), inputs.size(0))
    return losses.avg, val_acc.avg


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def test(use_cuda):
    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    if args.only_inference:
        test_set = data_gen.TestDataset_folder_as_input(root=args.test_txt_path, transform=transformations['test'])
    else:
        test_set = data_gen.TestDataset(root=args.test_txt_path, transform=transformations['test'])
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    # load model
    model = make_model(args)

    if args.model_path:
        # 加载模型
        model.load_state_dict(torch.load(args.model_path))

    if use_cuda:
        model.cuda()

    # evaluate
    y_pred = []
    y_true = []
    img_paths = []
    with torch.no_grad():
        model.eval()  # 设置成eval模式
        for (inputs, targets, paths) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            img_paths.extend(list(paths))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)  # (16,2)
            # dim=1 表示按行计算 即对每一行进行softmax
            # probability = torch.nn.functional.softmax(outputs,dim=1)[:,1].tolist()
            # probability = [1 if prob >= 0.5 else 0 for prob in probability]
            # 返回最大值的索引
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            y_pred.extend(probability)
        print("y_pred=", y_pred)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("accuracy=", accuracy)
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print("confusion_matrix=", confusion_matrix)
        print(metrics.classification_report(y_true, y_pred))
        # fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
        # print("roc-auc score=", metrics.roc_auc_score(y_true, y_pred))
        # ---------kkuhn-block------------------------------ todo. It's only for pig dataset spliting. If you see it, comment it.
        pig_face_folder = Path(r"D:\ANewspace\code\pig_face_weight_correlation\datasets\pig_face_only")
        delete_folders(pig_face_folder)
        create_folders(pig_face_folder)

        # delete_folders("rubb", "rubb/1", "rubb/0")
        # create_folders("rubb", "rubb/1", "rubb/0")
        # ---------kkuhn-block------------------------------

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                shutil.copy(os.path.join(args.test_txt_path, img_paths[i]), pig_face_folder)
            # else:
            #     shutil.copy(os.path.join(args.test_txt_path, img_paths[i]), "rubb/0")

        res_dict = {
            'img_path': img_paths,
            'label': y_true,
            'predict': y_pred,

        }
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv, index=False)
        print(f"write to {args.result_csv} succeed ")


def test_for_regression(use_cuda):
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    if args.only_inference:
        test_set = data_gen.TestDataset_folder_as_input(root=args.test_txt_path, transform=transformations['test'])
    else:
        test_set = data_gen.TestDataset(root=args.test_txt_path, transform=transformations['test'])
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    model = make_regression_model(args.model_path)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    if use_cuda:
        model.cuda()

    y_pred = []
    y_true = []
    img_paths = []
    with torch.no_grad():
        model.eval()
        for (inputs, targets, paths) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            img_paths.extend(list(paths))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = model(inputs) * loss_bias
            # convert to int
            outputs = [int(elm.data.cpu().numpy().squeeze()) for elm in outputs]

            y_pred.extend(outputs)
        print("y_pred=", y_pred)

        # if error is less than 10, then it's ok.
        error = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]
        ok_list = [i for i in range(len(y_true)) if error[i] < 10]
        precision = len(ok_list) / len(y_true)
        # print("precision=", precision)
        # print("error_mean=", np.mean(error))
        # print("error_std=", np.std(error))
        # print("error_max=", np.max(error))
        # print("error_min=", np.min(error))
        # print("error_median=", np.median(error))
        # print("error_var=", np.var(error))

        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
        print("accuracy=", mean_absolute_error)

        res_dict = {
            'img_path': img_paths,
            'label': y_true,
            'predict': y_pred,

        }
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv, index=False)
        print(f"write to {args.result_csv} succeed ")


def test_only_for_pig_dataset(use_cuda):  # kuhn edited
    """
    filter out the images with incomplete pig face.
    """
    # ---------kkuhn-block------------------------------ pig face folder
    total_number = 1000
    tbar = tqdm(total=total_number)
    pig_face_folder = Path(args.output_path)
    delete_folders(pig_face_folder)
    create_folders(pig_face_folder)
    # ---------kkuhn-block------------------------------
    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    # test_set = data_gen.TestDataset_folder_as_input(root=args.test_txt_path, transform=transformations['test'])
    test_set = data_gen.TestDataset(root=args.test_txt_path, transform=transformations['test'])
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    # load model
    model = make_model(args)

    if args.model_path:
        # 加载模型
        model.load_state_dict(torch.load(args.model_path))

    if use_cuda:
        model.cuda()

    # evaluate
    y_pred = []
    y_true = []
    img_paths = []
    with torch.no_grad():
        model.eval()  # 设置成eval模式
        for (inputs, targets, paths) in test_loader:
            # for (inputs, targets, paths) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            img_paths.extend(list(paths))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)  # (16,2)
            # dim=1 表示按行计算 即对每一行进行softmax
            # probability = torch.nn.functional.softmax(outputs,dim=1)[:,1].tolist()
            # probability = [1 if prob >= 0.5 else 0 for prob in probability]
            # 返回最大值的索引
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            if probability.shape == ():
                probability = [int(probability)]

            y_pred.extend(probability)
            # ---------kkuhn-block------------------------------ save the images if complete pig head is detected.
            prob_list = probability if isinstance(probability, list) else probability.tolist()
            # prob_list = probability.tolist() if isinstance(probability)
            for i in range(len(prob_list)):
                if prob_list[i] == 1:
                    targetPath = Path(args.test_txt_path)
                    if args.test_txt_path.endswith('.txt'):
                        targetPath = Path(args.test_txt_path).parent
                    shutil.copy(os.path.join(targetPath, paths[i]), pig_face_folder)
                    tbar.update(1)
                    if tbar.n == total_number:
                        exit()
            # ---------kkuhn-block------------------------------

        # ---------kkuhn-block------------------------------ metrics
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("accuracy=", accuracy)
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print("confusion_matrix=", confusion_matrix)
        print(metrics.classification_report(y_true, y_pred))
        # ---------kkuhn-block------------------------------

        res_dict = {
            'img_path': img_paths,
            'label': y_true,
            'predict': y_pred,

        }
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv, index=False)
        print(f"write to {args.result_csv} succeed ")


class getOutofLoop(Exception):
    pass


def generate_pig_face_only_data_for_regresssion():
    original_agg_pig_dataset = Path(r"D:\ANewspace\code\pig_face_weight_correlation\datasets\pig_aggregated")
    new_agg_pig_complete = Path(r"d:\ANewspace\code\pig_face_weight_correlation\datasets\new_agg_complete")

    delete_folders(new_agg_pig_complete)
    create_folders(new_agg_pig_complete)

    # ---------kkuhn-block------------------------------ load model
    model = make_model(args)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    # ---------kkuhn-block------------------------------

    def dataLoad_modelInference_imageSave(datafolder_path, savefolder_path):
        transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
        test_set = data_gen.TestDataset_folder_as_input(root=datafolder_path, transform=transformations['test'])
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        with torch.no_grad():
            model.eval()
            for (inputs, targets, paths) in test_loader:
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs = model(inputs)  # (16,2)

                probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
                # ---------kkuhn-block------------------------------ save the images if complete pig head is detected.
                probability.tolist()
                prob_list = probability.tolist()
                prob_list = [prob_list] if isinstance(prob_list, int) else prob_list  # in case that prob_list has only one element.
                for i in range(len(prob_list)):
                    if prob_list[i] == 1:
                        shutil.copy(os.path.join(args.test_txt_path, paths[i]), savefolder_path)
                        tbar.update(1)
                        if tbar.n == enough_num:
                            raise getOutofLoop()
                # ---------kkuhn-block------------------------------

    for id_folder in original_agg_pig_dataset.iterdir():
        enough_num = 100
        tbar = tqdm(total=enough_num)
        save_folder = new_agg_pig_complete / id_folder.name
        create_folders(save_folder)
        try:
            dataLoad_modelInference_imageSave(id_folder, save_folder)
        except getOutofLoop:
            pass


if __name__ == "__main__":
    # main()
    # 划分数据集
    # data_gen.Split_datatset(args.dataset_txt_path, args.train_txt_path, args.test_txt_path)
    # data_gen.Split_datatset(args.train_txt_path, args.train_txt_path, args.val_txt_path)
    logger.info("-------------------------------------------------- all is well!")
    if args.mode == 'train':
        main()
    else:
        # test(use_cuda)
        # test_only_for_pig_dataset(use_cuda)
        test_for_regression(use_cuda)

        # generate_pig_face_only_data_for_regresssion()
        pass
