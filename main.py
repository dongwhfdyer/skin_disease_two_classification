""" 
@ author: Qmh
@ file_name: main.py
@ time: 2019:11:20:11:24
"""
import logging
import shutil
import time
from pathlib import Path

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
from build_net import make_model, log_mse_loss, make_regression_model
from utils import get_optimizer, AverageMeter, save_checkpoint, accuracy
import torchnet.meter as meter
import pandas as pd
from sklearn import metrics

# Use CUDA
torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
# use_cuda = False

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

best_acc = 0


def main():
    global best_acc

    time_now = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
    checkpoints_path = os.path.join(args.checkpoint, str(time_now))

    create_folders(checkpoints_path)

    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    train_set = data_gen.Dataset(root=args.train_txt_path, transform=transformations['val_train'])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_set = data_gen.ValDataset(root=args.val_txt_path, transform=transformations['val_test'])
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_regression_model(args)
    # model = make_model(args)
    if use_cuda:
        model.cuda()

    if args.if_regression:
        # criterion = log_mse_loss()
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if use_cuda:
        criterion = criterion.cuda()

    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

    # load checkpoint
    start_epoch = args.start_epoch
    # if args.resume:
    #     print("===> Resuming from checkpoint")
    #     assert os.path.isfile(args.resume),'Error: no checkpoint directory found'
    #     args.checkpoint = os.path.dirname(args.resume)  # 去掉文件名 返回目录
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['best_acc']
    #     start_epoch = checkpoint['epoch']
    #     model.module.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # train
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        if args.if_regression:
            # print('Train Loss: %.8f' % train_loss)
            test_loss, val_acc = val_for_regression(val_loader, model, criterion, epoch, use_cuda)
        else:
            test_loss, val_acc = val(val_loader, model, criterion, epoch, use_cuda)

        scheduler.step(test_loss)

        print(f'train_loss:{train_loss}\t val_loss:{test_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        print(f'Memory used: {mem} GB')
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


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for (inputs, targets) in tqdm(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # kuhn edited
            targets = torch.reshape(targets, (-1, 1))
            # inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        logger.debug(f'outputs:{outputs * 150.0}')
        if args.if_regression:
            loss = criterion(outputs, targets / 150.0)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs.data, targets.data)

        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))

    return losses.avg, train_acc.avg


def val_for_regression(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()
    # 混淆矩阵
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            targets = torch.reshape(targets, (-1, 1))
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets / 150.0)
            acc1 = accuracy(outputs.data, targets.data)

            losses.update(loss.item(), inputs.size(0))
            val_acc.update(acc1.item(), inputs.size(0))
    return losses.avg, val_acc.avg


def val(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()  # 将模型设置为验证模式
    # 混淆矩阵
    confusion_matrix = meter.ConfusionMeter(args.num_classes)
    for _, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        confusion_matrix.add(outputs.data.squeeze(), targets.long())
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
    test_set = data_gen.TestDataset_folder_as_input(root=args.test_txt_path, transform=transformations['test'])
    # test_set = data_gen.TestDataset(root=args.test_txt_path, transform=transformations['test'])
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


def test_only_for_pig_dataset(use_cuda):  # kuhn edited

    # ---------kkuhn-block------------------------------ pig face folder
    total_number = 1000
    tbar = tqdm(total=total_number)
    pig_face_folder = Path(r"D:\ANewspace\code\pig_face_weight_correlation\datasets\pig_face_only")
    delete_folders(pig_face_folder)
    create_folders(pig_face_folder)
    # ---------kkuhn-block------------------------------
    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    test_set = data_gen.TestDataset_folder_as_input(root=args.test_txt_path, transform=transformations['test'])
    # test_set = data_gen.TestDataset(root=args.test_txt_path, transform=transformations['test'])
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
            y_pred.extend(probability)
            # ---------kkuhn-block------------------------------ save the images if complete pig head is detected.
            prob_list = probability.tolist()
            for i in range(len(prob_list)):
                if prob_list[i] == 1:
                    shutil.copy(os.path.join(args.test_txt_path, paths[i]), pig_face_folder)
                    tbar.update(1)
                    if tbar.n == total_number:
                        exit()
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
    if args.mode == 'train':
        main()
    else:
        # test(use_cuda)
        # test_only_for_pig_dataset(use_cuda)

        generate_pig_face_only_data_for_regresssion()
