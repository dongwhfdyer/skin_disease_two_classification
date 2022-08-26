""" 
@ author: Qmh
@ file_name: args.py
@ time: 2019:11:20:11:14
"""
import argparse

# # ---------kkuhn-block------------------------------ binary classification train settings
# data_path = r"datasets/pig_if_complete_0825"
# train_txt_path = r"datasets/pig_if_complete_0825/train.txt"
# val_txt_path = r"datasets/pig_if_complete_0825/val.txt"
# # test_data_path = "datasets/pig_if_complete/train/complete"
# # test_data_path = "datasets/pig_if_complete/train/incomplete"
# test_data_path = "D:\ANewspace\code\pig_face_weight_correlation\datasets\selected_pig_all"
# output_data_path = r"rubb/pig_face_only" # useless here
#
# batch_size = 16
# num_workers = 10
# # batch_size = 4
# # num_workers = 4
# lr = 0.001
# # mode = "test"
# mode = "train"
# # pt_path = r"checkpoints/model_16_9283_9187.pth"
# pt_path = r"checkpoints/resnet50-19c8e357.pth"
# # pt_path = r"models/model_cur.pth"
#
# img_size = 512
# if_regression = False
# optimizer = "adam"
# epochs = 300
# only_inference = False
# save_prefix = "binary_cls"
# # ---------kkuhn-block------------------------------

# # ---------kkuhn-block------------------------------ binary classification test settings
# data_path = r"datasets/pig_if_complete_0825"
# train_txt_path = r"datasets/pig_if_complete_0825/train.txt"
# val_txt_path = r"datasets/pig_if_complete_0825/val.txt"
# # test_data_path = "datasets/pig_if_complete/train/complete"
# # test_data_path = "datasets/pig_if_complete/train/incomplete"
# # test_data_path = "D:\ANewspace\code\pig_face_weight_correlation\datasets\selected_pig_all"
# test_data_path = val_txt_path
# output_data_path = r"rubb/pig_face_only"
# batch_size = 4
# num_workers = 4
# # batch_size = 28
# # num_workers = 10
# lr = 0.001
# mode = "test"
# # pt_path = r"checkpoints/model_16_9283_9187.pth"
# # pt_path = r"models/binary_cls_if_pighead_complete_v1.pth"
# pt_path = r"models/model_11_8290_8970.pth"
#
# img_size = 416
# if_regression = False
# optimizer = "adam"
# epochs = 300
# only_inference = True
# save_prefix = "binary_cls"
# # ---------kkuhn-block------------------------------

# ---------kkuhn-block------------------------------ weight regression train settings
data_path = r"datasets/exact_face_only_cleaned_train_val"
train_txt_path = r"datasets/exact_face_only_cleaned_train_val/train.txt"
val_txt_path = r"datasets/exact_face_only_cleaned_train_val/val.txt"
# test_data_path = "datasets/pig_if_complete/train/complete"
# test_data_path = "datasets/pig_if_complete/train/incomplete"
test_data_path = "D:/ANewspace/code/pig_face_weight_correlation/datasets/selected_pig_all"
output_data_path = r"rubb/pig_face_only" # useless here
batch_size = 16
num_workers = 10
# batch_size = 4
# num_workers = 4

lr = 0.0007
# mode = "test"
mode = "train"
pt_path = r"checkpoints/resnet50-19c8e357.pth"
img_size = 512
if_regression = True
optimizer = "adam"
epochs = 300
only_inference = False
save_prefix = "weight_regression"
# ---------kkuhn-block------------------------------

# # ---------kkuhn-block------------------------------ weight regression test settings
# data_path = r"datasets/exact_face_only_cleaned_train_val"
# train_txt_path = r"datasets/exact_face_only_cleaned_train_val/train.txt"
# val_txt_path = r"datasets/exact_face_only_cleaned_train_val/val.txt"
# # test_data_path = "datasets/pig_if_complete/train/complete"
# # test_data_path = "datasets/pig_if_complete/train/incomplete"
# test_data_path = train_txt_path
# # test_data_path = val_txt_path
# # test_data_path = "D:/ANewspace/code/pig_face_weight_correlation/datasets/selected_pig_all"
# # batch_size = 28
# # num_workers = 10
# batch_size = 4
# num_workers = 4
#
# lr = 0.0007
# mode = "test"
# pt_path = r"models/model_100_0_0.pth"
# img_size = 416
# if_regression = True
# optimizer = "adam"
# epochs = 300
# only_inference = False
# save_prefix = "binary_cls"
# # ---------kkuhn-block------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default=mode, help='train or test')

# datasets 
parser.add_argument('--dataset_path', type=str, default=data_path, help='the path to save imgs')
# parser.add_argument('-dataset_txt_path',type=str,default='./dataset/small_dataset.txt')
parser.add_argument('-train_txt_path', type=str, default=train_txt_path)
parser.add_argument('-test_txt_path', type=str, default=test_data_path)
parser.add_argument('-val_txt_path', type=str, default=val_txt_path)

# optimizer
parser.add_argument('--optimizer', default=optimizer, choices=['sgd', 'rmsprop', 'adam', 'radam'])
parser.add_argument("--lr", type=float, default=lr)
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                    action='store_false',
                    help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                    help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                    help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                    help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# training
parser.add_argument("--checkpoint", type=str, default='./checkpoints')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to save the latest checkpoint')
parser.add_argument("--batch_size", type=int, default=batch_size)
parser.add_argument("--start_epoch", default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=epochs, type=int, metavar='N')

parser.add_argument('--image-size', type=int, default=img_size)
parser.add_argument('--arch', default='resnet50', choices=['resnet34', 'resnet18', 'resnet50'])
parser.add_argument('--num_classes', default=2, type=int)

# model path
parser.add_argument('--model_path', default=pt_path, type=str)
parser.add_argument('--result_csv', default='./result.csv')
parser.add_argument('--if_regression', default=if_regression, help="if regression, then and the model will output one value and will use regression loss.")
parser.add_argument('--num_workers', default=num_workers)
parser.add_argument('--only_inference', default=only_inference, help="if only inference, then the dataset will have no labels")
parser.add_argument("--output_path", default=output_data_path, help="only used when testing. Also, it's optional")
parser.add_argument("--save_prefix", default=save_prefix, help="config save prefix. used to distinguish different task")

args = parser.parse_args()
