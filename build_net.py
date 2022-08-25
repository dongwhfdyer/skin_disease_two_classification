""" 
@ author: Qmh
@ file_name: build_net.py
@ time: 2019:11:20:10:04
"""
import torch
from torch import nn
import torchvision.models as models
import models as customized_models
from args import args

# Models
from models import Res

default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


class log_mse_loss(nn.Module):
    def __init__(self):
        super(log_mse_loss, self).__init__()

    def forward(self, input, target):
        return torch.log(torch.pow(input - target, 2))


def make_regression_model(args):
    model = Res.resnet50_output_1()
    ckpt = torch.load(args.model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict, strict=False)
    return model


def make_model(args):
    print("=> creating model '{}'".format(args.arch))
    # 加载预训练模型 
    model = models.resnet50()
    model.load_state_dict(torch.load(r"checkpoints/resnet50-19c8e357.pth"))
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # 最后一层全连接层
    fc_inputs = model.fc.in_features
    if args.if_regression:
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )
        print("=> creating model 1")
    else:
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, args.num_classes),
        )
    return model


if __name__ == '__main__':
    all_model = sorted(name for name in models.__dict__ if not name.startswith("__"))
    print(all_model)
    model = make_model(args)
