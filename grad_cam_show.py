import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms

from build_net import make_regression_model, make_model_for_24_classes
# from utils import GradCAM, show_cam_on_image
from grad_cam import GradCAM, show_cam_on_image
from args import args
import timm

def main():
    model = make_model_for_24_classes(args)
    target_layers = [model.layer4[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def grad_show_per_img(img_path, save_folder):
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # [N, C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category_id)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.savefig(os.path.join(save_folder, img_path.name))
        # plt.show()

    #---------kkuhn-block------------------------------ param settings
    img_folder = Path("datasets/exact_face_only_cleaned_train_val/val")
    save_folder = Path("rubb/grad_show")
    target_category_id = 0 # the mapping between id and weight is in the dataset path
    #---------kkuhn-block------------------------------
    delete_folders(save_folder)
    create_folders(save_folder)
    for img_path in img_folder.glob("*"):
        grad_show_per_img(img_path, save_folder)


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__':
    main()
