import os
from pathlib import Path


def generate_data_info_txt_for_pig_dataset():
    data_path = Path(r"datasets/pig_if_complete")
    f_train = open(data_path / "train.txt", "w")
    f_val = open(data_path / "val.txt", "w")
    for folder_path in data_path.glob("*"):
        if folder_path.is_dir():
            for class_folder_path in folder_path.glob("*"):
                if class_folder_path.is_dir():
                    for img_path in class_folder_path.glob("*.jpg"):
                        if img_path.is_file():
                            class_id = "1" if class_folder_path.name == "complete" else "0"
                            if folder_path.name == "train":
                                f_train.write(str(img_path) + "," + class_id + "\n")
                            else:
                                f_val.write(str(img_path) + "," + class_id + "\n")
                            # if folder_path.name == "train":
                            #     f_train.write(str(img_path)[3:] + "," + class_id + "\n")
                            # else:
                            #     f_val.write(str(img_path)[3:] + "," + class_id + "\n")

    f_train.close()
    f_val.close()

    # for root, dirs, files in os.walk(data_path):
    #     for file in files:
    #         if file.endswith(".jpg"):
    #             if "train" in root:
    #                 f_train.write(os.path.join(root, file) + "\n")
    #             else:
    #                 f_val.write(os.path.join(root, file) + "\n")
    #
    #


def get_classes_of_pig_data():
    train_txt_path = r"datasets/exact_face_only_cleaned_train_val/train.txt"
    val_txt_path = r"datasets/exact_face_only_cleaned_train_val/val.txt"
    weight_class = []
    id2weight = {0: 206, 1: 214, 2: 223, 3: 230, 4: 235, 5: 237, 6: 239, 7: 240, 8: 243, 9: 249, 10: 252, 11: 253, 12: 254, 13: 260, 14: 267, 15: 273, 16: 276, 17: 283, 18: 287, 19: 290, 20: 293, 21: 295, 22: 315, 23: 321}
    weight2id = {v: k for k, v in id2weight.items()}

    train_txt_for_classification_path = r"datasets/exact_face_only_cleaned_train_val/train_for_classification.txt"
    val_txt_for_classification_path = r"datasets/exact_face_only_cleaned_train_val/val_for_classification.txt"
    f_train = open(train_txt_for_classification_path, "w")
    f_val = open(val_txt_for_classification_path, "w")

    with open(train_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            path = line.split(",")[0]
            weight = int(line.split(",")[1])
            if weight not in weight_class:
                weight_class.append(weight)
            f_train.write(path + "," + str(weight2id[weight]) + "\n")
    with open(val_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            weight = int(line.split(",")[1])
            if weight not in weight_class:
                weight_class.append(weight)
            f_val.write(line.split(",")[0] + "," + str(weight2id[weight]) + "\n")
    f_train.close()
    f_val.close()
    # sort weight_class
    weight_class.sort()
    id2weight = dict(enumerate(weight_class))
    weight2id = {v: k for k, v in id2weight.items()}
    print(id2weight)
    # save id2weight to file
    with open(r"datasets/exact_face_only_cleaned_train_val/id2weight_for_weight_classification.txt", "w") as f:
        for key, value in id2weight.items():
            f.write(str(key) + "," + str(value) + "\n")

    with open(r"datasets/exact_face_only_cleaned_train_val/weight2id_for_weight_classification.txt", "w") as f:
        for key, value in weight2id.items():
            f.write(str(key) + "," + str(value) + "\n")


if __name__ == '__main__':
    # generate_data_info_txt_for_pig_dataset()
    get_classes_of_pig_data()
    pass
