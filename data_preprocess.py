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



if __name__ == '__main__':
    generate_data_info_txt_for_pig_dataset()
