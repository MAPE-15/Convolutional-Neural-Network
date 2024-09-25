import os
import numpy as np
from PIL import Image


DATA_DIR = os.path.join(os.getcwd(), "data", "images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")


def load_data(folder: str, image_size: tuple[str, str]) -> tuple[np.array, np.array, dict]:

    """
    Load data from the specified folder path.
    This path leeds to a number of folders, these folders are named after a facial expression.
    This facial expressions folder contains images of that facial expression

    Expected folder tree:
    - folder
        - facial_expression_folder1
            - image1
            - image2
            - ...
        - facial_expression_folder2
            - image1
            - image2
            - ...
        - ...
        - facial_expression_folderN
            - image1
            - image2
            - ...

    :param folder: path to the folder containing the facial expression folders, which then contains the images
    :param image_size: size of the images to be loaded
    :return: a tuple,
        where 1st element of the tuple is a 3D numpy array of images, each image is 2D numpy array
        where 2nd element of the tuple is a 1D numpy array of labels for each facial expression folder (labels as: 0, 1, 2, ..., N)
        where 3rd element is a dictionary - label_dict, where key is the label (label as: 0, 1, 2, ..., N) and
         its value is the anme of the facial expression, to know what each label means
    """

    images = []
    labels = []
    label_dict = {}

    for label, facial_expression_folder in enumerate(sorted(os.listdir(folder))):

        facial_expression_folder_path = os.path.join(folder, facial_expression_folder)

        for image in os.listdir(facial_expression_folder_path):
            image_path = os.path.join(facial_expression_folder_path, image)
            img = Image.open(image_path).resize(image_size)
            img = np.array(img) / 255.0  # normalize the image pixels
            # img is a 2D array, in s shape of a image_size

            images.append(img)
            labels.append(label)

        label_dict[label] = facial_expression_folder

    return np.array(images), np.array(labels), label_dict


def to_one_hot(labels: np.array, num_classes: int) -> np.array:
    """
    Create a 2D numpy array of one-hot encoded labels.

    np.eye(): on diagonal is 1, elsewhere is 0
    f.e. np.eye(3) -> [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    then np.eye(3)[[1, 2]] -> [ [0, 1, 0], [0, 0, 1] ]

    :param labels: 1D array of labels
    :param num_classes: number of unique classes
    :return: one-hot encoded labels
    """
    return np.eye(num_classes)[labels]


def main() -> None:
    image_size = (48, 48)

    train_images, train_labels, label_dict = load_data(folder=TRAIN_DIR, image_size=image_size)
    val_images, val_labels, _ = load_data(folder=VALIDATION_DIR, image_size=image_size)

    num_classes = len(list(label_dict.keys()))

    train_labels_one_hot = to_one_hot(train_labels, num_classes)
    val_labels_one_hot = to_one_hot(val_labels, num_classes)

    print("Labels:")
    for label, facial_expression in label_dict.items():
        print(f"\t{facial_expression} -> {label}")



if __name__ == '__main__':
    main()