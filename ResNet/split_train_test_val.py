import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def main():

    random.seed(0)


    val_rate=0.12
    test_rate = 0.1

    cwd = os.getcwd()
    data_root = os.path.join(cwd, "data")
    origin_image_path = os.path.join(data_root, "ravdess/spectrogram")
    assert os.path.exists(origin_image_path), "path '{}' does not exist.".format(origin_image_path)

    emotion_class = [cla for cla in os.listdir(origin_image_path)
                    if os.path.isdir(os.path.join(origin_image_path, cla))]


    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in emotion_class:

        mk_file(os.path.join(train_root, cla))

    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in emotion_class:
        mk_file(os.path.join(val_root, cla))

    test_root = os.path.join(data_root, "test")
    mk_file(test_root)
    for cla in emotion_class:
        mk_file(os.path.join(test_root, cla))

    for cla in emotion_class:
        cla_path = os.path.join(origin_image_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        eval_index = random.sample(images, k=int(num*val_rate))
        rest = tuple(set(images)-set(eval_index))
        test_index = random.sample(rest, k=int(num*test_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            elif image in test_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
