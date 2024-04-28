from typing import Tuple

from tensorflow.data import Dataset
from tensorflow.keras.preprocessing import image_dataset_from_directory

from config import BATCH_SIZE, DIR_NAMES


def load_data(dataset_name: str, rgb: bool, image_size: Tuple[int, int]) -> Tuple[Dataset, Dataset, Dataset]:
    color = 'grayscale'
    if rgb:
        color = 'rgb'
    train_data = image_dataset_from_directory(
        f'datasets/{dataset_name}/{DIR_NAMES["TRAIN_DIR"]}',
        color_mode=color,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        )

    val_data = image_dataset_from_directory(
        f'datasets/{dataset_name}/{DIR_NAMES["VAL_DIR"]}',
        color_mode=color,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    test_data = image_dataset_from_directory(
        f'datasets/{dataset_name}/{DIR_NAMES["TEST_DIR"]}',
        color_mode=color,
        batch_size=BATCH_SIZE,
        image_size=image_size,
        label_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data
