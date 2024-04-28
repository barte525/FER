import os
import shutil

import Augmentor
from sklearn.model_selection import train_test_split

from config import DIR_NAMES


def train_val_split(train_dir: str, val_dir: str) -> None:
    os.makedirs(val_dir, exist_ok=True)

    for folder_name in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder_name)
        if os.path.isdir(folder_path):
            val_subfolder = os.path.join(val_dir, folder_name)
            os.makedirs(val_subfolder, exist_ok=True)
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     os.path.isfile(os.path.join(folder_path, f))]
            train_files, validation_files = train_test_split(files, test_size=0.1, random_state=42)

            for file in validation_files:
                shutil.move(file, os.path.join(val_subfolder, os.path.basename(file)))
                

# add dataset handling
def augment(emotion: str):
    p = Augmentor.Pipeline(f'{DIR_NAMES["TEST_DIR"]}/{emotion}',
                           output_directory=f'../../{DIR_NAMES["AUG_DIR"]}/{emotion}')
    # p.random_distortion(probability=0.5, grid_height=16, grid_width=16, magnitude=8)
    p.flip_top_bottom(probability=0.5)
    p.sample(100)


def copy_augmented_to_train_aug(emotion: str):
    files = [os.path.join(f'{DIR_NAMES["AUG_DIR"]}/{emotion}', f) for f in
             os.listdir(f'{DIR_NAMES["AUG_DIR"]}/{emotion}')]
    for file in files:
        shutil.copy(file, os.path.join(f'{DIR_NAMES["AUG_TRAIN_DIR"]}/{emotion}', os.path.basename(file)))
