import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential


def plot_class_distribution(data_dir: str, dataset_name: str, data_dir_type: str) -> None:
    class_counts = {}
    for root, dirs, files in os.walk(data_dir):
        labels = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if labels:
            break
    for label in labels:
        path = os.path.join(data_dir, label)
        class_counts[label] = len(os.listdir(path))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(f'Class Distribution for {dataset_name} {data_dir_type} set')
    plt.xticks(rotation=45)
    plt.ylabel('Number of images')
    plt.show()


# Calculate class weights to handle class imbalance - cos sie zdaje tu nie dzialac
def calculate_class_weights(data_dir):
    # Compute the class weight for each class
    for root, dirs, files in os.walk(data_dir):
        labels = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if labels:
            break
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(zip(np.unique(labels), weights))
    return class_weights


def plot_loss_and_acc(history: dict) -> None:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


class TestDataMetrics:
    def __init__(self, test_data: Dataset, model: Sequential):
        self.__test_data = test_data
        self.__model = model

    @property
    def predictions(self) -> np.ndarray:
        return self.__model.predict(self.__test_data, verbose='silent').argmax(1)

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate([y for _, y in self.__test_data], axis=0).argmax(1)

    @property
    def display_labels(self) -> list:
        return self.__test_data.class_names

    @property
    def accuracy(self) -> float:
        return self.__model.evaluate(self.__test_data, zeverbose='silent')[1]

    def print_classification_report(self) -> None:
        print(classification_report(self.labels, self.predictions, target_names=self.display_labels))

    def print_confusion_matrix(self, normalize: bool = True) -> None:
        cm = confusion_matrix(self.labels, self.predictions)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.subplots(figsize=(9, 9))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=self.display_labels, yticklabels=self.display_labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show(block=False)
