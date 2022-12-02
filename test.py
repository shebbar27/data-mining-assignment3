import classification
import numpy as np
import os
import shutil


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TEST_DATA_DIR = 'testPatient/test_Data/'
TEST_LABELS_FILE = 'testPatient/test_Labels.csv'
LABELLED_DATA_DIR = 'testPatient/LabelledData/'
TEST_PREFIX = 'test'
LABEL_FILE_EXTENSION = '.csv'
LABEL_FILE_SUFFIX = 'Labels' + LABEL_FILE_EXTENSION


# function to read all the brain images form test data directory
# and move them to class folder based on labels
def read_and_organize_image_data(image_dir, labels_file):
    print("Reading and oragnizing image data with labels")
    classification.init_empty_dirs(classification.join_path(LABELLED_DATA_DIR, 'Noise/'))
    classification.init_empty_dirs(classification.join_path(LABELLED_DATA_DIR, 'RNN/'))

    labels_dict = classification.read_from_csv_file(labels_file)
    if len(labels_dict) == 0:
        print("Error labels not found!")
    else:
        for file_name in os.listdir(image_dir):
            if file_name.endswith(classification.IMAGE_FILE_SUFFIX):
                label = 'Noise' if labels_dict[classification.remove_file_extension(file_name)] == 0 else 'RNN'
                shutil.copy(classification.join_path(image_dir, file_name), classification.join_path(LABELLED_DATA_DIR + label, file_name))
    print("Labelled training data ready")


# function to load test dataset
def load_test_dataset(test_data_dir):
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_dataset = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(classification.IMAGE_HEIGHT, classification.IMAGE_WIDTH),
        interpolation='bilinear',
        batch_size = classification.BATCH_SIZE,
        class_mode = 'binary')
    return test_dataset


def main():
    read_and_organize_image_data(TEST_DATA_DIR, TEST_LABELS_FILE)
    test_dataset = load_test_dataset(LABELLED_DATA_DIR)
    print(f"Model labels: {test_dataset.class_indices}")

    model = keras.models.load_model(classification.MODEL_NAME)
    metrics = model.evaluate(
        test_dataset,
        batch_size=classification.BATCH_SIZE,
        verbose=1,
        return_dict=True) 
    print("Model performance: ")
    for metric_name, metric_value in metrics.items():
        if metric_name != 'loss':
            print(f'{metric_name} = {metric_value*100}')

if __name__ == '__main__':
    main()