import classification
import numpy as np
import os
import shutil


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


TEST_DATA_DIR = 'testPatient/test_Data/'
TEST_LABELS_FILE = 'testPatient/test_Labels.csv'
RESULT_LABELS_FILE = 'Results.csv'
LABELLED_DATA_DIR = 'testPatient/LabelledData/'
TEST_PREFIX = 'test'
LABEL_FILE_EXTENSION = '.csv'
LABEL_FILE_SUFFIX = 'Labels' + LABEL_FILE_EXTENSION
NOISE = 'Noise'
RNN = 'RNN'


# function to read all the brain images form test data directory
# and move them to class folder based on labels
def read_and_organize_image_data(test_data_dir, labels_file):
    print("Reading and oragnizing image data with labels")
    classification.init_empty_dirs(classification.join_path(LABELLED_DATA_DIR, NOISE))
    classification.init_empty_dirs(classification.join_path(LABELLED_DATA_DIR, RNN))

    labels_dict = classification.read_from_csv_file(labels_file)
    if len(labels_dict) == 0:
        print("Error labels not found!")
    else:
        for file_name in os.listdir(test_data_dir):
            if file_name.endswith(classification.IMAGE_FILE_SUFFIX):
                label = NOISE if labels_dict[classification.remove_file_extension(file_name)] == 0 else RNN
                prefix = classification.IC_HEADER + classification.FILE_NAME_SEPERATOR
                suffix = classification.FILE_NAME_SEPERATOR + classification.IMAGE_FILE_SUFFIX
                new_file_name = file_name
                if new_file_name.startswith(prefix):
                    new_file_name = new_file_name[len(prefix):]
                if new_file_name.endswith(suffix):
                    new_file_name = new_file_name[:-len(suffix)]
                    new_file_name = new_file_name + classification.IMAGE_EXTENSION
                shutil.copy(classification.join_path(test_data_dir, file_name), classification.join_path(LABELLED_DATA_DIR + label, new_file_name))
    print("Labelled training data ready")


# function to load labelled dataset using generator
def load_test_dataset_gen(labelled_data_dir):
    print(f'Loading labelled test dataset using Keras ImagDataGenerator from the dir: {LABELLED_DATA_DIR}')
    test_datagen = ImageDataGenerator(
        rescale=classification.RESCALE_FACTOR,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_dataset = test_datagen.flow_from_directory(
        labelled_data_dir,
        target_size=(classification.IMAGE_HEIGHT, classification.IMAGE_WIDTH),
        interpolation='bilinear',
        batch_size = classification.BATCH_SIZE,
        class_mode = 'binary')
    return test_dataset


# function to load labelled dataset manually
def load_test_dataset_man(labelled_data_dir):
    print(f'Loading labelled test dataset manually from the dir: {LABELLED_DATA_DIR}')
    noise = []
    noise_dir = classification.join_path(labelled_data_dir, NOISE)
    for file_name in os.listdir(noise_dir):
        if file_name.endswith(classification.IMAGE_EXTENSION):
            image = load_img(
                    classification.join_path(noise_dir, file_name),
                    target_size=(classification.IMAGE_WIDTH, classification.IMAGE_HEIGHT))
            image = np.array(image)
            image = image * classification.RESCALE_FACTOR
            image = image.reshape(1, classification.IMAGE_WIDTH, classification.IMAGE_HEIGHT, 3)
            noise.append((classification.remove_file_extension(file_name), image))

    rnn = []
    rnn_dir = classification.join_path(labelled_data_dir, RNN)
    for file_name in os.listdir(rnn_dir):
        if file_name.endswith(classification.IMAGE_EXTENSION):
            image = load_img(
                    classification.join_path(rnn_dir, file_name),
                    target_size=(classification.IMAGE_WIDTH, classification.IMAGE_HEIGHT))
            image = np.array(image)
            image = image * classification.RESCALE_FACTOR
            image = image.reshape(1, classification.IMAGE_WIDTH, classification.IMAGE_HEIGHT, 3)
            rnn.append((classification.remove_file_extension(file_name), image))

    return noise, rnn


# function to predictions for test data
def get_predictions(model, dataset):
    actual_results = []
    for data in dataset:
        actual_result = model.predict(
            x=data[1],
            batch_size=classification.BATCH_SIZE,
            verbose=0)
        actual_results.append(actual_result)
    actual_results = [1 if x>0.5 else 0 for x in actual_results]
    return actual_results


# function to write output labels to csv:
def write_result_labels_to_csv(dataset, labels):
    print(f'Writing prediction labels to {RESULT_LABELS_FILE}')
    rows = []
    index = 0
    while index < len(dataset):
        rows.append([dataset[index][0], labels[index]])
        index += 1

    sorted(rows)
    classification.write_to_csv_file(RESULT_LABELS_FILE, [classification.IC_HEADER + '_Number', classification.LABEL_HEADER], rows)

def main():
    print(f'Laoding saved model: {classification.MODEL_NAME}')
    model = keras.models.load_model(classification.MODEL_NAME)
    read_and_organize_image_data(TEST_DATA_DIR, TEST_LABELS_FILE)
    """
    test_dataset = load_test_dataset_gen(LABELLED_DATA_DIR)
    print(f"Model labels: {test_dataset.class_indices}")
    metrics = model.evaluate(
        test_dataset,
        batch_size=classification.BATCH_SIZE,
        verbose=1,
        return_dict=True) 
    print("Model performance: ")
    for metric_name, metric_value in metrics.items():
        if metric_name != 'loss':
            print(f'{metric_name} = {metric_value*100}')
    """

    noise, rnn = load_test_dataset_man(LABELLED_DATA_DIR)
    noise_pred_results = get_predictions(model, noise)
    print(noise_pred_results)
    rnn_pred_results = get_predictions(model, rnn)
    print(rnn_pred_results)
    write_result_labels_to_csv(noise+rnn, noise_pred_results+rnn_pred_results)


if __name__ == '__main__':
    main()