import csv
import os
import shutil
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PATIENT_DIR = 'PatientData/'
PATIENT_PREFIX = 'Patient'
LABELLED_DATA_DIR = PATIENT_DIR + 'LabelledData/'
IMAGE_EXTENSION = '.png'
IMAGE_FILE_SUFFIX = 'thresh' + IMAGE_EXTENSION
LABEL_FILE_EXTENSION = '.csv'
LABEL_FILE_SUFFIX = 'Labels' + LABEL_FILE_EXTENSION
FILE_NAME_SEPERATOR = '_'
IC_HEADER = 'IC'
LABEL_HEADER = 'Label'
MODEL_NAME = 'final_model'
IMAGE_HEIGHT = 432
IMAGE_WIDTH = 432
BATCH_SIZE = 8


# utility function to remove file extension form file name
def remove_file_extension(file_name):
    return os.path.splitext(file_name)[0]


# utility function to join directory path with file name
def join_path(dir, filename):
    return os.path.join(dir, filename)


# utlity function to clear all contents of given directory
def init_empty_dirs(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path)


# utility function to write data to csv file
def write_to_csv_file(file_path, header, rows):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# utility function to read data from csv file
def read_from_csv_file(file_path):
    csv_dict = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = IC_HEADER + FILE_NAME_SEPERATOR + row[IC_HEADER] + FILE_NAME_SEPERATOR + remove_file_extension(IMAGE_FILE_SUFFIX)
                csv_dict[key] = 1 if row[LABEL_HEADER] != '0' else 0
    else:
        print(f'Error file: {file_path} does not exist!')

    #print(len(csv_dict))
    return csv_dict


# function to get training and validation dataset
def get_train_and_test_dataset():
    print("Generating training and valdiation datasets")
    data_generator = ImageDataGenerator(
        validation_split=0.5,
        rescale=1/255,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True)
    train_dataset = data_generator.flow_from_directory(
        LABELLED_DATA_DIR,
        subset='training',
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        interpolation='bilinear',
        keep_aspect_ratio=True,
        shuffle=True,
        batch_size = BATCH_SIZE,
        class_mode = 'binary')
                                         
    test_dataset = data_generator.flow_from_directory(
        LABELLED_DATA_DIR,
        subset='validation',
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        interpolation='bilinear',
        keep_aspect_ratio=True,
        shuffle=True,
        batch_size = BATCH_SIZE,
        class_mode = 'binary')
    return train_dataset, test_dataset


# function to get CNN model
def get_cnn_model():
    KERNEL = (3, 3)
    pool_size = (3, 3)
    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32, KERNEL, activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, 3)))
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))

    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(64, KERNEL, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))

    # Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(128, KERNEL, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))

    # Convolutional layer and maxpool layer 4
    model.add(keras.layers.Conv2D(128, KERNEL, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=pool_size))

    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())

    # Add a dropout layer
    model.add(keras.layers.Dropout(rate=0.3))

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    model.add(keras.layers.Dense(512, activation='relu'))

    # Output layer with single neuron which gives 0 for Cat or 1 for Dog 
    #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
                tf.keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='Precision'),
                tf.keras.metrics.SensitivityAtSpecificity(0.5, name='Sensitivity'),
                tf.keras.metrics.SpecificityAtSensitivity(0.5, name='Specificity')
            ])
    print(model.summary())
    return model


# function to read all the brain images form all the sub directories under the given directory
# and move them to class folder based on labels
def read_and_organize_image_data(image_dir, dir_prefix):
    print("Reading and oragnizing image data with labels")
    init_empty_dirs(join_path(LABELLED_DATA_DIR, 'Noise/'))
    init_empty_dirs(join_path(LABELLED_DATA_DIR, 'RNN/'))

    # get list of all sub directories under Slices folder    
    image_dirs = [dir for dir in os.listdir(image_dir) if (os.path.isdir(join_path(image_dir, dir)) and dir.__contains__(dir_prefix))]

    for dir in image_dirs:
        sub_dir_path = join_path(image_dir, dir)
        labels_dict = read_from_csv_file(join_path(image_dir, dir + FILE_NAME_SEPERATOR + LABEL_FILE_SUFFIX))
        if len(labels_dict) == 0:
            print("Error labels not found!")
        else:
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith(IMAGE_FILE_SUFFIX):
                    label = 'Noise' if labels_dict[remove_file_extension(file_name)] == 0 else 'RNN'
                    new_file_name = dir + FILE_NAME_SEPERATOR + file_name
                    shutil.copy(join_path(sub_dir_path, file_name), join_path(LABELLED_DATA_DIR + label, new_file_name))
    print("Labelled training data ready")


def main():
    read_and_organize_image_data(PATIENT_DIR, PATIENT_PREFIX)
    train_dataset, test_dataset = get_train_and_test_dataset()
    print(f"Model labels: {train_dataset.class_indices}")
    model = get_cnn_model()
    
    print('Training CNN model started')
    # """ 
    model.fit(
        train_dataset,
        batch_size=BATCH_SIZE,
        epochs=15,
        verbose=1,
        validation_data=test_dataset,
        steps_per_epoch=train_dataset.samples/BATCH_SIZE)
    # """
    print('Training CNN model completed successfully')

    model.save(MODEL_NAME)
    print(f'Model saved as: {MODEL_NAME}')

if __name__ == '__main__':
    main()