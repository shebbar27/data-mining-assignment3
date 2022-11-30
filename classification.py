import csv
import numpy as np
import os
import pickle
import shutil
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


TEST_DIR = 'testPatient/'
TEST_LABELS_FILE = 'test_Labels.csv'
PATIENT_DIR = 'PatientData/'
LABELLED_DATA_DIR = PATIENT_DIR + 'LabelledData/'
IMAGE_EXTENSION = '.png'
IMAGE_FILE_SUFFIX = 'thresh' + IMAGE_EXTENSION
LABEL_FILE_EXTENSION = '.csv'
LABEL_FILE_SUFFIX = 'Labels' + LABEL_FILE_EXTENSION
FILE_NAME_SEPERATOR = '_'
IC_HEADER = 'IC'
LABEL_HEADER = 'Label'
MODEL_FILE_NAME = 'svm_model.sav'
PATIENT = 'Patient'


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
    return csv_dict


# function to get training and validation dataset
def get_train_and_test_dataset():
    print("Generating training and valdiation datasets")
    data_generator = ImageDataGenerator(validation_split=0.2, rescale=1/255)
    train_dataset = data_generator.flow_from_directory(
                                            LABELLED_DATA_DIR,
                                            subset='training',
                                            target_size=(500,600),
                                            interpolation='bilinear',
                                            keep_aspect_ratio=True,
                                            shuffle=True,
                                            batch_size = 32,
                                            class_mode = 'binary'
                                        )
                                         
    test_dataset = data_generator.flow_from_directory(
                                            LABELLED_DATA_DIR,
                                            subset='validation',
                                            target_size=(500,600),
                                            interpolation='bilinear',
                                            keep_aspect_ratio=True,
                                            shuffle=True,
                                            batch_size = 32,
                                            class_mode = 'binary'
                                        )
    return train_dataset, test_dataset



# function to read all the brain images form all the sub directories under the given directory
# and move them to class folder based on labels
def read_and_organize_image_data(image_dir):
    print("Reading and oragnizing image data with labels")
    init_empty_dirs(join_path(LABELLED_DATA_DIR, '0/'))
    init_empty_dirs(join_path(LABELLED_DATA_DIR, '1/'))

    # get list of all sub directories under Slices folder    
    image_dirs = [dir for dir in os.listdir(image_dir) if (os.path.isdir(join_path(image_dir, dir)) and dir.__contains__(PATIENT))]

    for dir in image_dirs:
        sub_dir_path = join_path(image_dir, dir)
        labels_dict = read_from_csv_file(join_path(image_dir, dir + FILE_NAME_SEPERATOR + LABEL_FILE_SUFFIX))
        if len(labels_dict) == 0:
            print("Error labels not found!")
        else:
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith(IMAGE_FILE_SUFFIX):
                    label = str(labels_dict[remove_file_extension(file_name)])
                    new_file_name = dir + FILE_NAME_SEPERATOR + file_name
                    shutil.copy(join_path(sub_dir_path, file_name), join_path(LABELLED_DATA_DIR + label, new_file_name))
    print("Labelled training data ready")


# function to save model
def save_model(model, model_filename):
    print("Saving model to file: " + model_filename)
    pickle.dump(model, open(model_filename, 'wb'))


def main():
    read_and_organize_image_data(PATIENT_DIR)
    train_dataset, test_dataset = get_train_and_test_dataset()
    
if __name__ == '__main__':
    main()