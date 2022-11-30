import csv
import cv2
import numpy as np
import os
import pandas as pd
import pickle

from skimage.transform import resize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


TEST_DIR = 'testPatient/'
TEST_LABELS_FILE = 'test_Labels.csv'
TRAIN_DIR = 'PatientData/'
IMAGE_EXTENSION = '.png'
IMAGE_FILE_SUFFIX = 'thresh' + IMAGE_EXTENSION
LABEL_FILE_EXTENSION = '.csv'
LABEL_FILE_SUFFIX = 'Labels' + LABEL_FILE_EXTENSION
FILE_NAME_SEPERATOR = '_'
IC_HEADER = 'IC'
LABEL_HEADER = 'Label'
NAME_HEADER = 'Name'
MODEL_FILE_NAME = 'svm_model.sav'


# utility function to remove file extension form file name
def remove_file_extension(file_name):
    return os.path.splitext(file_name)[0]


# utility function to join directory path with file name
def join_path(dir, filename):
    return os.path.join(dir, filename)


# utility function to write data to csv file
def write_to_csv_file(file_path, header, rows):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# utility function to read data from csv file
def read_from_csv_file(dir, filename):
    csvDict = {}
    with open(join_path(dir, filename), 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = filename
            key = key.removesuffix(LABEL_FILE_SUFFIX) + IC_HEADER + FILE_NAME_SEPERATOR + row[IC_HEADER] + FILE_NAME_SEPERATOR + remove_file_extension(IMAGE_FILE_SUFFIX)
            csvDict[key] = 1 if row[LABEL_HEADER] != '0' else 0
    return csvDict


# function to read all the brain images form all the sub directories under the given directory
def read_image_data(image_dir):
    print("Reading image data")
    images = {}

    # get list of all sub directories under Slices folder    
    image_dirs = [dir for dir in os.listdir(image_dir) if os.path.isdir(join_path(image_dir, dir))]

    for dir in image_dirs:
        sub_dir_path = join_path(image_dir, dir)
        for file_name in os.listdir(sub_dir_path):
            if file_name.endswith(IMAGE_FILE_SUFFIX):
                brain_image = cv2.imread(join_path(sub_dir_path, file_name))
                key = dir + FILE_NAME_SEPERATOR + remove_file_extension(file_name)
                images[key] = brain_image
    return images


# function to read lables from the given directory
def read_data_labels(labels_dir):
    print("Reading label data")
    data_labels = []
    for file_name in os.listdir(labels_dir):
        if file_name.endswith(LABEL_FILE_EXTENSION):
            csvDict = read_from_csv_file(labels_dir, file_name)
            data_labels.extend(list(csvDict.items()))
    return data_labels


# function to save model
def save_model(model, model_filename):
    print("Saving model to file: " + model_filename)
    pickle.dump(model, open(model_filename, 'wb'))


# function to combime images tuple list with labels dictionary into dataframe
def combine_images_and_labels(images, labels):
    images_data_arr = []
    label_data_arr = []
    count = 0
    h = 0
    w = 0
    print("Combining data into DataFrame")
    for item in labels:
        key = item[0]
        label = item[1]
        if key in images:
            image = images[key]
            if count == 0:
                h, w, _ = image.shape
                # print(h, w)
                count += 1
            else:
                h_i, w_i, _ = image.shape
                if h_i != h or w_i != w:
                    image = resize(image, (h, w, 3))
                    # print(h_i, w_i)
        
            images_data_arr.append(image.flatten())
            label_data_arr.append(label)

    images_features = np.array(images_data_arr)
    labels_col = np.array(label_data_arr)
    df = pd.DataFrame(images_features)
    df[LABEL_HEADER] = labels_col

    # print(df.head(n=5))
    # print(df.info())
    print("DataFrame created sucessfully")
    return df


# function to get SVM model
def get_svm_model(enable_probability=True):
    param_grid = {
                    'C': [0.1, 1, 10, 100], 
                    'gamma': [0.0001, 0.001, 0.1, 1],
                    'kernel': ['rbf', 'poly']
                }

    svc = svm.SVC(probability=enable_probability)
    model = GridSearchCV(svc, param_grid, n_jobs=-1, verbose=5)
    return model


def main():
    images = read_image_data(TRAIN_DIR)
    labels = read_data_labels(TRAIN_DIR)
    df = combine_images_and_labels(images, labels)
    labelled_df = df.copy()

    # supervised machine learning using SVM model
    x=labelled_df.iloc[:,:-1]
    y=labelled_df.iloc[:,-1]
    model = get_svm_model()
    
    print("Generating training and testing dataset")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    
    print("Training model started")
    model.fit(x_train, y_train)
    print("Training model started")

    print("Evaluating model started")
    y_pred = model.predict(x_test)
    print("Evaluating model completed")

    print("Model Performance: ")
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

    save_model(model, MODEL_FILE_NAME)


if __name__ == '__main__':
    main()