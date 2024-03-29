import csv
import matplotlib.pyplot as plt
import os
import shutil


from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
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
MODEL_NAME = 'final_model.h5'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16
RESCALE_FACTOR = 1./255
USE_VALIDATION_DATASET = False


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


# function to plot loss and accuracy curves
def plot_loss_and_accuracy_curves(history, plot_validation_info=False):
    print('Plotting loss and accuracy curves')
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy'] if plot_validation_info else None
    val_loss = history.history['val_loss'] if plot_validation_info else None

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    if plot_validation_info:
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Accuracy with Epochs')
    plt.legend()

    plt.savefig('accuracy_with_epochs.png')
    plt.close()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    if plot_validation_info:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss with Epochs')
    plt.legend()

    plt.savefig('loss_with_epochs.png')
    plt.close()


# function to get keras image data generator
def get_image_data_generator(get_validation_data):
    if get_validation_data:
        return ImageDataGenerator(
            validation_split=0.2,
            rescale=RESCALE_FACTOR,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    return ImageDataGenerator(
        rescale=RESCALE_FACTOR,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# function to get training and validation dataset
def get_train_dataset(use_validation_data):
    print("Generating training and valdiation datasets")
    data_generator = get_image_data_generator(use_validation_data)
                                      
    train_dataset = None
    test_dataset = None

    if use_validation_data:
        train_dataset = data_generator.flow_from_directory(
            LABELLED_DATA_DIR,
            subset='training',
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            interpolation='bilinear',
            batch_size = BATCH_SIZE,
            class_mode = 'binary')
        test_dataset = data_generator.flow_from_directory(
            LABELLED_DATA_DIR,
            subset='validation',
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            interpolation='bilinear',
            batch_size = BATCH_SIZE,
            class_mode = 'binary')
    else:
        train_dataset = data_generator.flow_from_directory(
            LABELLED_DATA_DIR,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            interpolation='bilinear',
            batch_size = BATCH_SIZE,
            class_mode = 'binary')
    return train_dataset, test_dataset


# function to get CNN model
def get_cnn_model():
    print('Generating Keras Sequential model for CNN network')
    KERNEL = (3, 3)
    POOL_SIZE = (3, 3)
    model = Sequential()

    # Convolutional layer 1 with maxpooling
    model.add(Conv2D(32, KERNEL, activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, 3)))
    model.add(MaxPool2D(pool_size=POOL_SIZE))

    # Convolutional layer 2 with maxpooling
    model.add(Conv2D(64, KERNEL, activation='relu'))
    model.add(MaxPool2D(pool_size=POOL_SIZE))

    # Convolutional layer 3 with maxpooling
    model.add(Conv2D(128, KERNEL, activation='relu'))
    model.add(MaxPool2D(pool_size=POOL_SIZE))

    # Convolutional layer 4 with maxpooling
    model.add(Conv2D(128, KERNEL, activation='relu'))
    model.add(MaxPool2D(pool_size=POOL_SIZE))

    # This layer flattens the resulting image array to 1D array and add a dropout layer
    model.add(Flatten())
    model.add(Dropout(rate=0.2))

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    model.add(Dense(512, activation='relu'))

    # Output layer with single neuron which gives 0 for Noise or 1 for RNN 
    # Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
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
    train_dataset, validation_dataset = get_train_dataset(USE_VALIDATION_DATASET)
    print(f"Model labels: {train_dataset.class_indices}")
    model = get_cnn_model()

    print('Training CNN model started')
    history = None
    if USE_VALIDATION_DATASET:
        history = model.fit(
            train_dataset,
            batch_size=BATCH_SIZE,
            epochs=50,
            steps_per_epoch=train_dataset.samples/BATCH_SIZE,
            verbose=1,
            validation_data=validation_dataset,
            validation_steps=validation_dataset.samples/BATCH_SIZE)
    else:
        history = model.fit(
            train_dataset,
            batch_size=BATCH_SIZE,
            epochs=50,
            steps_per_epoch=train_dataset.samples/BATCH_SIZE,
            verbose=1)
    print('Training CNN model completed successfully')

    plot_loss_and_accuracy_curves(history, USE_VALIDATION_DATASET)

    model.save(MODEL_NAME)
    print(f'Model saved as: {MODEL_NAME}')

if __name__ == '__main__':
    main()