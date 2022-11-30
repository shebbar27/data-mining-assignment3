import csv
import cv2
import os


TEST_DIR = 'testPatient/'
TEST_DATA_DIR = 'test_Data/'
TEST_LABELS_DIR = 'test_Labels/'
IMAGE_FILE_SUFFIX = 'thresh.png'
IMAGE_EXTENSION = '.png'


# utility function to remove file extension form file name
def remove_file_extension(file_name):
    return os.path.splitext(file_name)[0]


# utility function to join directory path with file name
def join_path(dir, filename):
    return os.path.join(dir, filename)


# utility function to read all the brain images form the given directory
def read_image_data(image_dir):
    images = []
    for file_name in os.listdir(image_dir):
        if(file_name.endswith(IMAGE_FILE_SUFFIX)):
            brain_image = cv2.imread(join_path(INPUT_DIR, file_name))
            images.append((file_name, brain_image))
    return images


# utility function to write data to csv file
def write_to_csv_file(file_path, header, rows):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# utility function to read data from csv file
def read_from_csv_file(file_path):
    csvDict = {}
    with open(file_path, 'r') as file:
        csvDict = csv.DictReader(file)
    return csvDict

def main():
    NotImplemented


if __name__ == '__main__':
    main()