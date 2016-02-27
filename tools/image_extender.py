import os.path
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

DESIRED_DIMENSION = 233
IMAGE_MULTIPLIER = 23
MAX_FILES_IN_SUBFOLDER = 98000
DESTINATION_FOLDER = r'C:\Downloads\person_detection\ready_img_ext\mountains'


def save_images(images, file_name, subfolder_path):

    '''
    :param images: list of images
    :param file_name: prefix of filename
    :return: None
    '''

    file_name = file_name.split('.')[0]
    image_cntr = 0
    for _image in images:
        file_path = os.path.join(subfolder_path, file_name+str(image_cntr)+'.jpg')
        scipy.misc.imsave(file_path, _image)
        image_cntr += 1


def is_file_extended(path, img, list_of_ready_images):
    if len(list_of_ready_images) == 0:
        list_of_ready_images = get_extended_files(path, list_of_ready_images)
    file_to_check = img.split('.')[0]+'0'+'.jpg'
    if file_to_check in list_of_ready_images:
        return True
    return False


def get_extended_files(path, list_of_ready_images):
    list_of_subfolders = os.listdir(path)
    for subfolder in list_of_subfolders:
        list_of_ready_images += os.listdir(os.path.join(path, subfolder))
    return list_of_ready_images


def get_destination_subfolder(path):
    subfolders = os.listdir(path)
    subfolders.sort(key=int)
    last_subfolder_path = os.path.join(path, subfolders[-1])
    number_of_files = os.listdir(last_subfolder_path)
    if len(number_of_files) < MAX_FILES_IN_SUBFOLDER:
        return subfolders[-1]
    else:
        return str(int(subfolders[-1])+1)


if __name__ == '__main__':
    input_path = r'C:\Downloads\person_detection\ready_img_all\mountains'
    ready_img_cntr = 0
    list_of_images = os.listdir(input_path)
    list_of_ready_images = []

    for image_file in list_of_images:
        if not is_file_extended(DESTINATION_FOLDER, image_file, list_of_ready_images):
            print 'image:', image_file
            destinations_subfolder = get_destination_subfolder(DESTINATION_FOLDER)
            destinations_subfolder_path = os.path.join(DESTINATION_FOLDER, destinations_subfolder)
            if not os.path.exists(destinations_subfolder_path):
                os.mkdir(destinations_subfolder_path)

            image_file_path = os.path.join(input_path, image_file)
            image = scipy.misc.imread(image_file_path)
            multiplied_images = []
            for i in xrange(0, IMAGE_MULTIPLIER, 2):
                for j in xrange(0, IMAGE_MULTIPLIER, 2):
                    multiplied_images.append(image[i:i+DESIRED_DIMENSION, j:j+DESIRED_DIMENSION])
            save_images(multiplied_images, file_name=image_file,
                        subfolder_path=destinations_subfolder_path)

        else:
            ready_img_cntr += 1
            print 'images already transformed: ', ready_img_cntr



