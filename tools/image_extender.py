import os.path
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

DESIRED_DIMENSION = 233
IMAGE_MULTIPLIER = 23
DESTINATION_FOLDER = r'C:\Downloads\person_detection\ready_img_ext\men'
BUCKED = r'C:\Downloads\person_detection\ready_img_ext\bucket'


def save_images(images, file_name):

    '''
    :param images: list of images
    :param file_name: prefix of filename
    :return: None
    '''

    file_name = file_name.split('.')[0]
    image_cntr = 0
    for _image in images:
        file_path = os.path.join(BUCKED, file_name+str(image_cntr)+'.jpg')
        scipy.misc.imsave(file_path, _image)
        image_cntr += 1


if __name__ == '__main__':
    path = r'C:\Downloads\person_detection\ready_img_all\men'
    list_of_images = os.listdir(path)
    ready_img_cntr = 0
    for image_file in list_of_images:
        saved_file = image_file.split('.')[0]+'0'+'.jpg'
        if not os.path.isfile(os.path.join(DESTINATION_FOLDER, saved_file)):
            print 'image:', image_file
            image_file_path = os.path.join(path, image_file)
            image = scipy.misc.imread(image_file_path)
            multiplied_images = []
            for i in xrange(IMAGE_MULTIPLIER):
                for j in xrange(IMAGE_MULTIPLIER):
                    multiplied_images.append(image[i:i+DESIRED_DIMENSION, j:j+DESIRED_DIMENSION])
            save_images(multiplied_images, file_name=image_file)

        else:
            ready_img_cntr += 1
            print 'images already transformed: ', ready_img_cntr



