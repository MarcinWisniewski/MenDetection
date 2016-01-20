import os.path
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

EXPECTED_MINIMAL_DIMENSION = 256
DESTINATION_FOLDER = r'C:\Downloads\person_detection\ready_img\mountains'


def prepare_image(_image, edge, origin='begin'):
    if origin == 'end':
        cropped_image = _image[0:edge, 0:edge]
    else:
        cropped_image = _image[-edge: -1, -edge: -1]

    return resize_image(cropped_image)


def resize_image(_image):
    return scipy.misc.imresize(_image, (EXPECTED_MINIMAL_DIMENSION, EXPECTED_MINIMAL_DIMENSION))


def save_images(images, file_name):
    file_name = file_name.split('.')[0]
    image_cntr = 0
    for _image in images:
        file_path = os.path.join(DESTINATION_FOLDER, file_name+str(image_cntr)+'.jpg')
        scipy.misc.imsave(file_path, _image)
        image_cntr += 1


if __name__ == '__main__':
    path = r'C:\Downloads\mountains'
    list_of_images = os.listdir(path)
    for image_file in list_of_images:
        print 'image:', image_file
        image_file_path = os.path.join(path, image_file)
        image = scipy.misc.imread(image_file_path)
        shape = image.shape
        print shape
        flipped_image = np.fliplr(image)
        shorter_edge = min(shape[:2])
        longer_edge = max(shape[:2])
        ratio = longer_edge/float(shorter_edge)
        print 'shorter edge: %s, longer edge: %s, ratio: %s' % (shorter_edge, longer_edge, ratio)
        if shorter_edge >= EXPECTED_MINIMAL_DIMENSION:
            cropped_images = []
            cropped_images.append(prepare_image(image, shorter_edge, origin='begin'))
            cropped_images.append(prepare_image(flipped_image, shorter_edge, origin='begin'))
            if ratio >= 1.25:
                cropped_images.append(prepare_image(image, shorter_edge, origin='end'))
                cropped_images.append(prepare_image(flipped_image, shorter_edge, origin='end'))
            save_images(cropped_images, file_name=image_file)


            #f, axarr = plt.subplots(3, 2)
            #axarr[0][0].imshow(image)
            #axarr[0][1].imshow(flipped_image)
            #axarr[1][0].imshow(cropped_image_left)
            #axarr[1][1].imshow(cropped_flipped_image_left)
            #axarr[2][0].imshow(cropped_image_right)
            #axarr[2][1].imshow(cropped_flipped_image_right)
            #plt.show()
        else:
            print 'image is to small'
