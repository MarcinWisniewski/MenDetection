
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
import argparse
import pickle as cPickle
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from CNN.conv_network import CNN

IMAGE_SIZE = 233
STEP = IMAGE_SIZE/3
N_KERNELS = (96, 256, 128, 128, 64)


def start_testing():

    parser = argparse.ArgumentParser(description='Path to image')
    parser.add_argument('path', help='path to analysed image')
    args = parser.parse_args()

    image = misc.imread(args.path)
    shape_y, shape_x, channel = image.shape

    cutted_images = cut_image(image, shape_x, shape_y)

    rng = np.random.RandomState(234555)
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images

    print '... building the model'
    cnn = CNN(rng, x, N_KERNELS)

    prediction = lasagne.layers.get_output(cnn.network, deterministic=True)

    model = theano.function([x], prediction)

    if os.path.isfile('model.bin'):
        print 'using model'
        f = open('model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()

    predict = model(cutted_images)
    predict = predict[:, 1]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(image)
    prediction_index = 0
    for x_pixel in xrange(0, shape_x-IMAGE_SIZE, STEP):
        for y_pixel in xrange(0, shape_y-IMAGE_SIZE, STEP):
            if predict[prediction_index] > 0.5:
                ax1.add_patch(patches.Rectangle((x_pixel, y_pixel),
                                                IMAGE_SIZE, IMAGE_SIZE,
                                                linewidth=(predict[prediction_index]-0.49)*5.0,
                                                fill=False))
            prediction_index += 1

    plt.show()


def cut_image(image, shape_x, shape_y):
    cutted_images = []
    for x_pixel in xrange(0, shape_x - IMAGE_SIZE, STEP):
        for y_pixel in xrange(0, shape_y - IMAGE_SIZE, STEP):
            temp_img = image[y_pixel:y_pixel + IMAGE_SIZE, x_pixel:x_pixel + IMAGE_SIZE]
            temp_img = np.asarray(temp_img / (256.0, 256.0, 256.0), dtype=theano.config.floatX)
            mean_image = np.mean(temp_img, axis=(0, 1), dtype='float')
            temp_img -= mean_image
            temp_img = temp_img.transpose(2, 0, 1)
            cutted_images.append(temp_img)
    cutted_images = np.asarray(cutted_images, dtype=theano.config.floatX)
    return cutted_images


if __name__ == '__main__':
    start_testing()

