
import os
import sys
import timeit
import time
import datetime
import numpy
import theano
import theano.tensor as T
import lasagne
import pickle as cPickle
from Readers.data_provider import DataProvider
from CNN.conv_network import CNN


def start_testing(n_kerns=(96, 256, 128, 128, 64), batch_size=512, reduce_training_set=True):
    """
    :type n_kerns: list of ints
    :param n_kerns: number of kernels on each layer

    :type batch_size: int
    :param batch_size: number of examples in minibatch

    :type reduce_training_set: bool
    :param reduce_training_set: debugging switch, reduce training set to check convergence of CNN
    """

    actual_time = datetime.datetime.now().time()
    print 'algorithm started at: %r' % actual_time.isoformat()

    rng = numpy.random.RandomState(234555)
    dp = DataProvider(
        input_dir='/home/marcin/data/men_detection',
        test_percentage_split=0.0001, validate_percentage_split=0.0001,
        batch=batch_size, reduce_training_set=reduce_training_set)

    print 'number of all images - %i' % dp.get_number_of_all_images()
    print 'number of training images - %i' % dp.get_number_of_trainig_images()

    # start-snippet-1
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.vector('y', dtype='int64')                 # the labels are presented as 1D vector of
                                                     # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    cnn = CNN(rng, x, n_kerns)

    test_prediction = lasagne.layers.get_output(cnn.network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                      dtype=theano.config.floatX)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    validate_model = theano.function([x, y], [test_loss, test_acc])

    ###############
    # TRAIN MODEL #
    ###############
    if os.path.isfile('model.bin'):
        print 'using model'
        f = open('model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()
    print '... testing'
    n_valid_batches = dp.get_number_of_training_batches()
    validation_losses = []
    validation_acc = []
    print 'number of batches - %i' % n_valid_batches

    for valid_batch in xrange(n_valid_batches):
        batch_valid_set_x, batch_valid_set_y = dp.get_batch_training_images()
        if batch_valid_set_x is not None and batch_valid_set_y is not None:
            tic = time.time()
            err, acc = validate_model(batch_valid_set_x, batch_valid_set_y)
            print '%i minibatch runs %f seconds and has %f %% mean error and %f %% mean acc' % \
                  (valid_batch, (time.time() - tic), numpy.mean(err) * 100, numpy.mean(acc) * 100)
            validation_losses.append(err)
            validation_acc.append(acc)

    this_validation_loss = numpy.mean(validation_losses)
    mean_validation_acc = numpy.mean(validation_acc)
    print 'validation error %f %% and accuracy  %f %%' % \
          (this_validation_loss * 100., mean_validation_acc * 100)


if __name__ == '__main__':
    start_testing()

